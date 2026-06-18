from __future__ import annotations

import sys
from dataclasses import dataclass
from math import atan, cos, hypot, sin, sqrt, tan
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fs_controller import (  # noqa: E402
    MAX_SPEED_30_KMH_MPS,
    PIConfig,
    PowertrainConfig,
    PowertrainModel,
    PurePursuitConfig,
    SpeedPIController,
    VehicleState,
    apply_friction_circle,
    requested_acceleration_from_actuators,
)
from visualization.autocross_track import TrackPoint  # noqa: E402
from visualization.track_registry import get_track  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    color: str
    config: PurePursuitConfig


@dataclass(frozen=True)
class Sample:
    time_s: float
    x_m: float
    y_m: float
    speed_mps: float
    target_speed_mps: float
    tracking_error_m: float
    steering_rad: float
    throttle: float
    brake: float
    requested_accel_mps2: float
    actual_accel_mps2: float
    lookahead_m: float
    target_index: int


@dataclass(frozen=True)
class Metrics:
    duration_s: float
    max_error_m: float
    avg_error_m: float
    rms_error_m: float
    final_error_m: float
    max_abs_steering_rad: float
    avg_abs_steering_rad: float
    max_speed_mps: float


WHEELBASE_M = 1.55
DT_S = 0.02
MAX_STEPS = 12000
TRACK_NAME = "hockenheim_fsg"

SPEED_PI_CONFIG = PIConfig(
    kp=0.35,
    ki=0.12,
    output_min=-1.0,
    output_max=1.0,
    integrator_min=-3.0,
    integrator_max=3.0,
)

VARIANTS = [
    Variant(
        name="Regular PP - fixed 5.0 m lookahead",
        color="#2563eb",
        config=PurePursuitConfig(
            wheelbase_m=WHEELBASE_M,
            min_lookahead_m=5.0,
            lookahead_gain_s=0.0,
            max_lookahead_m=5.0,
            max_steer_rad=0.5,
            default_target_speed_mps=8.0,
            finish_tolerance_m=2.0,
        ),
    ),
    Variant(
        name="Adaptive PP - 2.5 m + 0.18 s * speed",
        color="#218c5a",
        config=PurePursuitConfig(
            wheelbase_m=WHEELBASE_M,
            min_lookahead_m=2.5,
            lookahead_gain_s=0.18,
            max_lookahead_m=10.0,
            max_steer_rad=0.5,
            default_target_speed_mps=8.0,
            finish_tolerance_m=2.0,
        ),
    ),
]


def main() -> None:
    track = remove_duplicate_finish(get_track(TRACK_NAME).build_centerline())
    results = {variant: simulate_variant(track, variant) for variant in VARIANTS}
    output_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / f"{TRACK_NAME}_pure_pursuit_comparison.svg"
    )
    save_svg(track, results, output_path)
    print(f"Saved pure pursuit comparison graph to {output_path}")


def remove_duplicate_finish(track: list[TrackPoint]) -> list[TrackPoint]:
    if len(track) < 2:
        return track
    start = track[0]
    finish = track[-1]
    if hypot(finish.x - start.x, finish.y - start.y) < 1e-6:
        return track[:-1]
    return track


def simulate_variant(
    track: list[TrackPoint],
    variant: Variant,
) -> list[Sample]:
    speed_controller = SpeedPIController(SPEED_PI_CONFIG, brake_gain=1.3)
    powertrain = PowertrainModel()
    vehicle_config = PowertrainConfig()

    first = track[0]
    state = VehicleState(first.x - 1.2 * cos(first.yaw), first.y - 0.15, first.yaw, 0.0)
    samples: list[Sample] = []
    last_target_index = 0

    for step in range(MAX_STEPS):
        previous_target_index = last_target_index
        high = pure_pursuit_closed(state, track, last_target_index, variant.config)
        last_target_index = high["target_index"]
        low = speed_controller.update(high["target_speed"], state.speed, DT_S)

        requested_accel = requested_acceleration_from_actuators(low.throttle, low.brake)
        powertrain_accel = powertrain.actual_acceleration(state.speed, requested_accel)
        speed_before_lateral = max(0.0, state.speed + powertrain_accel * DT_S)
        commanded_lateral_accel = (
            speed_before_lateral * speed_before_lateral * tan(high["steering"]) / WHEELBASE_M
        )
        actual_long_force, actual_lateral_force, _ = apply_friction_circle(
            vehicle_config.mass_kg * powertrain_accel,
            vehicle_config.mass_kg * commanded_lateral_accel,
            vehicle_config.mass_kg,
        )
        actual_accel = actual_long_force / vehicle_config.mass_kg
        achievable_lateral_accel = actual_lateral_force / vehicle_config.mass_kg
        speed = min(MAX_SPEED_30_KMH_MPS, max(0.0, state.speed + actual_accel * DT_S))
        yaw_rate = achievable_lateral_accel / max(speed, 1e-3)
        yaw = state.yaw + yaw_rate * DT_S
        state = VehicleState(
            x=state.x + speed * cos(yaw) * DT_S,
            y=state.y + speed * sin(yaw) * DT_S,
            yaw=yaw,
            speed=speed,
        )

        samples.append(
            Sample(
                time_s=step * DT_S,
                x_m=state.x,
                y_m=state.y,
                speed_mps=state.speed,
                target_speed_mps=high["target_speed"],
                tracking_error_m=nearest_path_distance(state, track),
                steering_rad=high["steering"],
                throttle=low.throttle,
                brake=low.brake,
                requested_accel_mps2=requested_accel,
                actual_accel_mps2=actual_accel,
                lookahead_m=high["lookahead"],
                target_index=last_target_index,
            )
        )

        completed_lap = (
            previous_target_index > len(track) - 80
            and last_target_index < 80
            and step * DT_S > 5.0
        )
        if completed_lap:
            break

    return samples


def pure_pursuit_closed(
    state: VehicleState,
    path: list[TrackPoint],
    last_target_index: int,
    config: PurePursuitConfig,
) -> dict[str, float | int]:
    nearest = nearest_index_closed(state, path, last_target_index)
    lookahead = clamp(
        config.min_lookahead_m + config.lookahead_gain_s * state.speed,
        config.min_lookahead_m,
        config.max_lookahead_m,
    )
    target_index = nearest
    for offset in range(len(path)):
        index = (nearest + offset) % len(path)
        point = path[index]
        if hypot(point.x - state.x, point.y - state.y) >= lookahead:
            target_index = index
            break

    target = path[target_index]
    dx = target.x - state.x
    dy = target.y - state.y
    local_x = cos(state.yaw) * dx + sin(state.yaw) * dy
    local_y = -sin(state.yaw) * dx + cos(state.yaw) * dy
    distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
    curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
    steering = clamp(atan(WHEELBASE_M * curvature), -config.max_steer_rad, config.max_steer_rad)
    target_speed = target.speed if target.speed is not None else config.default_target_speed_mps
    return {
        "steering": steering,
        "target_speed": max(0.0, target_speed),
        "target_index": target_index,
        "lookahead": lookahead,
    }


def nearest_index_closed(state: VehicleState, path: list[TrackPoint], last_target_index: int) -> int:
    best_index = last_target_index
    best_distance = float("inf")
    for offset in range(len(path)):
        index = (last_target_index + offset) % len(path)
        point = path[index]
        distance = (state.x - point.x) ** 2 + (state.y - point.y) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = index
        if offset > 180 and distance > best_distance * 6.0:
            break
    return best_index


def nearest_path_distance(state: VehicleState, path: list[TrackPoint]) -> float:
    return min(hypot(state.x - point.x, state.y - point.y) for point in path)


def calculate_metrics(samples: list[Sample]) -> Metrics:
    errors = [sample.tracking_error_m for sample in samples]
    steering = [abs(sample.steering_rad) for sample in samples]
    return Metrics(
        duration_s=samples[-1].time_s,
        max_error_m=max(errors),
        avg_error_m=sum(errors) / len(errors),
        rms_error_m=sqrt(sum(error * error for error in errors) / len(errors)),
        final_error_m=errors[-1],
        max_abs_steering_rad=max(steering),
        avg_abs_steering_rad=sum(steering) / len(steering),
        max_speed_mps=max(sample.speed_mps for sample in samples),
    )


def save_svg(
    track: list[TrackPoint],
    results: dict[Variant, list[Sample]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1320
    height = 1450
    left = 92
    right = 40
    plot_width = width - left - right

    svg = [
        svg_open(width, height),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        text(34, 38, "Pure Pursuit Controller Comparison", 26, "#111827", 700),
        text(
            34,
            62,
            "Same Hockenheim FSG closed lap, same speed controller and vehicle model. Only lookahead behavior changes.",
            13,
            "#4b5563",
        ),
    ]

    svg.extend(draw_trajectory_panel(track, results, 34, 86, 1252, 365))
    svg.extend(draw_metric_table(results, 34, 478, 1252, 140))
    svg.extend(
        draw_time_panel(
            results,
            title="Lateral Tracking Error",
            unit="m",
            value_range=(0.0, 2.2),
            series_for=lambda samples: [("Tracking error", None, [s.tracking_error_m for s in samples], False)],
            x0=left,
            y0=665,
            width=plot_width,
            height=135,
        )
    )
    svg.extend(
        draw_time_panel(
            results,
            title="Lookahead Distance",
            unit="m",
            value_range=(0.0, 9.0),
            series_for=lambda samples: [("Lookahead", None, [s.lookahead_m for s in samples], False)],
            x0=left,
            y0=845,
            width=plot_width,
            height=135,
        )
    )
    svg.extend(
        draw_time_panel(
            results,
            title="Steering Command",
            unit="rad",
            value_range=(-0.55, 0.55),
            series_for=lambda samples: [("Steering", None, [s.steering_rad for s in samples], False)],
            x0=left,
            y0=1025,
            width=plot_width,
            height=135,
        )
    )
    svg.extend(
        draw_time_panel(
            results,
            title="Speed Tracking",
            unit="m/s",
            value_range=(0.0, 24.0),
            series_for=lambda samples: [
                ("Actual speed", None, [s.speed_mps for s in samples], False),
                ("Target speed", "#6b7280", [s.target_speed_mps for s in samples], True),
            ],
            x0=left,
            y0=1205,
            width=plot_width,
            height=135,
        )
    )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def draw_trajectory_panel(
    track: list[TrackPoint],
    results: dict[Variant, list[Sample]],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    all_x = [point.x for point in track]
    all_y = [point.y for point in track]
    for samples in results.values():
        all_x.extend(sample.x_m for sample in samples)
        all_y.extend(sample.y_m for sample in samples)

    min_x = min(all_x) - 4.0
    max_x = max(all_x) + 4.0
    min_y = min(all_y) - 4.0
    max_y = max(all_y) + 4.0
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    used_w = (max_x - min_x) * scale
    used_h = (max_y - min_y) * scale
    offset_x = x0 + (width - used_w) / 2.0
    offset_y = y0 + (height - used_h) / 2.0

    def px(x_m: float) -> float:
        return offset_x + (x_m - min_x) * scale

    def py(y_m: float) -> float:
        return offset_y + used_h - (y_m - min_y) * scale

    def points_attr(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)

    centerline = [(point.x, point.y) for point in track]
    svg = [
        rect(x0, y0, width, height),
        text(x0 + 14, y0 + 25, "Trajectory Comparison", 15, "#111827", 700),
        f'<polyline points="{points_attr(centerline)}" fill="none" stroke="#d1d5db" stroke-width="{3.0 * scale:.2f}" stroke-linecap="round" stroke-linejoin="round" opacity="0.50"/>',
        f'<polyline points="{points_attr(centerline)}" fill="none" stroke="#111827" stroke-width="2.0" stroke-dasharray="7 7"/>',
    ]

    legend_x = x0 + width - 335
    legend_y = y0 + 28
    svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" stroke="#111827" stroke-width="2" stroke-dasharray="7 7"/>')
    svg.append(text(legend_x + 34, legend_y + 4, "reference centerline", 12, "#374151"))

    for index, (variant, samples) in enumerate(results.items(), start=1):
        vehicle_path = [(sample.x_m, sample.y_m) for sample in samples]
        svg.append(
            f'<polyline points="{points_attr(vehicle_path)}" fill="none" stroke="{variant.color}" stroke-width="2.8"/>'
        )
        ly = legend_y + index * 24
        svg.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 26}" y2="{ly}" stroke="{variant.color}" stroke-width="3"/>')
        svg.append(text(legend_x + 34, ly + 4, variant.name, 12, "#374151"))

    return svg


def draw_metric_table(
    results: dict[Variant, list[Sample]],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    metrics = {variant: calculate_metrics(samples) for variant, samples in results.items()}
    columns = [
        ("Controller", 0.00),
        ("Max Err", 0.33),
        ("Avg Err", 0.45),
        ("RMS Err", 0.57),
        ("Max |Steer|", 0.69),
        ("Avg |Steer|", 0.81),
        ("Duration", 0.93),
    ]

    svg = [rect(x0, y0, width, height), text(x0 + 14, y0 + 25, "Summary Metrics", 15, "#111827", 700)]
    for label, fraction in columns:
        svg.append(text(x0 + 14 + width * fraction, y0 + 58, label, 12, "#374151", 700))

    for row, (variant, data) in enumerate(metrics.items()):
        y = y0 + 88 + row * 30
        values = [
            variant.name,
            f"{data.max_error_m:.2f} m",
            f"{data.avg_error_m:.2f} m",
            f"{data.rms_error_m:.2f} m",
            f"{data.max_abs_steering_rad:.2f} rad",
            f"{data.avg_abs_steering_rad:.2f} rad",
            f"{data.duration_s:.1f} s",
        ]
        svg.append(f'<circle cx="{x0 + 7}" cy="{y - 4}" r="4" fill="{variant.color}"/>')
        for value, (_, fraction) in zip(values, columns):
            svg.append(text(x0 + 14 + width * fraction, y, value, 12, "#111827"))

    return svg


def draw_time_panel(
    results: dict[Variant, list[Sample]],
    title: str,
    unit: str,
    value_range: tuple[float, float],
    series_for: Callable[[list[Sample]], list[tuple[str, str | None, list[float], bool]]],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    value_min, value_max = value_range
    plot_top = y0 + 24
    plot_bottom = y0 + height
    plot_h = plot_bottom - plot_top
    max_time = max(samples[-1].time_s for samples in results.values())

    def sx(time_s: float) -> float:
        return x0 + width * time_s / max_time

    def sy(value: float) -> float:
        normalized = clamp((value - value_min) / (value_max - value_min), 0.0, 1.0)
        return plot_top + plot_h * (1.0 - normalized)

    svg = [
        text(34, y0 + 20, title, 15, "#111827", 700),
        text(34, y0 + 40, unit, 12, "#6b7280"),
        rect(x0, plot_top, width, plot_h),
    ]

    for grid_index in range(5):
        gy = plot_top + plot_h * grid_index / 4
        value = value_max - (value_max - value_min) * grid_index / 4
        svg.append(f'<line x1="{x0}" y1="{gy:.2f}" x2="{x0 + width}" y2="{gy:.2f}" stroke="#e5e7eb"/>')
        svg.append(text(x0 - 10, gy + 4, f"{value:.2f}", 11, "#6b7280", anchor="end"))

    for tick_index in range(6):
        tx_value = max_time * tick_index / 5
        tx = sx(tx_value)
        svg.append(f'<line x1="{tx:.2f}" y1="{plot_top}" x2="{tx:.2f}" y2="{plot_bottom}" stroke="#f1f5f9"/>')
        svg.append(text(tx, plot_bottom + 22, f"{tx_value:.1f}s", 11, "#6b7280", anchor="middle"))

    legend_x = x0 + 14
    legend_y = plot_top + 18
    legend_index = 0
    for variant, samples in results.items():
        time_values = [sample.time_s for sample in samples]
        for label, color_override, values, dashed in series_for(samples):
            color = color_override or variant.color
            path_data = make_path_data(time_values, values, sx, sy)
            dash = ' stroke-dasharray="7 6"' if dashed else ""
            svg.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.3"{dash}/>')
            if color_override is None:
                legend_label = variant.name
            else:
                legend_label = label
            lx = legend_x + legend_index * 290
            svg.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dash}/>')
            svg.append(text(lx + 32, legend_y + 4, legend_label, 12, "#374151"))
            legend_index += 1

    return svg


def make_path_data(
    time_values: list[float],
    values: list[float],
    sx: Callable[[float], float],
    sy: Callable[[float], float],
) -> str:
    coords = []
    for time_s, value in zip(time_values, values):
        command = "M" if not coords else "L"
        coords.append(f"{command}{sx(time_s):.2f},{sy(value):.2f}")
    return " ".join(coords)


def svg_open(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def rect(x: float, y: float, width: float, height: float) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>'


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    color: str,
    weight: int | None = None,
    anchor: str | None = None,
) -> str:
    weight_attr = f' font-weight="{weight}"' if weight is not None else ""
    anchor_attr = f' text-anchor="{anchor}"' if anchor is not None else ""
    return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" fill="{color}"{weight_attr}{anchor_attr}>{escape_xml(value)}</text>'


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


if __name__ == "__main__":
    main()
