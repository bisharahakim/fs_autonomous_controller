from __future__ import annotations

import sys
from dataclasses import dataclass
from math import atan, cos, hypot, sin, tan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fs_controller import (  # noqa: E402
    MAX_SPEED_30_KMH_MPS,
    PowertrainConfig,
    PowertrainModel,
    VehicleState,
    apply_friction_circle,
    requested_acceleration_from_actuators,
)

try:
    from autocross_track import TrackPoint  # noqa: E402
    from track_registry import get_track  # noqa: E402
except ModuleNotFoundError:
    from visualization.autocross_track import TrackPoint  # noqa: E402
    from visualization.track_registry import get_track  # noqa: E402


TRACK_NAME = "hockenheim_fsg"
MAX_STEPS = 12000


@dataclass(frozen=True)
class Sample:
    time_s: float
    x_m: float
    y_m: float
    speed_mps: float
    target_speed_mps: float
    steering_rad: float
    throttle: float
    brake: float
    requested_accel_mps2: float
    actual_accel_mps2: float
    target_index: int
    lateral_error_m: float
    lookahead_m: float


@dataclass
class ClosedLoopControllerState:
    target_index: int = 0
    integrator: float = 0.0


def make_track() -> list[TrackPoint]:
    return remove_duplicate_finish(get_track(TRACK_NAME).build_centerline())


def simulate() -> list[Sample]:
    wheelbase = 1.55
    path = make_track()
    controller = ClosedLoopControllerState()
    powertrain = PowertrainModel()
    vehicle_config = PowertrainConfig()
    state = VehicleState(
        x=path[0].x - 1.2 * cos(path[0].yaw),
        y=path[0].y - 0.15,
        yaw=path[0].yaw,
        speed=0.0,
    )
    dt = 0.02
    samples: list[Sample] = []

    for step in range(MAX_STEPS):
        previous_target_index = controller.target_index
        high = pure_pursuit(state, path, controller, wheelbase)
        low = speed_pi(
            target_speed_mps=high["target_speed"],
            measured_speed_mps=state.speed,
            dt_s=dt,
            controller=controller,
        )
        requested_accel = requested_acceleration_from_actuators(low["throttle"], low["brake"])
        powertrain_accel = powertrain.actual_acceleration(state.speed, requested_accel)
        speed_before_lateral = max(0.0, state.speed + powertrain_accel * dt)
        commanded_lateral_accel = speed_before_lateral * speed_before_lateral * tan(high["steering"]) / wheelbase
        actual_long_force, actual_lateral_force, _ = apply_friction_circle(
            vehicle_config.mass_kg * powertrain_accel,
            vehicle_config.mass_kg * commanded_lateral_accel,
            vehicle_config.mass_kg,
        )
        actual_accel = actual_long_force / vehicle_config.mass_kg
        achievable_lateral_accel = actual_lateral_force / vehicle_config.mass_kg
        speed = min(MAX_SPEED_30_KMH_MPS, max(0.0, state.speed + actual_accel * dt))
        yaw_rate = achievable_lateral_accel / max(speed, 1e-3)
        yaw = state.yaw + yaw_rate * dt
        x = state.x + speed * cos(yaw) * dt
        y = state.y + speed * sin(yaw) * dt
        state = VehicleState(x=x, y=y, yaw=yaw, speed=speed)

        samples.append(
            Sample(
                time_s=step * dt,
                x_m=state.x,
                y_m=state.y,
                speed_mps=state.speed,
                target_speed_mps=high["target_speed"],
                steering_rad=high["steering"],
                throttle=low["throttle"],
                brake=low["brake"],
                requested_accel_mps2=requested_accel,
                actual_accel_mps2=actual_accel,
                target_index=int(high["target_index"]),
                lateral_error_m=nearest_path_distance(state, path),
                lookahead_m=high["lookahead"],
            )
        )

        completed_lap = (
            previous_target_index > len(path) - 80
            and int(high["target_index"]) < 80
            and step * dt > 5.0
        )
        if completed_lap:
            break

    return samples


def pure_pursuit(
    state: VehicleState,
    path: list[TrackPoint],
    controller: ClosedLoopControllerState,
    wheelbase_m: float,
) -> dict[str, float]:
    nearest = nearest_index(state, path, controller.target_index)
    lookahead = clamp(3.0 + 0.2 * state.speed, 3.0, 18.0)
    target_index = nearest

    for offset in range(len(path)):
        index = (nearest + offset) % len(path)
        point = path[index]
        if hypot(point.x - state.x, point.y - state.y) >= lookahead:
            target_index = index
            break

    controller.target_index = target_index
    target = path[target_index]
    dx = target.x - state.x
    dy = target.y - state.y
    local_x = cos(state.yaw) * dx + sin(state.yaw) * dy
    local_y = -sin(state.yaw) * dx + cos(state.yaw) * dy
    distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
    curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
    steering = clamp(atan(wheelbase_m * curvature), -0.5, 0.5)

    return {
        "steering": steering,
        "target_speed": target.speed,
        "target_index": float(target_index),
        "lookahead": lookahead,
    }


def speed_pi(
    target_speed_mps: float,
    measured_speed_mps: float,
    dt_s: float,
    controller: ClosedLoopControllerState,
) -> dict[str, float]:
    error = target_speed_mps - measured_speed_mps
    controller.integrator = clamp(controller.integrator + error * dt_s, -3.0, 3.0)
    command = clamp(0.35 * error + 0.12 * controller.integrator, -1.0, 1.0)
    return {
        "throttle": clamp(command, 0.0, 1.0),
        "brake": clamp(-1.3 * command, 0.0, 1.0),
    }


def nearest_index(state: VehicleState, path: list[TrackPoint], start_index: int) -> int:
    best_index = start_index
    best_distance = float("inf")
    for offset in range(len(path)):
        index = (start_index + offset) % len(path)
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


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def save_svg(samples: list[Sample], path: list[TrackPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 1220
    margin_left = 86
    margin_right = 34
    map_height = 300
    panel_height = 120
    panel_gap = 34
    top = 78
    plot_width = width - margin_left - margin_right
    time_panel_top = top + map_height + 56

    panels = [
        {
            "title": "Speed Tracking",
            "unit": "m/s",
            "min": 0.0,
            "max": 24.0,
            "series": [
                ("Actual speed", "#218c5a", [s.speed_mps for s in samples]),
                ("Target speed", "#6b7280", [s.target_speed_mps for s in samples]),
            ],
        },
        {
            "title": "Centerline Tracking Error",
            "unit": "m",
            "min": 0.0,
            "max": 2.0,
            "series": [
                ("Lateral error", "#9333ea", [s.lateral_error_m for s in samples]),
            ],
        },
        {
            "title": "Steering Command",
            "unit": "rad",
            "min": -0.55,
            "max": 0.55,
            "series": [("Steering", "#c2413b", [s.steering_rad for s in samples])],
        },
        {
            "title": "Longitudinal Actuators",
            "unit": "normalized",
            "min": 0.0,
            "max": 1.0,
            "series": [
                ("Throttle", "#2563eb", [s.throttle for s in samples]),
                ("Brake", "#d97706", [s.brake for s in samples]),
            ],
        },
        {
            "title": "Lookahead Distance",
            "unit": "m",
            "min": 0.0,
            "max": 26.0,
            "series": [
                ("Lookahead", "#0f766e", [s.lookahead_m for s in samples]),
            ],
        },
    ]

    time_values = [s.time_s for s in samples]
    time_min = min(time_values)
    time_max = max(time_values)

    def sx(time_s: float) -> float:
        return margin_left + plot_width * (time_s - time_min) / (time_max - time_min)

    def sy(value: float, panel: dict[str, object], y0: float) -> float:
        vmin = float(panel["min"])
        vmax = float(panel["max"])
        normalized = (value - vmin) / (vmax - vmin)
        normalized = max(0.0, min(1.0, normalized))
        return y0 + panel_height * (1.0 - normalized)

    def path_for(values: list[float], panel: dict[str, object], y0: float) -> str:
        coords = []
        for time_s, value in zip(time_values, values):
            command = "M" if not coords else "L"
            coords.append(f"{command}{sx(time_s):.2f},{sy(value, panel, y0):.2f}")
        return " ".join(coords)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="34" y="38" font-family="Arial" font-size="26" font-weight="700" fill="#111827">Hockenheim FSG Controller Static Data</text>',
        f'<text x="34" y="62" font-family="Arial" font-size="13" fill="#4b5563">Samples: {len(samples)} | Duration: {time_max:.2f} s | Max tracking error: {max(s.lateral_error_m for s in samples):.2f} m | Average tracking error: {sum(s.lateral_error_m for s in samples) / len(samples):.2f} m</text>',
    ]

    svg.extend(trajectory_panel(samples, path, margin_left, top, plot_width, map_height))

    for panel_index, panel in enumerate(panels):
        y0 = time_panel_top + panel_index * (panel_height + panel_gap)
        svg.extend(
            [
                f'<rect x="{margin_left}" y="{y0}" width="{plot_width}" height="{panel_height}" fill="#ffffff" stroke="#d1d5db"/>',
                f'<text x="34" y="{y0 + 18}" font-family="Arial" font-size="15" font-weight="700" fill="#111827">{panel["title"]}</text>',
                f'<text x="34" y="{y0 + 38}" font-family="Arial" font-size="12" fill="#6b7280">{panel["unit"]}</text>',
            ]
        )

        for grid_index in range(5):
            gy = y0 + panel_height * grid_index / 4
            value = float(panel["max"]) - (float(panel["max"]) - float(panel["min"])) * grid_index / 4
            svg.append(
                f'<line x1="{margin_left}" y1="{gy:.2f}" x2="{margin_left + plot_width}" y2="{gy:.2f}" stroke="#e5e7eb"/>'
            )
            svg.append(
                f'<text x="{margin_left - 10}" y="{gy + 4:.2f}" font-family="Arial" font-size="11" text-anchor="end" fill="#6b7280">{value:.2f}</text>'
            )

        for tick_index in range(6):
            tx_value = time_min + (time_max - time_min) * tick_index / 5
            tx = sx(tx_value)
            svg.append(
                f'<line x1="{tx:.2f}" y1="{y0}" x2="{tx:.2f}" y2="{y0 + panel_height}" stroke="#f1f5f9"/>'
            )
            if panel_index == len(panels) - 1:
                svg.append(
                    f'<text x="{tx:.2f}" y="{y0 + panel_height + 22}" font-family="Arial" font-size="11" text-anchor="middle" fill="#6b7280">{tx_value:.1f}s</text>'
                )

        legend_x = margin_left + 14
        for series_index, (name, color, values) in enumerate(panel["series"]):
            path_data = path_for(values, panel, y0)
            dash = ' stroke-dasharray="8 6"' if "Target" in name else ""
            svg.append(
                f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.4"{dash}/>'
            )
            lx = legend_x + series_index * 150
            svg.append(f'<line x1="{lx}" y1="{y0 + 18}" x2="{lx + 22}" y2="{y0 + 18}" stroke="{color}" stroke-width="3"{dash}/>')
            svg.append(
                f'<text x="{lx + 28}" y="{y0 + 22}" font-family="Arial" font-size="12" fill="#374151">{name}</text>'
            )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def trajectory_panel(
    samples: list[Sample],
    path: list[TrackPoint],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    all_x = [point.x for point in path] + [sample.x_m for sample in samples]
    all_y = [point.y for point in path] + [sample.y_m for sample in samples]
    min_x = min(all_x) - 3.0
    max_x = max(all_x) + 3.0
    min_y = min(all_y) - 3.0
    max_y = max(all_y) + 3.0
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

    centerline = [(point.x, point.y) for point in path]
    vehicle = [(sample.x_m, sample.y_m) for sample in samples]
    half_track_px = 1.5 * scale

    return [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>',
        f'<text x="34" y="{y0 + 18}" font-family="Arial" font-size="15" font-weight="700" fill="#111827">Trajectory Map</text>',
        f'<text x="34" y="{y0 + 38}" font-family="Arial" font-size="12" fill="#6b7280">Reference path and simulated vehicle path</text>',
        f'<polyline points="{points_attr(centerline)}" fill="none" stroke="#d1d5db" stroke-width="{2 * half_track_px:.2f}" stroke-linecap="round" stroke-linejoin="round" opacity="0.55"/>',
        f'<polyline points="{points_attr(centerline)}" fill="none" stroke="#111827" stroke-width="2.0" stroke-dasharray="7 7"/>',
        f'<polyline points="{points_attr(vehicle)}" fill="none" stroke="#218c5a" stroke-width="3.0"/>',
        f'<line x1="{x0 + width - 250}" y1="{y0 + 24}" x2="{x0 + width - 220}" y2="{y0 + 24}" stroke="#111827" stroke-width="2" stroke-dasharray="7 7"/>',
        f'<text x="{x0 + width - 212}" y="{y0 + 28}" font-family="Arial" font-size="12" fill="#374151">reference centerline</text>',
        f'<line x1="{x0 + width - 250}" y1="{y0 + 46}" x2="{x0 + width - 220}" y2="{y0 + 46}" stroke="#218c5a" stroke-width="3"/>',
        f'<text x="{x0 + width - 212}" y="{y0 + 50}" font-family="Arial" font-size="12" fill="#374151">vehicle trajectory</text>',
        f'<rect x="{x0 + width - 250}" y="{y0 + 62}" width="30" height="12" fill="#d1d5db" opacity="0.55"/>',
        f'<text x="{x0 + width - 212}" y="{y0 + 73}" font-family="Arial" font-size="12" fill="#374151">3 m cone corridor</text>',
    ]


def main() -> None:
    path = make_track()
    samples = simulate()
    output_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / f"{TRACK_NAME}_controller_static_graph.svg"
    )
    save_svg(samples, path, output_path)
    print(f"Saved static graph to {output_path}")


def remove_duplicate_finish(path: list[TrackPoint]) -> list[TrackPoint]:
    if len(path) < 2:
        return path
    start = path[0]
    finish = path[-1]
    if hypot(finish.x - start.x, finish.y - start.y) < 1e-6:
        return path[:-1]
    return path


if __name__ == "__main__":
    main()
