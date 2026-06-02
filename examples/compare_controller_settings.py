from __future__ import annotations

import sys
from dataclasses import dataclass
from math import atan, cos, hypot, sin, tan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fs_controller import VehicleState  # noqa: E402
from visualization.autocross_track import TrackPoint, build_fsd_autocross_track  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    color: str
    min_lookahead_m: float
    lookahead_gain_s: float
    max_lookahead_m: float
    kp: float
    ki: float


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


@dataclass
class SimState:
    target_index: int = 0
    integrator: float = 0.0


VARIANTS = [
    Variant(
        name="Baseline",
        color="#218c5a",
        min_lookahead_m=3.0,
        lookahead_gain_s=0.20,
        max_lookahead_m=18.0,
        kp=0.35,
        ki=0.12,
    ),
    Variant(
        name="Short Lookahead",
        color="#d97706",
        min_lookahead_m=2.0,
        lookahead_gain_s=0.12,
        max_lookahead_m=12.0,
        kp=0.35,
        ki=0.12,
    ),
    Variant(
        name="Long Lookahead",
        color="#2563eb",
        min_lookahead_m=5.0,
        lookahead_gain_s=0.35,
        max_lookahead_m=24.0,
        kp=0.35,
        ki=0.12,
    ),
    Variant(
        name="Aggressive PI",
        color="#9333ea",
        min_lookahead_m=3.0,
        lookahead_gain_s=0.20,
        max_lookahead_m=18.0,
        kp=0.60,
        ki=0.22,
    ),
]


def simulate_variant(path: list[TrackPoint], variant: Variant) -> list[Sample]:
    wheelbase_m = 1.55
    dt_s = 0.02
    controller = SimState()
    first = path[0]
    state = VehicleState(first.x - 1.2, first.y - 0.15, first.yaw, 0.0)
    samples: list[Sample] = []

    for step in range(2600):
        high = pure_pursuit(state, path, controller, variant, wheelbase_m)
        low = speed_pi(high["target_speed"], state.speed, dt_s, controller, variant)

        accel = 5.0 * low["throttle"] - 6.0 * low["brake"] - 0.08 * state.speed
        speed = max(0.0, state.speed + accel * dt_s)
        yaw_rate = speed / wheelbase_m * tan(high["steering"])
        yaw = state.yaw + yaw_rate * dt_s
        state = VehicleState(
            x=state.x + speed * cos(yaw) * dt_s,
            y=state.y + speed * sin(yaw) * dt_s,
            yaw=yaw,
            speed=speed,
        )

        samples.append(
            Sample(
                time_s=step * dt_s,
                x_m=state.x,
                y_m=state.y,
                speed_mps=state.speed,
                target_speed_mps=high["target_speed"],
                tracking_error_m=nearest_path_distance(state, path),
                steering_rad=high["steering"],
                throttle=low["throttle"],
                brake=low["brake"],
            )
        )

        finish = path[-1]
        if high["target_index"] >= len(path) - 2 and hypot(state.x - finish.x, state.y - finish.y) < 2.0:
            break

    return samples


def pure_pursuit(
    state: VehicleState,
    path: list[TrackPoint],
    controller: SimState,
    variant: Variant,
    wheelbase_m: float,
) -> dict[str, float]:
    nearest = nearest_index(state, path, controller.target_index)
    lookahead = clamp(
        variant.min_lookahead_m + variant.lookahead_gain_s * state.speed,
        variant.min_lookahead_m,
        variant.max_lookahead_m,
    )
    target_index = nearest

    for index in range(nearest, len(path)):
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
    }


def speed_pi(
    target_speed_mps: float,
    measured_speed_mps: float,
    dt_s: float,
    controller: SimState,
    variant: Variant,
) -> dict[str, float]:
    error = target_speed_mps - measured_speed_mps
    controller.integrator = clamp(controller.integrator + error * dt_s, -3.0, 3.0)
    command = clamp(variant.kp * error + variant.ki * controller.integrator, -1.0, 1.0)
    return {
        "throttle": clamp(command, 0.0, 1.0),
        "brake": clamp(-1.3 * command, 0.0, 1.0),
    }


def nearest_index(state: VehicleState, path: list[TrackPoint], start_index: int) -> int:
    best_index = start_index
    best_distance = float("inf")
    for index in range(start_index, len(path)):
        point = path[index]
        distance = (state.x - point.x) ** 2 + (state.y - point.y) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = index
        if index - start_index > 180 and distance > best_distance * 6.0:
            break
    return best_index


def nearest_path_distance(state: VehicleState, path: list[TrackPoint]) -> float:
    return min(hypot(state.x - point.x, state.y - point.y) for point in path)


def save_svg(
    path: list[TrackPoint],
    results: dict[Variant, list[Sample]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1280
    height = 1080
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="34" y="40" font-family="Arial" font-size="26" font-weight="700" fill="#111827">Controller Settings Comparison</text>',
        '<text x="34" y="64" font-family="Arial" font-size="13" fill="#4b5563">Same FSD autocross track, different pure pursuit lookahead and PI gains.</text>',
    ]

    svg.extend(draw_trajectory_panel(path, results, 34, 86, 1212, 365))
    svg.extend(draw_metric_table(results, 34, 480, 1212, 150))
    svg.extend(draw_time_panel(results, "speed_mps", "Speed Tracking", "m/s", 0.0, 22.0, 34, 665, 1212, 150))
    svg.extend(draw_time_panel(results, "tracking_error_m", "Tracking Error", "m", 0.0, 3.0, 34, 845, 1212, 150))

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def draw_trajectory_panel(
    path: list[TrackPoint],
    results: dict[Variant, list[Sample]],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    all_x = [point.x for point in path]
    all_y = [point.y for point in path]
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

    centerline = [(point.x, point.y) for point in path]
    svg = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>',
        f'<text x="{x0 + 14}" y="{y0 + 24}" font-family="Arial" font-size="15" font-weight="700" fill="#111827">Trajectory Comparison</text>',
        f'<polyline points="{points_attr(centerline)}" fill="none" stroke="#d1d5db" stroke-width="{3.0 * scale:.2f}" stroke-linecap="round" stroke-linejoin="round" opacity="0.45"/>',
        f'<polyline points="{points_attr(centerline)}" fill="none" stroke="#111827" stroke-width="2.0" stroke-dasharray="7 7"/>',
    ]

    legend_x = x0 + width - 230
    legend_y = y0 + 24
    for index, (variant, samples) in enumerate(results.items()):
        vehicle_path = [(sample.x_m, sample.y_m) for sample in samples]
        svg.append(
            f'<polyline points="{points_attr(vehicle_path)}" fill="none" stroke="{variant.color}" stroke-width="2.5"/>'
        )
        ly = legend_y + index * 22
        svg.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 24}" y2="{ly}" stroke="{variant.color}" stroke-width="3"/>')
        svg.append(
            f'<text x="{legend_x + 32}" y="{ly + 4}" font-family="Arial" font-size="12" fill="#374151">{variant.name}</text>'
        )

    return svg


def draw_metric_table(
    results: dict[Variant, list[Sample]],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    col_w = width / 5.0
    row_h = 30.0
    svg = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>',
        f'<text x="{x0 + 14}" y="{y0 + 24}" font-family="Arial" font-size="15" font-weight="700" fill="#111827">Summary</text>',
    ]
    headers = ["Setting", "Max Error", "Avg Error", "Max Speed", "Duration"]
    for i, header in enumerate(headers):
        svg.append(
            f'<text x="{x0 + 14 + i * col_w}" y="{y0 + 56}" font-family="Arial" font-size="12" font-weight="700" fill="#374151">{header}</text>'
        )

    for row, (variant, samples) in enumerate(results.items()):
        y = y0 + 84 + row * row_h
        values = [
            variant.name,
            f"{max(s.tracking_error_m for s in samples):.2f} m",
            f"{sum(s.tracking_error_m for s in samples) / len(samples):.2f} m",
            f"{max(s.speed_mps for s in samples) * 3.6:.1f} km/h",
            f"{samples[-1].time_s:.1f} s",
        ]
        svg.append(f'<circle cx="{x0 + 6}" cy="{y - 4}" r="4" fill="{variant.color}"/>')
        for i, value in enumerate(values):
            svg.append(
                f'<text x="{x0 + 14 + i * col_w}" y="{y}" font-family="Arial" font-size="12" fill="#111827">{value}</text>'
            )
    return svg


def draw_time_panel(
    results: dict[Variant, list[Sample]],
    key: str,
    title: str,
    unit: str,
    value_min: float,
    value_max: float,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    left = x0 + 58
    right = x0 + width - 20
    top = y0 + 22
    bottom = y0 + height - 28
    plot_w = right - left
    plot_h = bottom - top
    max_time = max(samples[-1].time_s for samples in results.values())

    def sx(time_s: float) -> float:
        return left + plot_w * time_s / max_time

    def sy(value: float) -> float:
        normalized = clamp((value - value_min) / (value_max - value_min), 0.0, 1.0)
        return top + plot_h * (1.0 - normalized)

    svg = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>',
        f'<text x="{x0 + 14}" y="{y0 + 24}" font-family="Arial" font-size="15" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{x0 + 14}" y="{y0 + 44}" font-family="Arial" font-size="12" fill="#6b7280">{unit}</text>',
    ]
    for i in range(5):
        gy = top + plot_h * i / 4
        value = value_max - (value_max - value_min) * i / 4
        svg.append(f'<line x1="{left}" y1="{gy:.2f}" x2="{right}" y2="{gy:.2f}" stroke="#e5e7eb"/>')
        svg.append(
            f'<text x="{left - 8}" y="{gy + 4:.2f}" font-family="Arial" font-size="10" text-anchor="end" fill="#6b7280">{value:.1f}</text>'
        )

    for variant, samples in results.items():
        coords = []
        for sample in samples:
            value = getattr(sample, key)
            command = "M" if not coords else "L"
            coords.append(f"{command}{sx(sample.time_s):.2f},{sy(value):.2f}")
        svg.append(
            f'<path d="{" ".join(coords)}" fill="none" stroke="{variant.color}" stroke-width="2.2"/>'
        )

    return svg


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def main() -> None:
    path = build_fsd_autocross_track()
    results = {variant: simulate_variant(path, variant) for variant in VARIANTS}
    output_path = Path(__file__).resolve().parents[1] / "outputs" / "controller_settings_comparison.svg"
    save_svg(path, results, output_path)
    print(f"Saved visual comparison graph to {output_path}")


if __name__ == "__main__":
    main()
