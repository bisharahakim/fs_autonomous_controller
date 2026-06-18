from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fs_controller import (  # noqa: E402
    MAX_SPEED_30_KMH_MPS,
    PowertrainConfig,
    PowertrainModel,
    VehicleState,
    apply_friction_circle,
    requested_acceleration_from_actuators,
    tire_fit_summary,
)
from visualization.autocross_track import TrackPoint  # noqa: E402
from visualization.hockenheim_fsg_track import (  # noqa: E402
    DEFAULT_DS_M as HOCKENHEIM_DS_M,
    build_fsg_hockenheim_segment_indices,
)
from visualization.track_registry import get_track  # noqa: E402


@dataclass(frozen=True)
class Sample:
    time_s: float
    state: VehicleState
    target_index: int
    steering_rad: float
    throttle: float
    brake: float
    target_speed_mps: float
    lateral_error_m: float
    lateral_accel_mps2: float
    commanded_lateral_accel_mps2: float
    achievable_lateral_accel_mps2: float
    requested_accel_mps2: float
    actual_accel_mps2: float
    nearest_index: int
    segment_index: int


@dataclass(frozen=True)
class Event:
    kind: str
    time_s: float
    x_m: float
    y_m: float
    nearest_index: int
    segment_index: int
    detail: str


@dataclass
class ClosedTrackControllerState:
    target_index: int = 0
    integrator: float = 0.0


@dataclass(frozen=True)
class RunResult:
    track_name: str
    samples: list[Sample]
    events: list[Event]
    completed_lap: bool
    output_path: Path
    acceleration_output_path: Path
    lateral_acceleration_output_path: Path


DT_S = 0.02
WHEELBASE_M = 1.55
CAR_WIDTH_M = 1.5
MAX_STEPS = 12000
PREVIOUS_SIMPLE_MODEL_HOCKENHEIM_LAP_S = 100.52
PREVIOUS_POWERTRAIN_ONLY_HOCKENHEIM_LAP_S = 91.52


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline controller stack on a registered track.")
    parser.add_argument("--track", default="hockenheim_fsg", help="Track name, e.g. autocross or hockenheim_fsg")
    args = parser.parse_args()
    print_tire_fit_summary()

    try:
        result = run_baseline(args.track)
    except Exception:
        print("Baseline run failed with stack trace:")
        traceback.print_exc()
        raise

    report(result)


def run_baseline(track_name: str) -> RunResult:
    track = get_track(track_name)
    centerline = track.build_centerline()
    left_cones, right_cones = track.build_cones()
    drive_centerline = remove_duplicate_finish(centerline)
    segment_indices = segment_indices_for(track_name, len(centerline))[: len(drive_centerline)]
    controller = ClosedTrackControllerState()
    powertrain = PowertrainModel()
    vehicle_config = PowertrainConfig()

    first = drive_centerline[0]
    state = VehicleState(
        x=first.x - 1.2 * math.cos(first.yaw),
        y=first.y - 1.2 * math.sin(first.yaw) - 0.15,
        yaw=first.yaw,
        speed=0.0,
    )
    samples: list[Sample] = []
    events: list[Event] = []
    completed_lap = False

    for step in range(MAX_STEPS):
        previous_target_index = controller.target_index
        high = pure_pursuit_closed(state, drive_centerline, controller)
        low = speed_pi(
            target_speed_mps=high["target_speed"],
            measured_speed_mps=state.speed,
            dt_s=DT_S,
            controller=controller,
        )

        requested_accel = requested_acceleration_from_actuators(low["throttle"], low["brake"])
        powertrain_accel = powertrain.actual_acceleration(state.speed, requested_accel)
        speed_before_lateral = max(0.0, state.speed + powertrain_accel * DT_S)
        commanded_lateral_accel = (
            speed_before_lateral * speed_before_lateral * math.tan(high["steering"]) / WHEELBASE_M
        )
        actual_long_force, actual_lateral_force, _ = apply_friction_circle(
            longitudinal_force_n=vehicle_config.mass_kg * powertrain_accel,
            lateral_force_n=vehicle_config.mass_kg * commanded_lateral_accel,
            mass_kg=vehicle_config.mass_kg,
        )
        actual_accel = actual_long_force / vehicle_config.mass_kg
        achievable_lateral_accel = actual_lateral_force / vehicle_config.mass_kg
        speed = min(MAX_SPEED_30_KMH_MPS, max(0.0, state.speed + actual_accel * DT_S))
        yaw_rate = achievable_lateral_accel / max(speed, 1e-3)
        yaw = state.yaw + yaw_rate * DT_S
        x = state.x + speed * math.cos(yaw) * DT_S
        y = state.y + speed * math.sin(yaw) * DT_S
        state = VehicleState(x=x, y=y, yaw=yaw, speed=speed)

        nearest_index, lateral_error = nearest_path_distance(state, drive_centerline)
        segment_index = segment_indices[min(nearest_index, len(segment_indices) - 1)]
        lateral_accel = abs(achievable_lateral_accel)
        samples.append(
            Sample(
                time_s=step * DT_S,
                state=state,
                target_index=int(high["target_index"]),
                steering_rad=high["steering"],
                throttle=low["throttle"],
                brake=low["brake"],
                target_speed_mps=high["target_speed"],
                lateral_error_m=lateral_error,
                lateral_accel_mps2=lateral_accel,
                commanded_lateral_accel_mps2=commanded_lateral_accel,
                achievable_lateral_accel_mps2=achievable_lateral_accel,
                requested_accel_mps2=requested_accel,
                actual_accel_mps2=actual_accel,
                nearest_index=nearest_index,
                segment_index=segment_index,
            )
        )

        cone_event = cone_hit_event(
            state=state,
            cones=[*left_cones, *right_cones],
            time_s=step * DT_S,
            nearest_index=nearest_index,
            segment_index=segment_index,
        )
        if cone_event is not None:
            record_event(events, cone_event)

        if lateral_error > track.track_width_m / 2.0:
            record_event(
                events,
                Event(
                    kind="off_track",
                    time_s=step * DT_S,
                    x_m=state.x,
                    y_m=state.y,
                    nearest_index=nearest_index,
                    segment_index=segment_index,
                    detail=(
                        f"centerline error {lateral_error:.2f} m > "
                        f"{track.track_width_m / 2.0:.2f} m half-width"
                    ),
                ),
            )

        completed_lap = (
            previous_target_index > len(drive_centerline) - 80
            and int(high["target_index"]) < 80
            and step * DT_S > 5.0
        )
        if completed_lap:
            completed_lap = True
            break

    output_path = Path(__file__).resolve().parents[1] / "outputs" / f"{track_name}_baseline_trajectory.png"
    acceleration_output_path = (
        Path(__file__).resolve().parents[1] / "outputs" / f"{track_name}_powertrain_acceleration.png"
    )
    lateral_acceleration_output_path = (
        Path(__file__).resolve().parents[1] / "outputs" / f"{track_name}_tire_lateral_acceleration.png"
    )
    save_trajectory_plot(
        output_path=output_path,
        track_name=track_name,
        centerline=centerline,
        left_cones=left_cones,
        right_cones=right_cones,
        samples=samples,
        events=events,
    )
    save_acceleration_plot(acceleration_output_path, track_name, samples)
    save_lateral_acceleration_plot(lateral_acceleration_output_path, track_name, samples)
    return RunResult(
        track_name=track_name,
        samples=samples,
        events=events,
        completed_lap=completed_lap,
        output_path=output_path,
        acceleration_output_path=acceleration_output_path,
        lateral_acceleration_output_path=lateral_acceleration_output_path,
    )


def remove_duplicate_finish(centerline: list[TrackPoint]) -> list[TrackPoint]:
    if len(centerline) < 2:
        return centerline

    start = centerline[0]
    finish = centerline[-1]
    if math.hypot(finish.x - start.x, finish.y - start.y) < 1e-6:
        return centerline[:-1]
    return centerline


def segment_indices_for(track_name: str, centerline_length: int) -> list[int]:
    if track_name == "hockenheim_fsg":
        return build_fsg_hockenheim_segment_indices(HOCKENHEIM_DS_M)
    return [0 for _ in range(centerline_length)]


def nearest_path_distance(state: VehicleState, path: list[TrackPoint]) -> tuple[int, float]:
    best_index = 0
    best_distance = float("inf")
    for index, point in enumerate(path):
        distance = math.hypot(state.x - point.x, state.y - point.y)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index, best_distance


def pure_pursuit_closed(
    state: VehicleState,
    path: list[TrackPoint],
    controller: ClosedTrackControllerState,
) -> dict[str, float]:
    nearest = nearest_index_closed(state, path, controller.target_index)
    lookahead = clamp(3.0 + 0.2 * state.speed, 3.0, 18.0)
    target_index = nearest

    for offset in range(len(path)):
        index = (nearest + offset) % len(path)
        point = path[index]
        if math.hypot(point.x - state.x, point.y - state.y) >= lookahead:
            target_index = index
            break

    controller.target_index = target_index
    target = path[target_index]
    dx = target.x - state.x
    dy = target.y - state.y
    local_x = math.cos(state.yaw) * dx + math.sin(state.yaw) * dy
    local_y = -math.sin(state.yaw) * dx + math.cos(state.yaw) * dy
    distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
    curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
    steering = clamp(math.atan(WHEELBASE_M * curvature), -0.5, 0.5)

    return {
        "steering": steering,
        "target_speed": target.speed,
        "target_index": float(target_index),
        "lookahead": lookahead,
    }


def nearest_index_closed(
    state: VehicleState,
    path: list[TrackPoint],
    last_target_index: int,
) -> int:
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


def speed_pi(
    target_speed_mps: float,
    measured_speed_mps: float,
    dt_s: float,
    controller: ClosedTrackControllerState,
) -> dict[str, float]:
    error = target_speed_mps - measured_speed_mps
    controller.integrator = clamp(controller.integrator + error * dt_s, -3.0, 3.0)
    command = clamp(0.35 * error + 0.12 * controller.integrator, -1.0, 1.0)
    return {
        "throttle": clamp(command, 0.0, 1.0),
        "brake": clamp(-1.3 * command, 0.0, 1.0),
    }


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def cone_hit_event(
    state: VehicleState,
    cones: list[TrackPoint],
    time_s: float,
    nearest_index: int,
    segment_index: int,
) -> Event | None:
    hit_radius = CAR_WIDTH_M / 2.0
    closest_distance = float("inf")
    closest_cone = None
    for cone in cones:
        distance = math.hypot(state.x - cone.x, state.y - cone.y)
        if distance < closest_distance:
            closest_distance = distance
            closest_cone = cone

    if closest_cone is None or closest_distance > hit_radius:
        return None

    return Event(
        kind="cone_hit",
        time_s=time_s,
        x_m=state.x,
        y_m=state.y,
        nearest_index=nearest_index,
        segment_index=segment_index,
        detail=(
            f"vehicle point within {closest_distance:.2f} m of cone at "
            f"({closest_cone.x:.2f}, {closest_cone.y:.2f}); threshold {hit_radius:.2f} m"
        ),
    )


def record_event(events: list[Event], event: Event) -> None:
    if events:
        previous = events[-1]
        same_area = previous.kind == event.kind and previous.segment_index == event.segment_index
        close_in_time = event.time_s - previous.time_s < 1.0
        if same_area and close_in_time:
            return
    events.append(event)


def save_trajectory_plot(
    output_path: Path,
    track_name: str,
    centerline: list[TrackPoint],
    left_cones: list[TrackPoint],
    right_cones: list[TrackPoint],
    samples: list[Sample],
    events: list[Event],
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fs_autonomous_controller_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.plot(
        [point.x for point in centerline],
        [point.y for point in centerline],
        color="#111827",
        linewidth=1.4,
        label="centerline",
    )
    ax.scatter(
        [point.x for point in left_cones],
        [point.y for point in left_cones],
        s=8,
        color="#f4a23a",
        label="left cones",
    )
    ax.scatter(
        [point.x for point in right_cones],
        [point.y for point in right_cones],
        s=8,
        color="#3aa0f4",
        label="right cones",
    )
    if samples:
        ax.plot(
            [sample.state.x for sample in samples],
            [sample.state.y for sample in samples],
            color="#ef4444",
            linewidth=1.8,
            label="driven trajectory",
        )
        ax.scatter(
            [samples[0].state.x],
            [samples[0].state.y],
            s=45,
            color="#16a34a",
            marker="o",
            label="start",
            zorder=5,
        )
    for event in events[:12]:
        ax.scatter(
            [event.x_m],
            [event.y_m],
            s=90,
            color="#dc2626",
            marker="x",
            label=f"incident: {event.kind}" if event == events[0] else "",
            zorder=6,
        )
        ax.text(event.x_m + 2.0, event.y_m + 2.0, f"segment {event.segment_index}", color="#991b1b")

    ax.set_title(f"{track_name} baseline trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_acceleration_plot(output_path: Path, track_name: str, samples: list[Sample]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fs_autonomous_controller_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(
        [sample.time_s for sample in samples],
        [sample.requested_accel_mps2 for sample in samples],
        color="#2563eb",
        linewidth=2.6,
        alpha=0.72,
        label="a requested",
    )
    ax.plot(
        [sample.time_s for sample in samples],
        [sample.actual_accel_mps2 for sample in samples],
        color="#dc2626",
        linewidth=1.5,
        linestyle="--",
        label="a actual",
    )
    ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_title(f"{track_name} requested vs actual longitudinal acceleration")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("acceleration [m/s^2]")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_lateral_acceleration_plot(output_path: Path, track_name: str, samples: list[Sample]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fs_autonomous_controller_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(
        [sample.time_s for sample in samples],
        [sample.commanded_lateral_accel_mps2 for sample in samples],
        color="#2563eb",
        linewidth=2.2,
        alpha=0.72,
        label="commanded lateral acceleration",
    )
    ax.plot(
        [sample.time_s for sample in samples],
        [sample.achievable_lateral_accel_mps2 for sample in samples],
        color="#dc2626",
        linewidth=1.5,
        linestyle="--",
        label="achievable lateral acceleration",
    )
    ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_title(f"{track_name} commanded vs achievable lateral acceleration")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lateral acceleration [m/s^2]")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_tire_fit_summary() -> None:
    print("Tire grip fit sanity check:")
    for row in tire_fit_summary():
        print(f"  {row}")


def report(result: RunResult) -> None:
    samples = result.samples
    if not samples:
        print("No samples were produced.")
        return

    max_lateral_accel = max(sample.lateral_accel_mps2 for sample in samples)
    max_speed = max(sample.state.speed for sample in samples)
    clipped_samples = [
        sample
        for sample in samples
        if sample.requested_accel_mps2 > 0.0
        and sample.actual_accel_mps2 < sample.requested_accel_mps2 - 0.05
    ]
    last = samples[-1]
    print(f"Track: {result.track_name}")
    if result.completed_lap:
        print(f"Lap time: {last.time_s:.2f} s")
        if result.track_name == "hockenheim_fsg":
            delta_powertrain = last.time_s - PREVIOUS_POWERTRAIN_ONLY_HOCKENHEIM_LAP_S
            delta_simple = last.time_s - PREVIOUS_SIMPLE_MODEL_HOCKENHEIM_LAP_S
            print(
                "Previous powertrain-only lap time: "
                f"{PREVIOUS_POWERTRAIN_ONLY_HOCKENHEIM_LAP_S:.2f} s "
                f"(delta {delta_powertrain:+.2f} s)"
            )
            print(
                "Previous simple-model lap time: "
                f"{PREVIOUS_SIMPLE_MODEL_HOCKENHEIM_LAP_S:.2f} s "
                f"(delta {delta_simple:+.2f} s)"
            )
    else:
        print(f"Lap status: did not complete within {last.time_s:.2f} s")
    if result.events:
        print(f"Incidents recorded: {len(result.events)}")
        first = result.events[0]
        print(f"First incident: {first.kind} at t={first.time_s:.2f} s")
        print(f"First incident location: ({first.x_m:.2f}, {first.y_m:.2f})")
        print(f"First incident nearest centerline index: {first.nearest_index}")
        print(f"First incident segment index: {first.segment_index}")
        print(f"First incident detail: {first.detail}")
    else:
        print("Incidents recorded: 0")
    print(f"Max lateral acceleration: {max_lateral_accel:.2f} m/s^2")
    print(f"Max speed: {max_speed:.2f} m/s")
    print(
        "Powertrain-limited drive samples: "
        f"{len(clipped_samples)} / {len(samples)} "
        f"({100.0 * len(clipped_samples) / len(samples):.1f}%)"
    )
    print(f"Final target index: {last.target_index}")
    print(f"Final nearest segment index: {last.segment_index}")
    print(f"Trajectory plot: {result.output_path}")
    print(f"Acceleration plot: {result.acceleration_output_path}")
    print(f"Lateral acceleration plot: {result.lateral_acceleration_output_path}")


if __name__ == "__main__":
    main()
