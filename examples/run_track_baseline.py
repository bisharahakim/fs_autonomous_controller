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
from fs_controller.raceline import Raceline, load_raceline, wrap_angle  # noqa: E402
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
    lookahead_speed_m: float
    lookahead_curv_m: float
    lookahead_used_m: float
    kappa_max_horizon_radpm: float


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
    previous_accel_cmd_mps2: float = 0.0


@dataclass(frozen=True)
class RacelineStart:
    index: int
    x_m: float
    y_m: float
    yaw_rad: float
    yaw_deg: float
    s_m: float
    used_flip: bool
    used_psi_alignment: bool


@dataclass(frozen=True)
class RunResult:
    track_name: str
    samples: list[Sample]
    events: list[Event]
    completed_lap: bool
    output_path: Path
    acceleration_output_path: Path
    lateral_acceleration_output_path: Path
    lookahead_output_path: Path
    raceline_start: RacelineStart | None = None
    raceline_path: Path | None = None


DT_S = 0.02
WHEELBASE_M = 1.55
CAR_WIDTH_M = 1.1
MAX_STEPS = 12000
PREVIOUS_SIMPLE_MODEL_HOCKENHEIM_LAP_S = 100.52
PREVIOUS_POWERTRAIN_ONLY_HOCKENHEIM_LAP_S = 91.52
OPTIMIZER_BENJAMIN24_LAP_S = 78.64
OPTIMIZER_BENJAMIN24_TOP_SPEED_MPS = 24.92
OPTIMIZER_BENJAMIN24_MAX_LAT_ACCEL_MPS2 = 13.27
OPTIMIZER_BENJAMIN24_SAFE_LAP_S = 90.36
OPTIMIZER_BENJAMIN24_SAFE_TOP_SPEED_MPS = 22.00
OPTIMIZER_BENJAMIN24_SAFE_MAX_LAT_ACCEL_MPS2 = 10.30
SAFE_BRAKING_ACCEL_MPS2 = 8.0
SAFE_THROTTLE_ACCEL_MPS2 = 5.0
RACELINE_SPEED_KP = 2.0
RACELINE_SPEED_KI = 0.5
RACELINE_INTEGRAL_LIMIT_MPS2 = 5.0
MAX_JERK_THROTTLE_MPS3 = 8.0
MAX_JERK_BRAKE_MPS3 = 20.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline controller stack on a registered track.")
    parser.add_argument("--track", default="hockenheim_fsg", help="Track name, e.g. autocross or hockenheim_fsg")
    parser.add_argument("--raceline", type=Path, help="Optional optimizer trajectory CSV to track instead of centerline")
    parser.add_argument(
        "--speed-cap",
        nargs="?",
        const=MAX_SPEED_30_KMH_MPS,
        type=float,
        help="Optional max speed in m/s. Passing --speed-cap without a value uses 30 km/h.",
    )
    args = parser.parse_args()
    print_tire_fit_summary()

    try:
        result = run_baseline(args.track, args.raceline, args.speed_cap)
    except Exception:
        print("Baseline run failed with stack trace:")
        traceback.print_exc()
        raise

    report(result)


def run_baseline(track_name: str, raceline_path: Path | None = None, speed_cap_mps: float | None = None) -> RunResult:
    track = get_track(track_name)
    centerline = track.build_centerline()
    left_cones, right_cones = track.build_cones()
    drive_centerline = remove_duplicate_finish(centerline)
    segment_indices = segment_indices_for(track_name, len(centerline))[: len(drive_centerline)]
    raceline = load_raceline(raceline_path) if raceline_path is not None else None
    raceline_start = prepare_raceline_start(raceline) if raceline is not None else None
    if raceline is None and speed_cap_mps is None:
        speed_cap_mps = MAX_SPEED_30_KMH_MPS
    controller = ClosedTrackControllerState()
    powertrain = PowertrainModel()
    vehicle_config = PowertrainConfig()

    if raceline is None:
        first = drive_centerline[0]
        state = VehicleState(x=first.x, y=first.y, yaw=first.yaw, speed=0.5)
    else:
        assert raceline_start is not None
        controller.target_index = raceline_start.index
        state = VehicleState(
            x=raceline_start.x_m,
            y=raceline_start.y_m,
            yaw=raceline_start.yaw_rad,
            speed=0.5,
        )
    samples: list[Sample] = []
    events: list[Event] = []
    completed_lap = False
    append_initial_sample(samples, state, drive_centerline, segment_indices, raceline)

    for step in range(1, MAX_STEPS):
        previous_target_index = controller.target_index
        if raceline is None:
            high = pure_pursuit_closed(state, drive_centerline, controller)
        else:
            high = raceline_pure_pursuit_closed(state, raceline, controller)
        if raceline is None:
            low = speed_pi(
                target_speed_mps=high["target_speed"],
                measured_speed_mps=state.speed,
                dt_s=DT_S,
                controller=controller,
            )
            requested_accel = requested_acceleration_from_actuators(low["throttle"], low["brake"])
        else:
            low = speed_feedforward_pi(
                target_speed_mps=high["target_speed"],
                target_accel_mps2=high["target_accel"],
                measured_speed_mps=state.speed,
                dt_s=DT_S,
                controller=controller,
            )
            requested_accel = low["requested_accel"]
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
        speed = max(0.0, state.speed + actual_accel * DT_S)
        if speed_cap_mps is not None:
            speed = min(speed_cap_mps, speed)
        yaw_rate = achievable_lateral_accel / max(speed, 1e-3)
        yaw = state.yaw + yaw_rate * DT_S
        x = state.x + speed * math.cos(yaw) * DT_S
        y = state.y + speed * math.sin(yaw) * DT_S
        state = VehicleState(x=x, y=y, yaw=yaw, speed=speed)

        centerline_nearest_index, lateral_error = nearest_path_distance(state, drive_centerline)
        segment_index = segment_indices[min(centerline_nearest_index, len(segment_indices) - 1)]
        nearest_index = int(high["nearest_index"]) if raceline is not None else centerline_nearest_index
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
                lookahead_speed_m=high["lookahead_speed"],
                lookahead_curv_m=high["lookahead_curv"],
                lookahead_used_m=high["lookahead"],
                kappa_max_horizon_radpm=high["kappa_max_horizon"],
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
            previous_target_index > reference_length_for_completion(drive_centerline, raceline) - 80
            and int(high["target_index"]) < 80
            and step * DT_S > 5.0
        )
        if completed_lap:
            completed_lap = True
            break

    output_name = f"{track_name}_raceline_tracking.png" if raceline is not None else f"{track_name}_baseline_trajectory.png"
    output_path = Path(__file__).resolve().parents[1] / "outputs" / output_name
    acceleration_output_path = (
        Path(__file__).resolve().parents[1] / "outputs" / f"{track_name}_powertrain_acceleration.png"
    )
    lateral_acceleration_output_path = (
        Path(__file__).resolve().parents[1] / "outputs" / f"{track_name}_tire_lateral_acceleration.png"
    )
    lookahead_output_path = Path(__file__).resolve().parents[1] / "outputs" / "diag_lookahead.png"
    save_trajectory_plot(
        output_path=output_path,
        track_name=track_name,
        centerline=centerline,
        left_cones=left_cones,
        right_cones=right_cones,
        samples=samples,
        events=events,
        raceline=raceline,
    )
    save_acceleration_plot(acceleration_output_path, track_name, samples)
    save_lateral_acceleration_plot(lateral_acceleration_output_path, track_name, samples)
    if raceline is not None:
        save_lookahead_diagnostic_plot(lookahead_output_path, track_name, samples)
    return RunResult(
        track_name=track_name,
        samples=samples,
        events=events,
        completed_lap=completed_lap,
        output_path=output_path,
        acceleration_output_path=acceleration_output_path,
        lateral_acceleration_output_path=lateral_acceleration_output_path,
        lookahead_output_path=lookahead_output_path,
        raceline_start=raceline_start,
        raceline_path=raceline_path,
    )


def remove_duplicate_finish(centerline: list[TrackPoint]) -> list[TrackPoint]:
    if len(centerline) < 2:
        return centerline

    start = centerline[0]
    finish = centerline[-1]
    if math.hypot(finish.x - start.x, finish.y - start.y) < 1e-6:
        return centerline[:-1]
    return centerline


def append_initial_sample(
    samples: list[Sample],
    state: VehicleState,
    drive_centerline: list[TrackPoint],
    segment_indices: list[int],
    raceline: Raceline | None,
) -> None:
    centerline_nearest_index, lateral_error = nearest_path_distance(state, drive_centerline)
    segment_index = segment_indices[min(centerline_nearest_index, len(segment_indices) - 1)]
    if raceline is None:
        target_index = 0
        nearest_index = centerline_nearest_index
        target_speed = drive_centerline[0].speed
    else:
        nearest_index = raceline.nearest_index(state.x, state.y)
        target_index = nearest_index
        target_speed = float(raceline.v_target_mps[target_index])

    samples.append(
        Sample(
            time_s=0.0,
            state=state,
            target_index=target_index,
            steering_rad=0.0,
            throttle=0.0,
            brake=0.0,
            target_speed_mps=target_speed,
            lateral_error_m=lateral_error,
            lateral_accel_mps2=0.0,
            commanded_lateral_accel_mps2=0.0,
            achievable_lateral_accel_mps2=0.0,
            requested_accel_mps2=0.0,
            actual_accel_mps2=0.0,
            nearest_index=nearest_index,
            segment_index=segment_index,
            lookahead_speed_m=0.0,
            lookahead_curv_m=0.0,
            lookahead_used_m=0.0,
            kappa_max_horizon_radpm=0.0,
        )
    )


def prepare_raceline_start(raceline: Raceline | None) -> RacelineStart | None:
    if raceline is None:
        return None

    used_flip = False
    used_psi_alignment = False
    start_idx = raceline.closest_index_to(0.0, 0.0)
    start_dist = math.hypot(float(raceline.x_m[start_idx]), float(raceline.y_m[start_idx]))
    if start_dist > 2.0:
        raise ValueError(
            f"No raceline point near origin (closest is {start_dist:.2f} m at index {start_idx}). "
            "Check that the raceline corresponds to this track."
        )

    yaw_rad = float(raceline.psi_rad[start_idx])
    yaw_deg = normalize_degrees(math.degrees(yaw_rad))
    if abs(yaw_deg) > 90.0:
        geometric_yaw = raceline.geometric_heading_at(start_idx)
        psi_error = wrap_angle(yaw_rad - geometric_yaw)
        if abs(abs(math.degrees(psi_error)) - 90.0) < 15.0:
            raceline.rotate_psi(-psi_error)
            used_psi_alignment = True
        else:
            raceline.reverse_direction()
            used_flip = True

        start_idx = raceline.closest_index_to(0.0, 0.0)
        yaw_rad = float(raceline.psi_rad[start_idx])
        yaw_deg = normalize_degrees(math.degrees(yaw_rad))
        if abs(yaw_deg) > 90.0:
            raise ValueError(
                f"Raceline at start points {yaw_deg:.1f} degrees from +x axis. "
                "Expected near 0 degrees. The raceline may be traversed in the wrong direction."
            )

    start = RacelineStart(
        index=start_idx,
        x_m=float(raceline.x_m[start_idx]),
        y_m=float(raceline.y_m[start_idx]),
        yaw_rad=yaw_rad,
        yaw_deg=yaw_deg,
        s_m=float(raceline.s_m[start_idx]),
        used_flip=used_flip,
        used_psi_alignment=used_psi_alignment,
    )
    print(f"Start index in raceline: {start.index}")
    print(f"Start position: ({start.x_m:.3f}, {start.y_m:.3f})")
    print(f"Start yaw: {start.yaw_rad:.3f} rad ({start.yaw_deg:.1f} degrees)")
    if start.used_flip:
        print("Raceline direction correction: Option A flip applied.")
    if start.used_psi_alignment:
        print("Raceline yaw correction: psi column aligned to geometric tangent.")
    return start


def normalize_degrees(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg <= -180.0:
        angle_deg += 360.0
    return angle_deg


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


def reference_length_for_completion(path: list[TrackPoint], raceline: Raceline | None) -> int:
    if raceline is None:
        return len(path)
    return len(raceline.s_m) - 1


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
        "target_accel": 0.0,
        "target_index": float(target_index),
        "nearest_index": float(nearest),
        "lookahead": lookahead,
        "lookahead_speed": lookahead,
        "lookahead_curv": lookahead,
        "kappa_max_horizon": 0.0,
    }


def compute_lookahead(
    v_actual: float,
    raceline: Raceline,
    s_now: float,
    horizon: float = 10.0,
) -> tuple[float, float, float, float]:
    sample_count = 20
    s_samples = [
        (s_now + horizon * i / (sample_count - 1)) % raceline.total_s
        for i in range(sample_count)
    ]
    kappa_max = max(abs(raceline.kappa_at(s)) for s in s_samples)
    lookahead_speed = v_actual * 0.45
    lookahead_curv = 1.0 / max(kappa_max, 1e-3)
    lookahead = clamp(min(lookahead_speed, lookahead_curv), 2.0, 6.0)
    return lookahead, lookahead_speed, lookahead_curv, kappa_max


def raceline_pure_pursuit_closed(
    state: VehicleState,
    raceline: Raceline,
    controller: ClosedTrackControllerState,
) -> dict[str, float]:
    nearest = raceline.nearest_index(state.x, state.y)
    s_now = float(raceline.s_m[nearest])
    lookahead, lookahead_speed, lookahead_curv, kappa_max = compute_lookahead(state.speed, raceline, s_now)
    target = raceline.target_at_s(s_now + lookahead)
    brake_distance = max(3.0, state.speed * state.speed / (2.0 * 8.0))
    speed_target = raceline.target_at_s(s_now + brake_distance)
    controller.target_index = target.index

    dx = target.x_m - state.x
    dy = target.y_m - state.y
    local_x = math.cos(state.yaw) * dx + math.sin(state.yaw) * dy
    local_y = -math.sin(state.yaw) * dx + math.cos(state.yaw) * dy
    distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
    curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
    steering = clamp(math.atan(WHEELBASE_M * curvature), -0.5, 0.5)

    return {
        "steering": steering,
        "target_speed": speed_target.v_target_mps,
        "target_accel": speed_target.a_target_mps2,
        "target_index": float(target.index),
        "nearest_index": float(nearest),
        "lookahead": lookahead,
        "lookahead_speed": lookahead_speed,
        "lookahead_curv": lookahead_curv,
        "kappa_max_horizon": kappa_max,
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


def speed_feedforward_pi(
    target_speed_mps: float,
    target_accel_mps2: float,
    measured_speed_mps: float,
    dt_s: float,
    controller: ClosedTrackControllerState,
) -> dict[str, float]:
    error = target_speed_mps - measured_speed_mps
    controller.integrator = clamp(
        controller.integrator + RACELINE_SPEED_KI * error * dt_s,
        -RACELINE_INTEGRAL_LIMIT_MPS2,
        RACELINE_INTEGRAL_LIMIT_MPS2,
    )
    raw_accel = target_accel_mps2 + RACELINE_SPEED_KP * error + controller.integrator
    lower = controller.previous_accel_cmd_mps2 - MAX_JERK_BRAKE_MPS3 * dt_s
    upper = controller.previous_accel_cmd_mps2 + MAX_JERK_THROTTLE_MPS3 * dt_s
    accel_cmd = clamp(raw_accel, lower, upper)
    controller.previous_accel_cmd_mps2 = accel_cmd

    return {
        "throttle": clamp(accel_cmd / SAFE_THROTTLE_ACCEL_MPS2, 0.0, 1.0),
        "brake": clamp(-accel_cmd / SAFE_BRAKING_ACCEL_MPS2, 0.0, 1.0),
        "requested_accel": accel_cmd,
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
    raceline: Raceline | None = None,
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
        if raceline is not None:
            ax.plot(
                raceline.x_m,
                raceline.y_m,
                color="#dc2626",
                linewidth=1.4,
                label="planned raceline",
            )
        ax.plot(
            [sample.state.x for sample in samples],
            [sample.state.y for sample in samples],
            color="#2563eb" if raceline is not None else "#ef4444",
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
    cone_hit_label_added = False
    cone_hit_label_count = 0
    for event in events:
        if event.kind != "cone_hit":
            continue
        ax.scatter(
            [event.x_m],
            [event.y_m],
            s=90,
            color="#dc2626",
            marker="x",
            label="" if cone_hit_label_added else "cone hit",
            zorder=6,
        )
        cone_hit_label_added = True
        if raceline is None or cone_hit_label_count < 12:
            ax.text(event.x_m + 2.0, event.y_m + 2.0, f"segment {event.segment_index}", color="#991b1b")
        cone_hit_label_count += 1

    title_suffix = "raceline tracking" if raceline is not None else "baseline trajectory"
    ax.set_title(f"{track_name} {title_suffix}")
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


def save_lookahead_diagnostic_plot(output_path: Path, track_name: str, samples: list[Sample]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fs_autonomous_controller_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    times = [sample.time_s for sample in samples]
    ax.plot(
        times,
        [sample.lookahead_speed_m for sample in samples],
        color="#2563eb",
        linewidth=2.0,
        alpha=0.75,
        label="L_speed",
    )
    ax.plot(
        times,
        [sample.lookahead_curv_m for sample in samples],
        color="#dc2626",
        linewidth=1.6,
        alpha=0.8,
        label="L_curv",
    )
    ax.plot(
        times,
        [sample.lookahead_used_m for sample in samples],
        color="#111827",
        linewidth=2.3,
        label="lookahead used",
    )
    ax.set_title(f"{track_name} curvature-aware lookahead diagnostic")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lookahead [m]")
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
    opt_lap, opt_top_speed, opt_lat_accel = optimizer_reference_metrics(result.raceline_path)
    print(f"Track: {result.track_name}")
    if result.raceline_path is not None:
        print(f"Raceline: {result.raceline_path}")
        print(f"Optimizer predicted lap time: {opt_lap:.2f} s")
    if result.completed_lap:
        print(f"Lap time: {last.time_s:.2f} s")
        if result.raceline_path is not None:
            gap = last.time_s - opt_lap
            gap_pct = 100.0 * gap / opt_lap
            print(f"Gap to optimizer: {gap:+.2f} s ({gap_pct:+.1f}%)")
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
    if result.raceline_path is not None:
        max_target_speed = max(sample.target_speed_mps for sample in samples)
        print(f"Max target speed seen by controller: {max_target_speed:.2f} m/s")
        print(f"Optimizer max lateral acceleration: {opt_lat_accel:.2f} m/s^2")
        print(f"Optimizer max speed: {opt_top_speed:.2f} m/s")
        print_lookahead_binding_summary(result)
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
    if result.raceline_path is not None:
        print(f"Lookahead diagnostic plot: {result.lookahead_output_path}")


def print_lookahead_binding_summary(result: RunResult) -> None:
    if result.raceline_path is None:
        return

    raceline = load_raceline(result.raceline_path)
    binding_samples = [
        sample
        for sample in result.samples
        if sample.lookahead_curv_m < sample.lookahead_speed_m - 1e-6
        and sample.lookahead_used_m <= sample.lookahead_curv_m + 1e-6
    ]
    if not binding_samples:
        print("Curvature lookahead binding: never")
        return

    first = binding_samples[0]
    last = binding_samples[-1]
    first_s = float(raceline.s_m[min(first.nearest_index, len(raceline.s_m) - 1)])
    last_s = float(raceline.s_m[min(last.nearest_index, len(raceline.s_m) - 1)])
    max_reduction = max(sample.lookahead_speed_m - sample.lookahead_used_m for sample in binding_samples)
    print(
        "Curvature lookahead binding: "
        f"{len(binding_samples)} samples, first at t={first.time_s:.2f} s / s={first_s:.2f} m, "
        f"last at t={last.time_s:.2f} s / s={last_s:.2f} m, "
        f"max reduction {max_reduction:.2f} m"
    )


def optimizer_reference_metrics(raceline_path: Path | None) -> tuple[float, float, float]:
    if raceline_path is not None and "safe" in raceline_path.name:
        return (
            OPTIMIZER_BENJAMIN24_SAFE_LAP_S,
            OPTIMIZER_BENJAMIN24_SAFE_TOP_SPEED_MPS,
            OPTIMIZER_BENJAMIN24_SAFE_MAX_LAT_ACCEL_MPS2,
        )
    return (
        OPTIMIZER_BENJAMIN24_LAP_S,
        OPTIMIZER_BENJAMIN24_TOP_SPEED_MPS,
        OPTIMIZER_BENJAMIN24_MAX_LAT_ACCEL_MPS2,
    )


if __name__ == "__main__":
    main()
