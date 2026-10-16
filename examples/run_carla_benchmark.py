"""Fair lateral-controller benchmark: Cross-Track PID vs Pure Pursuit vs RAPP.

Every run uses the same vehicle, raceline, start pose, speed reference,
velocity PID, actuator limits and fixed simulator timestep. Only the steering
law changes (--controller).

Typical use (CARLA running, FSD_Empty_Asphalt in Play, cones spawned):

    python3 examples/run_carla_benchmark.py --controller rapp
    python3 examples/run_carla_benchmark.py --controller pp
    python3 examples/run_carla_benchmark.py --controller pid --speed-scale 0.6
    python3 examples/run_carla_benchmark.py --controller rapp --repeat 5

Dry run without CARLA (simple kinematic car, for checking the code path):

    python3 examples/run_carla_benchmark.py --controller pp --plant kinematic

If a run is killed hard and the editor freezes waiting for ticks:

    python3 examples/run_carla_benchmark.py --restore-async
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fs_controller.lateral import (  # noqa: E402
    CrossTrackPID,
    CrossTrackPIDConfig,
    PurePursuitLateral,
    PurePursuitLateralConfig,
    RAPPLateral,
    signed_cross_track,
)
from fs_controller.exploration import (  # noqa: E402
    CarGeometry,
    ConePerception,
    ExplorationConfig,
    ExplorationLateral,
    KinoRRT,
    KinoRRTConfig,
    PerceptionConfig,
)
from fs_controller.raceline import Raceline, load_raceline  # noqa: E402
from fs_controller.rapp import RAPP, RAPPConfig  # noqa: E402
from fs_controller.velocity_pid import VelocityPID, VelocityPIDConfig  # noqa: E402
from fs_controller.vehicle import VehicleParams, load_vehicle  # noqa: E402

RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "carla"

# Set from the vehicle file (--vehicle) in main(); these are the Audi defaults.
VEHICLE: VehicleParams = load_vehicle("audi_tt_carla")
VEHICLE_BLUEPRINT = "vehicle.ue4.audi.tt"
WHEELBASE_M = 2.51
MAX_STEER_RAD = 0.5
STEER_MAP = "legacy"


REFERENCE = "centre"


def reference_point(x: float, y: float, yaw: float) -> tuple[float, float]:
    """Point the steering controller works from.

    centre: the vehicle's logged position (CARLA actor origin, the body centre).
    rear:   the rear axle, assumed half a wheelbase behind the centre. Pure Pursuit's
            geometry (steering angle from wheelbase and arc curvature) is exact only for
            the rear axle, because that is the point whose velocity follows the heading.
    """
    if REFERENCE == "rear":
        half = 0.5 * WHEELBASE_M
        return x - half * math.cos(yaw), y - half * math.sin(yaw)
    return x, y


def apply_grip_cap(raceline: Raceline, ay_max: float, smooth_m: float = 1.0) -> int:
    """Cap the planned speed so the car never needs more than ay_max in a corner.

    v <= sqrt(ay_max / |kappa|), with |kappa| averaged over +-smooth_m so single-point
    spikes in the curvature column do not dictate the speed. The planned acceleration
    is recomputed from the capped profile (a = v dv/ds). Returns the number of capped points.
    """
    n = len(raceline.s_m) - 1                          # last row repeats the first
    step = float(np.median(np.diff(raceline.s_m)))
    half = max(1, int(round(smooth_m / step)))
    kappa = np.abs(raceline.kappa_radpm[:n])
    padded = np.concatenate((kappa[-half:], kappa, kappa[:half]))
    kappa_avg = np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")
    v_old = raceline.v_target_mps[:n].copy()
    v_new = np.minimum(v_old, np.sqrt(ay_max / np.maximum(kappa_avg, 1e-6)))
    ds = np.diff(raceline.s_m)
    v_next = np.roll(v_new, -1)
    accel = (v_next**2 - v_new**2) / (2.0 * ds)
    raceline.v_target_mps = np.append(v_new, v_new[0])
    raceline.a_target_mps2 = np.append(accel, accel[0])
    return int(np.sum(v_new < v_old - 1e-6))


def steering_command(delta_rad: float, speed_mps: float) -> float:
    """Desired road-wheel angle -> normalised actuator command in [-1, 1] (track sign).

    legacy:     command = delta / MAX_STEER_RAD (assumes the car turns the same at every speed)
    calibrated: invert the vehicle's measured steering response at the current speed
    """
    if STEER_MAP == "calibrated":
        return VEHICLE.steering.command_for(delta_rad, speed_mps)
    return clamp(delta_rad / VEHICLE.legacy_max_steer_rad, -1.0, 1.0)

LOG_COLUMNS = [
    "t_s", "x_m", "y_m", "yaw_carla_deg", "yaw_rapp_rad", "v_mps",
    "steering_rad", "steer_norm", "target_speed_mps", "target_accel_mps2",
    "throttle", "brake", "nearest_index", "target_index", "target_x", "target_y",
    "lap_complete", "s_m", "xte_m", "lookahead_m",
]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------
# Plants: the thing being driven. Both expose the vehicle in TRACK coordinates
# (x, y, yaw counter-clockwise) so the controllers never see CARLA's frame.
# --------------------------------------------------------------------------

class CarlaPlant:
    """CARLA 0.10.0 vehicle, stepped in synchronous mode with a fixed timestep.

    Track frame <-> CARLA frame (CARLA is left-handed; the cones were spawned
    with --flip-y):   x_carla = x,  y_carla = -y,  yaw_carla = -yaw,
    and the normalised steer sign is inverted. These formulas are the ones
    verified by the September laps (run_carla_rapp_smoke_test.py).
    """

    def __init__(self, dt_s: float, sync: bool, host: str, port: int, realtime: bool) -> None:
        import carla  # imported here so --plant kinematic works without CARLA

        self._carla = carla
        self.dt_s = dt_s
        self.sync = sync
        self.realtime = realtime
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.vehicle = None
        self._collision_sensor = None
        self._collisions: list[tuple[int, str, int]] = []
        self._end_frame: int | None = None
        self._original_settings = None
        self._last_wall = None
        self.info = {
            "plant": "carla",
            "client_version": self.client.get_client_version(),
            "server_version": self.client.get_server_version(),
            "vehicle": VEHICLE_BLUEPRINT,
            "sync": sync,
        }
        if sync:
            self._original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = dt_s
            self.world.apply_settings(settings)

    def _advance(self, steps: int = 1) -> None:
        for _ in range(steps):
            if self.sync:
                self.world.tick()
                if self.realtime:
                    now = time.time()
                    if self._last_wall is not None:
                        time.sleep(max(0.0, self.dt_s - (now - self._last_wall)))
                    self._last_wall = time.time()
            else:
                time.sleep(self.dt_s)

    def sim_time(self) -> float:
        return float(self.world.get_snapshot().timestamp.elapsed_seconds)

    def _destroy_old_ego(self) -> None:
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "ego":
                actor.destroy()
        self._advance(2)

    def reset(self, raceline: Raceline) -> None:
        carla = self._carla
        self._destroy_old_ego()

        x = float(raceline.x_m[0])
        y = -float(raceline.y_m[0])
        yaw_deg = -math.degrees(float(raceline.psi_rad[0]) + math.pi / 2.0)

        matches = list(self.world.get_blueprint_library().filter(VEHICLE_BLUEPRINT))
        if not matches:
            raise RuntimeError(f"Vehicle blueprint not found: {VEHICLE_BLUEPRINT}")
        bp = matches[0]
        bp.set_attribute("role_name", "ego")

        self.vehicle = self.world.try_spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=x, y=y, z=1.0),
                carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
            ),
        )
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at the raceline start.")

        # Let it drop onto the plane, force it flat, let it settle again.
        self._advance(int(round(1.0 / self.dt_s)))
        flat = self.vehicle.get_transform()
        flat.rotation.pitch = 0.0
        flat.rotation.roll = 0.0
        flat.rotation.yaw = yaw_deg
        flat.location.z += 0.05
        self.vehicle.set_transform(flat)
        self._advance(int(round(0.5 / self.dt_s)))

        try:
            wheels = self.vehicle.get_physics_control().wheels
            self.info["wheel_max_steer_angle_deg"] = [float(w.max_steer_angle) for w in wheels]
        except Exception as exc:  # informational only
            self.info["wheel_max_steer_angle_deg"] = f"unavailable: {type(exc).__name__}"

        try:
            box = self.vehicle.bounding_box
            self.info["bbox_half_extent_m"] = [float(box.extent.x), float(box.extent.y), float(box.extent.z)]
            self.info["bbox_center_offset_m"] = [float(box.location.x), float(box.location.y), float(box.location.z)]
        except Exception as exc:  # informational only
            self.info["bbox_half_extent_m"] = f"unavailable: {type(exc).__name__}"

        # Ground-truth contact detection: the simulator reports what the car touched.
        try:
            sensor_bp = self.world.get_blueprint_library().find("sensor.other.collision")
            self._collision_sensor = self.world.spawn_actor(
                sensor_bp, carla.Transform(), attach_to=self.vehicle
            )
            self._collision_sensor.listen(self._on_collision)
            self.info["collision_sensor"] = True
        except Exception as exc:
            self._collision_sensor = None
            self.info["collision_sensor"] = f"unavailable: {type(exc).__name__}: {exc}"
        self._advance(2)
        self._collisions.clear()  # ignore anything from the spawn drop
        self._end_frame = None

    def _on_collision(self, event) -> None:
        other = event.other_actor
        type_id = other.type_id if other is not None else "unknown"
        actor_id = int(other.id) if other is not None else -1
        self._collisions.append((int(event.frame), type_id, actor_id))

    def mark_end(self) -> None:
        """Contacts after this frame (the braking phase) are not counted."""
        self._end_frame = int(self.world.get_snapshot().frame)

    def collision_summary(self) -> dict:
        if self.info.get("collision_sensor") is not True:
            return {"sim_cones_hit": None, "sim_collision_events": None}
        events = [e for e in self._collisions if self._end_frame is None or e[0] <= self._end_frame]
        cone_ids = {actor_id for _, type_id, actor_id in events if "wemas" in type_id or "cone" in type_id}
        types: dict[str, int] = {}
        for _, type_id, _ in events:
            types[type_id] = types.get(type_id, 0) + 1
        return {
            "sim_cones_hit": len(cone_ids),
            "sim_collision_events": len(events),
            "sim_collision_types": types,
        }

    def state(self) -> tuple[float, float, float, float, float]:
        tf = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        yaw_carla_deg = float(tf.rotation.yaw)
        return (
            float(tf.location.x),
            -float(tf.location.y),
            -math.radians(yaw_carla_deg),
            speed,
            yaw_carla_deg,
        )

    def apply(self, command: float, throttle: float, brake: float) -> float:
        """command: normalised steering in [-1, 1], + = left (track frame)."""
        steer_norm = clamp(-command, -1.0, 1.0)      # CARLA: + = right
        self.vehicle.apply_control(
            self._carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer_norm),
                brake=float(brake),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1,
            )
        )
        return steer_norm

    def step(self) -> None:
        self._advance(1)

    def draw_lookahead(self, overlay: dict) -> None:
        """Draw the lookahead ABOVE the car (see lookahead_overlay).

        It has to be raised: with a 2-3 m lookahead the target point is under or just
        ahead of the 4.2 m long car, so anything on the ground is hidden from a chase
        camera by the car body.
        """
        carla = self._carla
        debug = self.world.debug
        # Slightly longer than one step. A lifetime shorter than the step expires before
        # the frame is rendered and nothing shows.
        life = 1.5 * self.dt_s
        top = 2.3                       # metres above the ground: clear of the roof

        def loc(point, height):
            return carla.Location(x=float(point[0]), y=-float(point[1]), z=height)

        r, g, b = overlay["color"]
        color = carla.Color(r, g, b)
        white = carla.Color(255, 255, 255)
        grey = carla.Color(170, 170, 170)

        # Raised ribbon following the raceline: its length IS the lookahead distance.
        path = overlay["path"]
        for start, end in zip(path[:-1], path[1:]):
            debug.draw_line(loc(start, top), loc(end, top), thickness=0.12, color=color, life_time=life)
        # A white dot per metre along the ribbon.
        for point in overlay["ticks"]:
            debug.draw_point(loc(point, top + 0.02), size=0.14, color=white, life_time=life)
        # Posts at the shortest and longest lookahead the controller may use.
        for point in overlay["limits"]:
            debug.draw_line(loc(point, 0.0), loc(point, top + 0.5), thickness=0.04, color=grey, life_time=life)
        # Target point: a post from the ground up to the ribbon, and a ring on the ground.
        target = overlay["target"]
        debug.draw_line(loc(target, 0.0), loc(target, top), thickness=0.08, color=color, life_time=life)
        debug.draw_point(loc(target, top), size=0.22, color=color, life_time=life)
        ring = overlay["ring"]
        for start, end in zip(ring, ring[1:] + ring[:1]):
            debug.draw_line(loc(start, 0.08), loc(end, 0.08), thickness=0.06, color=color, life_time=life)
        # Text persists longer than lines in CARLA, so refresh it a few times a second
        # only; drawing it every step stacks several copies on top of each other.
        self._label_countdown = getattr(self, "_label_countdown", 0) - 1
        if self._label_countdown <= 0:
            self._label_countdown = max(1, int(round(0.3 / self.dt_s)))
            debug.draw_string(loc(target, top + 0.6), overlay["label"], draw_shadow=True,
                              color=color, life_time=0.3)

    def stop_vehicle(self, keep: bool) -> None:
        if self.vehicle is None:
            return
        try:
            self.vehicle.apply_control(
                self._carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            )
            self._advance(int(round(1.0 / self.dt_s)))
            if self._collision_sensor is not None:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            if not keep:
                self.vehicle.destroy()
                self._advance(2)
        finally:
            self.vehicle = None
            self._collision_sensor = None

    def close(self) -> None:
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            self._original_settings = None


class KinematicPlant:
    """Bicycle model for dry runs without CARLA, using the vehicle file's measured
    steering response (so it turns less at speed, like the CARLA car). Still not a
    vehicle-dynamics model: no slip angles, no load transfer. It proves the code path."""

    def __init__(self, dt_s: float, steering=None) -> None:
        self.dt_s = dt_s
        # The plant's own steering response. Default: the controller's vehicle file. Another
        # file (--plant-vehicle) tests what happens when the controller's model is wrong.
        self.steering = steering
        self.info = {"plant": "kinematic", "vehicle": "kinematic bicycle", "sync": True}
        self._t = 0.0

    def sim_time(self) -> float:
        return self._t

    def reset(self, raceline: Raceline) -> None:
        self.x = float(raceline.x_m[0])
        self.y = float(raceline.y_m[0])
        self.yaw = raceline.geometric_heading_at(0)
        self.v = 0.0
        self.delta = 0.0
        self._cmd = (0.0, 0.0, 0.0)

    def state(self) -> tuple[float, float, float, float, float]:
        return self.x, self.y, self.yaw, self.v, -math.degrees(self.yaw)

    def apply(self, command: float, throttle: float, brake: float) -> float:
        self._cmd = (clamp(command, -1.0, 1.0), throttle, brake)
        return -self._cmd[0]

    def step(self) -> None:
        dt = self.dt_s
        command, throttle, brake = self._cmd
        steer_cmd = (self.steering or VEHICLE.steering).angle(command, self.v)
        self.delta += (steer_cmd - self.delta) * dt / (0.05 + dt)  # steering lag
        accel = 4.5 * throttle - 9.0 * brake - 0.0035 * self.v * self.v - 0.1
        self.v = max(0.0, self.v + accel * dt)
        beta = math.atan(0.5 * math.tan(self.delta))  # reference point = car centre
        self.x += self.v * math.cos(self.yaw + beta) * dt
        self.y += self.v * math.sin(self.yaw + beta) * dt
        self.yaw += self.v / WHEELBASE_M * math.cos(beta) * math.tan(self.delta) * dt
        self._t += dt

    def mark_end(self) -> None:
        pass

    def draw_lookahead(self, *args) -> None:
        pass

    def collision_summary(self) -> dict:
        return {"sim_cones_hit": None, "sim_collision_events": None}

    def stop_vehicle(self, keep: bool) -> None:
        pass

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------

# Steering model used by the delay predictor (--predict-model):
#   vehicle = the vehicle file's identified response, through the active steering map
#   v14     = first fitted version: gain 2.2 / (1 + 0.02 v^2) on the legacy command
#   simple  = constant gain 0.86 (the first predictor)
PREDICT_MODEL = "vehicle"


def _effective_angle(delta_rad: float, speed_mps: float) -> float:
    if PREDICT_MODEL == "vehicle":
        return VEHICLE.steering.angle(steering_command(delta_rad, speed_mps), speed_mps)
    clamped = clamp(delta_rad, -MAX_STEER_RAD, MAX_STEER_RAD)
    if PREDICT_MODEL == "v14":
        return clamped * 2.2 / (1.0 + 0.02 * speed_mps * speed_mps)
    return clamped * 0.86


def predict_pose(x: float, y: float, yaw: float, v: float, pending, dt: float):
    """Delay compensation: where the car will be when a command computed now takes effect.

    The commands already sent but not yet applied are known exactly, so the pose is
    advanced through them with a bicycle model (speed held constant over the delay).
    The controller then steers from that predicted pose instead of the current one.
    """
    for steering, _, _ in pending:
        delta = _effective_angle(steering, v)
        beta = math.atan(0.5 * math.tan(delta))
        x += v * math.cos(yaw + beta) * dt
        y += v * math.sin(yaw + beta) * dt
        yaw += v / WHEELBASE_M * math.cos(beta) * math.tan(delta) * dt
    return x, y, yaw


LOOKAHEAD_COLORS = {
    "speed": (0, 200, 255),       # cyan: speed is setting the lookahead
    "curvature": (255, 120, 0),   # orange: a corner is shortening it
    "fixed": (255, 255, 255),     # white: fixed lookahead (Pure Pursuit)
}


def lookahead_overlay(raceline: Raceline, lat, controller) -> dict:
    """Geometry, in track coordinates, for showing the lookahead on the ground."""
    s_now = float(raceline.s_m[lat.nearest_index])
    lookahead = max(lat.lookahead_m, 0.5)
    samples = max(2, int(math.ceil(lookahead / 0.5)) + 1)
    path = []
    for i in range(samples):
        point = raceline.target_at_s(s_now + lookahead * i / (samples - 1))
        path.append((point.x_m, point.y_m))

    def on_line(distance_m: float):
        point = raceline.target_at_s(s_now + distance_m)
        return (point.x_m, point.y_m)

    ticks = [on_line(metre) for metre in range(1, int(lookahead) + 1)]

    limits = []
    config = getattr(controller, "config", None)
    if controller.name == "rapp":
        limits = [on_line(config.lookahead_min), on_line(config.lookahead_max)]

    target = (lat.target_x, lat.target_y)
    ring = [
        (target[0] + 0.45 * math.cos(2.0 * math.pi * k / 16), target[1] + 0.45 * math.sin(2.0 * math.pi * k / 16))
        for k in range(16)
    ]

    term = lat.lookahead_mode.split(",")[0]
    reason = {"speed": "speed", "curvature": "corner", "fixed": "fixed"}.get(term, term)
    if "floor" in lat.lookahead_mode:
        reason += ", at min"
    elif "ceiling" in lat.lookahead_mode:
        reason += ", at max"
    return {
        "path": path,
        "ticks": ticks,
        "limits": limits,
        "target": target,
        "ring": ring,
        "label": f"{lat.lookahead_m:.1f} m  [{reason}]",
        "color": LOOKAHEAD_COLORS.get(term, (255, 255, 255)),
    }


def build_explorer(args) -> ExplorationLateral:
    """First lap without a known track: simulated cone perception -> KinoRRT -> PFPP tracking."""
    import summarize_benchmark as sb

    cones = sb.load_cones(sb.TRACK_WIDTH_M, args.track)   # left/right pairs: even = blue, odd = yellow
    colours = np.arange(len(cones)) % 2
    perception = ConePerception(cones, colours, PerceptionConfig(
        range_m=args.explore_range, fov_deg=args.explore_fov, noise_m=args.explore_noise,
        miss_prob=args.explore_miss, seed=args.explore_seed))
    car = CarGeometry(
        wheelbase_m=WHEELBASE_M, half_length_m=0.5 * VEHICLE.length_m, half_width_m=0.5 * VEHICLE.width_m,
        body_centre_ahead_m=0.5 * WHEELBASE_M,       # rear axle -> body centre (the CARLA actor origin)
    )
    planner = KinoRRT(KinoRRTConfig(
        iterations=args.explore_samples, horizon_m=min(args.explore_range, 15.0),
        min_radius_m=args.explore_min_radius or VEHICLE.planner_min_radius_m(),
        margin_m=args.explore_margin, seed=args.explore_seed), car)
    return ExplorationLateral(
        perception, planner,
        ExplorationConfig(v_max_mps=args.explore_v_max, ay_max_mps2=args.explore_ay, replan_s=args.explore_replan,
                          lookahead_min_m=args.explore_lookahead_min, lookahead_max_m=args.rapp_lookahead_max,
                          lookahead_gain_s=args.rapp_speed_gain),
        wheelbase_m=WHEELBASE_M, max_steer_rad=MAX_STEER_RAD,
    )


def build_controller(args, raceline: Raceline):
    if args.controller == "explore":
        return build_explorer(args)
    if args.controller in ("rapp", "pfpp"):
        return RAPPLateral(
            raceline,
            RAPPConfig(
                wheelbase_m=WHEELBASE_M,
                max_steer_rad=MAX_STEER_RAD,
                speed_gain=args.rapp_speed_gain,
                lookahead_min=args.rapp_lookahead_min,
                lookahead_max=args.rapp_lookahead_max,
                curv_gain=args.rapp_curv_gain,
                delay_comp_s=args.rapp_delay_comp,
            ),
            feedforward=args.feedforward,
        )
    if args.controller == "pp":
        return PurePursuitLateral(
            raceline,
            PurePursuitLateralConfig(
                wheelbase_m=WHEELBASE_M,
                max_steer_rad=MAX_STEER_RAD,
                lookahead_m=args.pp_lookahead,
                lookahead_gain_s=args.pp_gain,
                feedforward=args.feedforward,
            ),
        )
    return CrossTrackPID(
        raceline,
        CrossTrackPIDConfig(kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd, max_steer_rad=MAX_STEER_RAD),
    )


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        dirty = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def video_title(args) -> tuple[str, str]:
    """Title and subtitle shown in the video's bottom bar."""
    extras = args.feedforward and args.predict_delay
    if args.controller == "explore":
        title = "First lap: KinoRRT + PFPP"
    elif args.controller == "pfpp":
        title = "PFPP (ours)"
    elif args.controller == "pp":
        title = (f"Pure pursuit {args.pp_lookahead:g} m" if not args.pp_gain else "Pure pursuit (speed-scaled)")
        title += " + additions" if extras else ""
    elif args.controller == "rapp":
        title = "RAPP" + (" + additions" if extras else "")
    else:
        title = "Cross-track PID"
    subtitle = f"actuation delay {args.delay_steps * args.dt:.2f} s"
    if args.controller == "explore":
        subtitle += f", track unknown, cones seen up to {args.explore_range:g} m, v <= {args.explore_v_max:g} m/s"
    elif "first_lap" in args.raceline.name:
        subtitle += ", first-lap mode"
    if args.track != "hockenheim_fsg":
        subtitle += f", track: {args.track} (unseen)"
    return title, subtitle


def make_recorder(args, plant, raceline: Raceline, out_dir: Path):
    from carla_video import LapRecorder
    import summarize_benchmark as sb

    if args.record == "auto":
        path = PROJECT_ROOT / "outputs" / "videos" / f"{out_dir.name}.mp4"
    else:
        path = Path(args.record)
        if path.suffix.lower() != ".mp4":
            path = path / f"{out_dir.name}.mp4"
        elif args.repeat > 1:
            path = path.with_name(f"{path.stem}_{out_dir.name}.mp4")
    title, subtitle = video_title(args)
    width, height = (int(v) for v in args.record_size.lower().split("x"))
    line_xy = np.column_stack((raceline.x_m, raceline.y_m))
    if args.controller == "explore":
        line_xy = np.zeros((0, 2))                     # the car does not know the track: draw its plan instead
    return LapRecorder(plant, path, title, subtitle, line_xy, sb.load_cones(sb.TRACK_WIDTH_M, args.track),
                       size=(width, height), fps=1.0 / args.dt, gamma=args.record_gamma)


def run_once(args, plant, raceline: Raceline, run_index: int) -> dict:
    controller = build_controller(args, raceline)
    controller.reset()
    explore = args.controller == "explore"
    velocity_pid = VelocityPID(
        VelocityPIDConfig(
            kp=args.speed_kp, ki=args.speed_ki, kd=args.speed_kd, max_throttle=args.max_throttle,
            brake_gain=args.brake_gain, kff_brake=args.kff_brake,
        )
    )
    # Speed reference (--speed-ref), chosen independently of the steering controller:
    #   plan      = planned speed at the car's position, no anticipation
    #   lookahead = RAPP's original rule: lowest planned speed within a braking distance
    #   envelope  = RAPP's braking envelope: slow early enough for every corner ahead
    speed_reference = RAPP(
        raceline,
        RAPPConfig(
            wheelbase_m=WHEELBASE_M, max_steer_rad=MAX_STEER_RAD,
            brake_decel=args.brake_decel, brake_envelope=(args.speed_ref == "envelope"),
        ),
    )

    plant.reset(raceline)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"bench_{args.controller}_{stamp}"
    suffix = 1
    while out_dir.exists():  # two runs inside the same second
        suffix += 1
        out_dir = OUTPUT_ROOT / f"bench_{args.controller}_{stamp}_{suffix}"
    out_dir.mkdir(parents=True)
    log_path = out_dir / "run_log.csv"
    # Written first so a run stopped with Ctrl+C is still labelled; replaced at the end.
    (out_dir / "meta.json").write_text(json.dumps({
        "controller": args.controller,
        "controller_params": controller.params(),
        "speed_scale": args.speed_scale,
        "speed_ref": args.speed_ref,
        "delay_s": round(args.delay_steps * args.dt, 4),
        "predict_delay": bool(args.predict_delay),
        "status": "interrupted",
        "lap_time_s": None,
        **plant.info,
    }, indent=2))
    recorder = make_recorder(args, plant, raceline, out_dir) if args.record else None
    if recorder and explore:
        recorder.explorer = controller                 # draws the tree, the plan and the cone map

    total_points = len(raceline.s_m) - 1
    end_zone = int(0.85 * total_points)
    start_zone = int(0.15 * total_points)
    armed = False
    status = "timeout"
    lap_time = None
    stuck_since = None
    max_abs_xte = 0.0

    dt = args.dt
    # Actuation delay: the command applied now is the one computed delay_steps ago.
    pending = collections.deque([(0.0, 0.0, 0.0)] * args.delay_steps)
    t0 = plant.sim_time()
    step = 0
    print(f"\n[{args.controller}] run {run_index + 1}/{args.repeat} -> {log_path}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)

        while True:
            t = plant.sim_time() - t0
            if t > args.max_time:
                break

            x, y, yaw, v, yaw_carla_deg = plant.state()
            if explore:
                controller.observe(x, y, yaw)            # the detector sees from the true pose
            if args.predict_delay and pending:
                px, py, pyaw = predict_pose(x, y, yaw, v, pending, dt)
                lat = controller.compute(*reference_point(px, py, pyaw), pyaw, v, dt)
                actual_index = raceline.nearest_index(x, y)
                actual_xte = signed_cross_track(raceline, x, y, actual_index)
            else:
                lat = controller.compute(*reference_point(x, y, yaw), yaw, v, dt)
                if explore:
                    actual_index = raceline.nearest_index(x, y)
                    actual_xte = signed_cross_track(raceline, x, y, actual_index)
                else:
                    actual_index = lat.nearest_index if args.reference == "centre" else raceline.nearest_index(x, y)
                    actual_xte = lat.cross_track_m
            s_actual = float(raceline.s_m[actual_index])
            s_now = float(raceline.s_m[actual_index if explore else lat.nearest_index])
            if explore:
                target_speed, target_accel = controller.speed_target()
            elif args.speed_ref == "plan":
                ref = raceline.target_at_s(s_now)
            else:
                ref, _ = speed_reference.speed_target(s_now, v)
            if not explore:
                target_speed = ref.v_target_mps * args.speed_scale
                target_accel = ref.a_target_mps2 * args.speed_scale**2
            lon = velocity_pid.update(target_speed, target_accel, v, dt)
            if args.draw and not explore:
                plant.draw_lookahead(lookahead_overlay(raceline, lat, controller))
            pending.append((lat.steering, lon.throttle, lon.brake))
            delta_cmd, throttle_cmd, brake_cmd = pending.popleft()
            steer_norm = plant.apply(steering_command(delta_cmd, v), throttle_cmd, brake_cmd)

            if actual_index > end_zone:
                armed = True
            lap_done = armed and actual_index < start_zone and t > 20.0
            max_abs_xte = max(max_abs_xte, abs(actual_xte))

            writer.writerow([
                f"{t:.3f}", f"{x:.6f}", f"{y:.6f}", f"{yaw_carla_deg:.6f}", f"{yaw:.6f}", f"{v:.6f}",
                f"{lat.steering:.6f}", f"{steer_norm:.6f}", f"{target_speed:.6f}", f"{target_accel:.6f}",
                f"{lon.throttle:.6f}", f"{lon.brake:.6f}", lat.nearest_index, lat.target_index,
                f"{lat.target_x:.6f}", f"{lat.target_y:.6f}", int(lap_done),
                f"{s_actual:.3f}", f"{actual_xte:.6f}", f"{lat.lookahead_m:.3f}",
            ])

            if step % int(round(1.0 / dt)) == 0:
                print(
                    f"t={t:6.2f}s | idx={actual_index:4d}/{total_points} | v={v:5.2f} "
                    f"(ref {target_speed:5.2f}) | L={lat.lookahead_m:3.1f} m | xte={actual_xte:+.2f} m | "
                    f"steer={steer_norm:+.2f} thr={lon.throttle:.2f} brk={lon.brake:.2f}"
                )

            if lap_done:
                status, lap_time = "completed", t
                break
            if abs(actual_xte) > args.abort_xte:
                status = "off_track"
                break
            if t > 5.0 and v < 0.5:
                stuck_since = t if stuck_since is None else stuck_since
                if t - stuck_since > 8.0:
                    status = "stuck"
                    break
            else:
                stuck_since = None

            if recorder:
                recorder.pre_tick()
            plant.step()
            step += 1
            if recorder:
                recorder.capture(t + dt, v, lat.cross_track_m if explore else actual_xte,
                                 (lat.target_x, lat.target_y), lat.lookahead_m)

    plant.mark_end()
    video = None
    if recorder:
        cones = len(recorder.hits)
        result = (f"lap {lap_time:.2f} s, {cones} cone{'s' if cones != 1 else ''}"
                  if status == "completed" else status.replace("_", " "))
        video = recorder.finish(result)
        print(f"Video written to {video}")
    plant.stop_vehicle(keep=args.keep_ego)
    collisions = plant.collision_summary()

    meta = {
        "controller": args.controller,
        "controller_params": controller.params(),
        "velocity_pid": velocity_pid.params(),
        "speed_scale": args.speed_scale,
        "speed_ref": args.speed_ref,
        "brake_decel": args.brake_decel,
        "delay_steps": args.delay_steps,
        "delay_s": round(args.delay_steps * dt, 4),
        "predict_delay": bool(args.predict_delay),
        "predict_model": args.predict_model if args.predict_delay else None,
        "steer_map": STEER_MAP,
        "reference": REFERENCE,
        "feedforward": bool(args.feedforward),
        "vehicle_params": VEHICLE.to_dict(),
        "dt_s": dt,
        "wheelbase_m": WHEELBASE_M,
        "max_steer_rad": MAX_STEER_RAD,
        "raceline": args.raceline.name,
        "track": args.track,
        "grip_cap_mps2": args.grip_cap,
        "video": video,
        "status": status,
        "lap_time_s": lap_time,
        "elapsed_s": round(t, 3),
        "steps": step,
        "max_abs_xte_m": round(max_abs_xte, 4),
        "git_commit": git_commit(),
        "timestamp": stamp,
        **collisions,
        **plant.info,
    }
    if explore:
        meta["exploration"] = controller.stats()
        meta["max_time_s"] = args.max_time
        save_exploration(out_dir / "exploration.npz", controller)
        st = meta["exploration"]
        print(f"Exploration: {st['replans']} replans ({st['failed_replans']} kept the previous path), planning "
              f"{st['plan_ms_mean']:.1f} ms mean / {st['plan_ms_p95']:.1f} ms 95% / {st['plan_ms_max']:.1f} ms max, "
              f"{st['cones_mapped']} cones mapped")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Cones in this level have no collision, so the simulator cannot count contacts.
    hit_text = "cones: run summarize_benchmark.py"
    if status == "completed":
        print(f"RESULT [{args.controller}]: lap completed in {lap_time:.2f} s "
              f"(max |xte| {max_abs_xte:.2f} m, {hit_text})")
    else:
        print(f"RESULT [{args.controller}]: {status.upper()} at t={t:.1f} s "
              f"(max |xte| {max_abs_xte:.2f} m, {hit_text})")
    return meta


def save_exploration(path: Path, explorer: ExplorationLateral) -> None:
    """Cone map and every planned path, for the exploration figures (and a later race line)."""
    paths = explorer.history or [np.zeros((0, 2), np.float32)]
    np.savez_compressed(
        path,
        map_xy=explorer.perception.map_xy, map_col=explorer.perception.map_col,
        ever_seen=explorer.perception.ever_seen,
        plan_t=np.array(explorer.history_t), plan_ms=np.array(explorer.plan_ms),
        replan_t=np.array(explorer.replan_t),
        paths=np.concatenate(paths), path_start=np.cumsum([0] + [len(q) for q in paths])[:-1],
        trees=np.concatenate(explorer.trees) if explorer.trees else np.zeros((0, 5), np.float32),
        tree_start=np.cumsum([0] + [len(q) for q in explorer.trees])[:-1],
        step_m=explorer.planner.c.step_m,
        perception=np.array([explorer.perception.config.range_m, explorer.perception.config.fov_deg,
                             explorer.perception.config.sensor_forward_m]),
    )


def restore_async(host: str, port: int) -> None:
    import carla

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    print("Simulator set back to asynchronous mode.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", choices=("pfpp", "explore", "rapp", "pp", "pid"), default="rapp",
                   help="pfpp = Predictive Feed-forward Pure Pursuit (ours): speed-scaled lookahead "
                        "(0.45 v in [3.5, 6] m), delay prediction, curvature feed-forward, measured "
                        "steering map, rear-axle reference. The flags it switches on are set automatically. "
                        "explore = first lap on an unknown track: cone perception -> KinoRRT -> PFPP tracking "
                        "(the --raceline is then only used to measure progress and error).")
    p.add_argument("--plant", choices=("carla", "kinematic"), default="carla")
    p.add_argument("--repeat", type=int, default=1, help="Number of consecutive laps (each from a fresh spawn).")
    p.add_argument("--speed-scale", type=float, default=1.0, help="Scale the raceline speed profile (1.0 = as planned).")
    p.add_argument("--dt", type=float, default=0.05, help="Fixed simulator timestep in seconds.")
    p.add_argument("--delay-steps", type=int, default=0,
                   help="Actuation delay in timesteps (3 = 0.15 s at the default dt). Same for every controller.")
    p.add_argument("--max-time", type=float, default=None,
                   help="Abort a lap after this much SIMULATOR time (default 240 s, 480 s for explore).")
    p.add_argument("--abort-xte", type=float, default=3.5, help="Abort when |cross-track error| exceeds this (m).")
    p.add_argument("--predict-delay", action="store_true",
                   help="Delay compensation: steer from the pose predicted through the pending commands. "
                        "Works with any controller.")
    p.add_argument("--predict-model", choices=("vehicle", "v14", "simple"), default="vehicle",
                   help="Steering model used by --predict-delay (see PREDICT_MODEL).")
    p.add_argument("--plant-vehicle", default=None,
                   help="Kinematic plant only: steering response of the simulated car from this vehicle file "
                        "(default: same as --vehicle), to test a wrong steering model offline.")
    p.add_argument("--vehicle", default="audi_tt_carla",
                   help="Vehicle file: a name in config/vehicles/ or a path to a JSON file.")
    p.add_argument("--grip-cap", nargs="?", type=float, const=-1.0, default=None,
                   help="Cap the planned speed at sqrt(ay_max/|kappa|). Bare flag: the vehicle's ay_max_mps2; "
                        "or give a value in m/s^2. Applies to every controller (shared speed plan).")
    p.add_argument("--steer-map", choices=("legacy", "calibrated"), default="legacy",
                   help="legacy: command = angle / 0.5 rad. calibrated: invert the car's measured steering response.")
    p.add_argument("--reference", choices=("centre", "rear"), default="centre",
                   help="Point the steering controller works from: body centre (original) or rear axle.")
    p.add_argument("--feedforward", action="store_true",
                   help="Curvature feed-forward for Pure Pursuit and RAPP (raceline curvature replaces the chord estimate).")
    p.add_argument("--no-sync", action="store_true", help="Do not switch CARLA to synchronous mode.")
    p.add_argument("--realtime", action="store_true", help="Pace synchronous ticks to wall-clock (for watching/recording).")
    p.add_argument("--keep-ego", action="store_true", help="Leave the car in the world after the run.")
    p.add_argument("--record", nargs="?", const="auto", default=None,
                   help="Record the lap as video: chase + top-down cameras with overlays (CARLA only). "
                        "Bare flag: outputs/videos/<run>.mp4; or give a folder or an .mp4 path.")
    p.add_argument("--record-size", default="960x540", help="Size of each camera view, WIDTHxHEIGHT.")
    p.add_argument("--record-gamma", type=float, default=1.0,
                   help="Brighten the camera image: below 1 lifts dark areas (e.g. 0.6). 1 = as rendered.")
    p.add_argument("--draw", action="store_true",
                   help="Draw the lookahead on the track (ribbon, metre marks, target, distance). Use with --realtime.")
    p.add_argument("--track", default="hockenheim_fsg",
                   help="Cone layout in the world (for cone counting and the video): hockenheim_fsg or a name "
                        "from inputs/tracks/. Use with the matching --raceline.")
    p.add_argument("--raceline", type=Path, default=RACELINE_PATH,
                   help="Raceline CSV to follow (default: the original Hockenheim line).")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--restore-async", action="store_true", help="Only switch CARLA back to async mode, then exit.")
    g = p.add_argument_group("RAPP (steering only; the shared speed reference is not affected)")
    g.add_argument("--rapp-speed-gain", type=float, default=RAPPConfig.speed_gain)
    g.add_argument("--rapp-lookahead-min", type=float, default=RAPPConfig.lookahead_min)
    g.add_argument("--rapp-lookahead-max", type=float, default=RAPPConfig.lookahead_max)
    g.add_argument("--rapp-curv-gain", type=float, default=RAPPConfig.curv_gain,
                   help="Lookahead <= this * corner radius. 1.0 = original RAPP.")
    g.add_argument("--rapp-delay-comp", type=float, default=RAPPConfig.delay_comp_s,
                   help="Actuation delay RAPP is designed for (s). Adds speed * delay to the lookahead.")
    g = p.add_argument_group("exploration lap (--controller explore)")
    g.add_argument("--explore-v-max", type=float, default=5.0, help="Speed cap on the first lap (m/s).")
    g.add_argument("--explore-ay", type=float, default=3.0, help="Lateral acceleration the local speed plan allows.")
    g.add_argument("--explore-range", type=float, default=15.0, help="Cone detection range (m).")
    g.add_argument("--explore-fov", type=float, default=120.0, help="Cone detection field of view (deg).")
    g.add_argument("--explore-noise", type=float, default=0.0, help="Std dev of detected cone positions (m).")
    g.add_argument("--explore-miss", type=float, default=0.0, help="Probability a visible cone is missed per frame.")
    g.add_argument("--explore-samples", type=int, default=300, help="KinoRRT samples per replan.")
    g.add_argument("--explore-replan", type=float, default=0.2, help="Replanning period (s).")
    g.add_argument("--explore-margin", type=float, default=0.30, help="Smallest car-to-cone gap a plan may have (m).")
    g.add_argument("--explore-min-radius", type=float, default=None,
                   help="Tightest turn the planner may use (m). Default: the vehicle file's min_turn_radius_m.")
    g.add_argument("--explore-seed", type=int, default=1, help="Random seed (planner and perception).")
    g.add_argument("--explore-lookahead-min", type=float, default=2.0,
                   help="Shortest lookahead of the first-lap tracker (m). The race lap's PFPP uses 3.5 m; at "
                        "first-lap speeds that is longer than a hairpin's radius.")
    g = p.add_argument_group("pure pursuit")
    g.add_argument("--pp-lookahead", type=float, default=PurePursuitLateralConfig.lookahead_m)
    g.add_argument("--pp-gain", type=float, default=PurePursuitLateralConfig.lookahead_gain_s)
    g = p.add_argument_group("cross-track PID")
    g.add_argument("--pid-kp", type=float, default=CrossTrackPIDConfig.kp)
    g.add_argument("--pid-ki", type=float, default=CrossTrackPIDConfig.ki)
    g.add_argument("--pid-kd", type=float, default=CrossTrackPIDConfig.kd)
    g = p.add_argument_group("shared velocity PID (keep identical across controllers)")
    g.add_argument("--speed-kp", type=float, default=VelocityPIDConfig.kp)
    g.add_argument("--speed-ki", type=float, default=VelocityPIDConfig.ki)
    g.add_argument("--speed-kd", type=float, default=VelocityPIDConfig.kd)
    g.add_argument("--max-throttle", type=float, default=VelocityPIDConfig.max_throttle)
    g = p.add_argument_group("speed target and braking (independent of the steering controller)")
    g.add_argument("--speed-ref", choices=("plan", "lookahead", "envelope"), default="lookahead",
                   help="plan: planned speed here, no anticipation. lookahead: original RAPP rule. "
                        "envelope: brake early enough for every corner ahead.")
    g.add_argument("--brake-decel", type=float, default=None,
                   help="Deceleration the speed target assumes (m/s^2). Default 8.0, or 5.0 with envelope.")
    g.add_argument("--brake-gain", type=float, default=None,
                   help="Brake per unit of negative controller output. Default 0.12, or 0.30 with envelope.")
    g.add_argument("--kff-brake", type=float, default=None,
                   help="Brake feed-forward per m/s^2 of requested deceleration. Default 0, or 0.05 with envelope.")
    args = p.parse_args()
    if args.controller == "pfpp":
        # PFPP = the RAPP lookahead law with a 3.5 m minimum, plus every addition.
        args.reference, args.steer_map = "rear", "calibrated"
        args.feedforward = args.predict_delay = True
        if args.rapp_lookahead_min == RAPPConfig.lookahead_min:
            args.rapp_lookahead_min = 3.5
    if args.controller == "explore":
        # Tracking uses the PFPP machinery; the racing line is replaced by the planner's path.
        args.reference, args.steer_map = "rear", "calibrated"
        args.feedforward = args.predict_delay = True
        if args.rapp_lookahead_min == RAPPConfig.lookahead_min:
            args.rapp_lookahead_min = 3.5
        if args.raceline == RACELINE_PATH:
            args.raceline = PROJECT_ROOT / "inputs" / "racelines" / f"{args.track}_first_lap.csv"
            if not args.raceline.exists():
                raise SystemExit(f"{args.raceline} (used only to measure progress) is missing; make it with\n"
                                 f"  python3 scripts/make_first_lap_line.py --track {args.track} --name {args.track}_first_lap")
    if args.max_time is None:
        args.max_time = 480.0 if args.controller == "explore" else 240.0
    global PREDICT_MODEL, VEHICLE, VEHICLE_BLUEPRINT, WHEELBASE_M, MAX_STEER_RAD, STEER_MAP, REFERENCE
    REFERENCE = args.reference
    PREDICT_MODEL = args.predict_model
    VEHICLE = load_vehicle(args.vehicle)
    VEHICLE_BLUEPRINT = VEHICLE.carla_blueprint or VEHICLE_BLUEPRINT
    WHEELBASE_M = VEHICLE.wheelbase_m
    STEER_MAP = args.steer_map
    # The controllers clamp the desired road-wheel angle. Legacy keeps the original 0.5 rad;
    # calibrated allows up to what a full command gives at low speed (the map saturates beyond).
    MAX_STEER_RAD = (VEHICLE.legacy_max_steer_rad if STEER_MAP == "legacy"
                     else VEHICLE.steering.full_command_angle_rad)

    envelope = args.speed_ref == "envelope"
    if args.brake_decel is None:
        args.brake_decel = 5.0 if envelope else RAPPConfig.brake_decel
    if args.brake_gain is None:
        args.brake_gain = 0.30 if envelope else VelocityPIDConfig.brake_gain
    if args.kff_brake is None:
        args.kff_brake = 0.05 if envelope else VelocityPIDConfig.kff_brake

    if args.restore_async:
        restore_async(args.host, args.port)
        return

    raceline = load_raceline(args.raceline)
    if args.grip_cap is not None:
        if args.grip_cap < 0:
            if VEHICLE.ay_max_mps2 is None:
                raise SystemExit(f"--grip-cap: vehicle '{VEHICLE.name}' has no ay_max_mps2; give a value.")
            args.grip_cap = VEHICLE.ay_max_mps2
        capped = apply_grip_cap(raceline, args.grip_cap)
        print(f"Grip cap {args.grip_cap:.1f} m/s^2: planned speed lowered at {capped} of {len(raceline.s_m) - 1} points.")
    if args.record and args.plant != "carla":
        raise SystemExit("--record needs the CARLA plant (the cameras live in the simulator).")
    if args.plant == "carla":
        plant = CarlaPlant(args.dt, sync=not args.no_sync, host=args.host, port=args.port, realtime=args.realtime)
    else:
        plant = KinematicPlant(args.dt, load_vehicle(args.plant_vehicle).steering if args.plant_vehicle else None)

    results = []
    try:
        for i in range(args.repeat):
            results.append(run_once(args, plant, raceline, i))
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
        plant.stop_vehicle(keep=False)
    finally:
        plant.close()

    done = [r for r in results if r["status"] == "completed"]
    print(f"\n{args.controller}: {len(done)}/{len(results)} laps completed", end="")
    if done:
        times = [r["lap_time_s"] for r in done]
        print(f" | lap time min {min(times):.2f} s, mean {sum(times) / len(times):.2f} s")
    else:
        print()


if __name__ == "__main__":
    main()
