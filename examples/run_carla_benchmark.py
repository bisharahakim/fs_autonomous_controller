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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fs_controller.lateral import (  # noqa: E402
    CrossTrackPID,
    CrossTrackPIDConfig,
    PurePursuitLateral,
    PurePursuitLateralConfig,
    RAPPLateral,
)
from fs_controller.raceline import Raceline, load_raceline  # noqa: E402
from fs_controller.rapp import RAPP, RAPPConfig  # noqa: E402
from fs_controller.velocity_pid import VelocityPID, VelocityPIDConfig  # noqa: E402

RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "carla"

VEHICLE_BLUEPRINT = "vehicle.ue4.audi.tt"
WHEELBASE_M = 2.51
MAX_STEER_RAD = 0.5

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

    def apply(self, steering_rad: float, throttle: float, brake: float) -> float:
        steer_norm = clamp(-steering_rad / MAX_STEER_RAD, -1.0, 1.0)
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
    """Tiny bicycle model for dry runs without CARLA. NOT a vehicle-dynamics
    model: no tyre slip, no load transfer. It only proves the code path."""

    def __init__(self, dt_s: float) -> None:
        self.dt_s = dt_s
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

    def apply(self, steering_rad: float, throttle: float, brake: float) -> float:
        self._cmd = (clamp(steering_rad, -MAX_STEER_RAD, MAX_STEER_RAD), throttle, brake)
        return clamp(-steering_rad / MAX_STEER_RAD, -1.0, 1.0)

    def step(self) -> None:
        dt = self.dt_s
        steer_cmd, throttle, brake = self._cmd
        self.delta += (steer_cmd - self.delta) * dt / (0.10 + dt)  # steering lag
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


def build_controller(args, raceline: Raceline):
    if args.controller == "rapp":
        return RAPPLateral(
            raceline,
            RAPPConfig(
                wheelbase_m=WHEELBASE_M,
                max_steer_rad=MAX_STEER_RAD,
                speed_gain=args.rapp_speed_gain,
                lookahead_min=args.rapp_lookahead_min,
                lookahead_max=args.rapp_lookahead_max,
                curv_gain=args.rapp_curv_gain,
            ),
        )
    if args.controller == "pp":
        return PurePursuitLateral(
            raceline,
            PurePursuitLateralConfig(
                wheelbase_m=WHEELBASE_M,
                max_steer_rad=MAX_STEER_RAD,
                lookahead_m=args.pp_lookahead,
                lookahead_gain_s=args.pp_gain,
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


def run_once(args, plant, raceline: Raceline, run_index: int) -> dict:
    controller = build_controller(args, raceline)
    controller.reset()
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
        "status": "interrupted",
        "lap_time_s": None,
        **plant.info,
    }, indent=2))

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
            lat = controller.compute(x, y, yaw, v, dt)
            s_now = float(raceline.s_m[lat.nearest_index])
            if args.speed_ref == "plan":
                ref = raceline.target_at_s(s_now)
            else:
                ref, _ = speed_reference.speed_target(s_now, v)
            target_speed = ref.v_target_mps * args.speed_scale
            target_accel = ref.a_target_mps2 * args.speed_scale**2
            lon = velocity_pid.update(target_speed, target_accel, v, dt)
            if args.draw:
                plant.draw_lookahead(lookahead_overlay(raceline, lat, controller))
            pending.append((lat.steering, lon.throttle, lon.brake))
            steer_norm = plant.apply(*pending.popleft())

            if lat.nearest_index > end_zone:
                armed = True
            lap_done = armed and lat.nearest_index < start_zone and t > 20.0
            max_abs_xte = max(max_abs_xte, abs(lat.cross_track_m))

            writer.writerow([
                f"{t:.3f}", f"{x:.6f}", f"{y:.6f}", f"{yaw_carla_deg:.6f}", f"{yaw:.6f}", f"{v:.6f}",
                f"{lat.steering:.6f}", f"{steer_norm:.6f}", f"{target_speed:.6f}", f"{target_accel:.6f}",
                f"{lon.throttle:.6f}", f"{lon.brake:.6f}", lat.nearest_index, lat.target_index,
                f"{lat.target_x:.6f}", f"{lat.target_y:.6f}", int(lap_done),
                f"{s_now:.3f}", f"{lat.cross_track_m:.6f}", f"{lat.lookahead_m:.3f}",
            ])

            if step % int(round(1.0 / dt)) == 0:
                print(
                    f"t={t:6.2f}s | idx={lat.nearest_index:4d}/{total_points} | v={v:5.2f} "
                    f"(ref {target_speed:5.2f}) | L={lat.lookahead_m:3.1f} m | xte={lat.cross_track_m:+.2f} m | "
                    f"steer={steer_norm:+.2f} thr={lon.throttle:.2f} brk={lon.brake:.2f}"
                )

            if lap_done:
                status, lap_time = "completed", t
                break
            if abs(lat.cross_track_m) > args.abort_xte:
                status = "off_track"
                break
            if t > 5.0 and v < 0.5:
                stuck_since = t if stuck_since is None else stuck_since
                if t - stuck_since > 8.0:
                    status = "stuck"
                    break
            else:
                stuck_since = None

            plant.step()
            step += 1

    plant.mark_end()
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
        "dt_s": dt,
        "wheelbase_m": WHEELBASE_M,
        "max_steer_rad": MAX_STEER_RAD,
        "raceline": args.raceline.name,
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
    p.add_argument("--controller", choices=("rapp", "pp", "pid"), default="rapp")
    p.add_argument("--plant", choices=("carla", "kinematic"), default="carla")
    p.add_argument("--repeat", type=int, default=1, help="Number of consecutive laps (each from a fresh spawn).")
    p.add_argument("--speed-scale", type=float, default=1.0, help="Scale the raceline speed profile (1.0 = as planned).")
    p.add_argument("--dt", type=float, default=0.05, help="Fixed simulator timestep in seconds.")
    p.add_argument("--delay-steps", type=int, default=0,
                   help="Actuation delay in timesteps (3 = 0.15 s at the default dt). Same for every controller.")
    p.add_argument("--max-time", type=float, default=240.0, help="Abort a lap after this much SIMULATOR time.")
    p.add_argument("--abort-xte", type=float, default=3.5, help="Abort when |cross-track error| exceeds this (m).")
    p.add_argument("--no-sync", action="store_true", help="Do not switch CARLA to synchronous mode.")
    p.add_argument("--realtime", action="store_true", help="Pace synchronous ticks to wall-clock (for watching/recording).")
    p.add_argument("--keep-ego", action="store_true", help="Leave the car in the world after the run.")
    p.add_argument("--draw", action="store_true",
                   help="Draw the lookahead on the track (ribbon, metre marks, target, distance). Use with --realtime.")
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
    if args.plant == "carla":
        plant = CarlaPlant(args.dt, sync=not args.no_sync, host=args.host, port=args.port, realtime=args.realtime)
    else:
        plant = KinematicPlant(args.dt)

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
