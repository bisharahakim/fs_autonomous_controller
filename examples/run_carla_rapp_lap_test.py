from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT_LOCAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT_LOCAL))

import carla

from fs_controller.rapp import RAPP, RAPPConfig

from run_carla_rapp_smoke_test import (
    PROJECT_ROOT,
    RACELINE_PATH,
    load_raceline,
    spawn_ego_at_raceline_start,
    speed_mps,
    rapp_yaw_rad_from_carla_yaw_deg,
    translate_rapp_to_carla_control,
)


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    print("Connected to CARLA")
    print("Client version:", client.get_client_version())
    print("Server version:", client.get_server_version())

    raceline = load_raceline(RACELINE_PATH)
    print("Loaded raceline points:", len(raceline.s_m))

    config = RAPPConfig()
    rapp = RAPP(raceline, config)

    vehicle = spawn_ego_at_raceline_start(world, raceline)

    out_dir = PROJECT_ROOT / "outputs" / "carla" / f"rapp_lap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.csv"

    print("Logging to:", log_path)

    dt_s = 0.05
    max_time_s = 180.0
    max_throttle = 0.45

    total_points = len(raceline.s_m)
    end_zone_index = int(0.85 * total_points)
    start_zone_index = int(0.15 * total_points)

    reached_end_zone = False
    lap_complete = False
    lap_time = None

    print()
    print("Lap detection:")
    print(f"  end zone index:   {end_zone_index}")
    print(f"  start zone index: {start_zone_index}")
    print("  lap complete when index goes near end, then wraps back near start")
    print()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s",
            "x_m",
            "y_m",
            "yaw_carla_deg",
            "yaw_rapp_rad",
            "v_mps",
            "steering_rad",
            "steer_norm",
            "target_speed_mps",
            "target_accel_mps2",
            "throttle",
            "brake",
            "nearest_index",
            "target_index",
            "target_x",
            "target_y",
            "lap_complete",
        ])

        start_time = time.time()
        step = 0

        while True:
            now = time.time()
            t = now - start_time

            if t > max_time_s:
                print("Reached max_time_s before detecting lap completion.")
                break

            transform = vehicle.get_transform()

            carla_x = float(transform.location.x)
            carla_y = float(transform.location.y)
            yaw_carla_deg = float(transform.rotation.yaw)

            # Convert CARLA mirrored coordinates back to RAPP track coordinates.
            x = carla_x
            y = -carla_y

            yaw_rapp_rad = rapp_yaw_rad_from_carla_yaw_deg(yaw_carla_deg)
            v = speed_mps(vehicle)

            cmd = rapp.compute_control(x, y, yaw_rapp_rad, v)

            control = translate_rapp_to_carla_control(
                cmd=cmd,
                current_speed=v,
                max_steer_rad=config.max_steer_rad,
                max_throttle=max_throttle,
            )

            vehicle.apply_control(control)

            nearest_index = int(cmd.nearest_index)

            if nearest_index > end_zone_index:
                reached_end_zone = True

            if reached_end_zone and nearest_index < start_zone_index and t > 20.0:
                lap_complete = True
                lap_time = t

            writer.writerow([
                f"{t:.3f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{yaw_carla_deg:.6f}",
                f"{yaw_rapp_rad:.6f}",
                f"{v:.6f}",
                f"{cmd.steering:.6f}",
                f"{control.steer:.6f}",
                f"{cmd.target_speed:.6f}",
                f"{cmd.target_accel:.6f}",
                f"{control.throttle:.6f}",
                f"{control.brake:.6f}",
                nearest_index,
                cmd.target_index,
                f"{cmd.target_x:.6f}",
                f"{cmd.target_y:.6f}",
                int(lap_complete),
            ])

            if step % 20 == 0:
                print(
                    f"t={t:6.2f}s | "
                    f"x={x:7.2f}, y={y:7.2f}, "
                    f"v={v:5.2f}, "
                    f"err idx={nearest_index:4d}/{total_points}, "
                    f"steer={cmd.steering:+.3f}, "
                    f"steer_norm={control.steer:+.3f}, "
                    f"thr={control.throttle:.2f}, brk={control.brake:.2f}"
                )

            if lap_complete:
                print()
                print("FULL LAP COMPLETE")
                print(f"Lap time: {lap_time:.2f} s")
                break

            step += 1
            time.sleep(dt_s)

    vehicle.apply_control(
        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    )

    time.sleep(1.0)

    end = vehicle.get_transform()
    print()
    print("Done.")
    print(f"Final position: x={end.location.x:.3f}, y={end.location.y:.3f}, yaw={end.rotation.yaw:.2f}")
    print("Log saved to:", log_path)

    if lap_time is not None:
        print(f"Final result: completed one lap in {lap_time:.2f} s")
    else:
        print("Final result: lap was not completed before timeout.")


if __name__ == "__main__":
    main()
