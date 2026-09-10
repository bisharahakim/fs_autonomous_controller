from pathlib import Path
import csv
import math
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"
LOG_ROOT = PROJECT_ROOT / "outputs" / "carla"


def latest_log_path() -> Path:
    logs = []
    logs += list(LOG_ROOT.glob("rapp_smoke_*/run_log.csv"))
    logs += list(LOG_ROOT.glob("rapp_lap_*/run_log.csv"))

    if not logs:
        raise RuntimeError("No run_log.csv found in outputs/carla/rapp_smoke_* or outputs/carla/rapp_lap_*")

    return max(logs, key=lambda path: path.stat().st_mtime)


def main() -> None:
    raceline = np.loadtxt(RACELINE_PATH, delimiter=";", comments="#")
    rx = raceline[:, 1]
    ry = raceline[:, 2]

    log_path = latest_log_path()
    print("Analyzing:", log_path)

    times = []
    errors = []
    speeds = []
    steering = []
    steer_norms = []
    target_speeds = []
    target_accels = []
    throttles = []
    brakes = []
    speed_errors = []
    nearest_indices = []
    target_indices = []

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            t = float(row["t_s"])
            x = float(row["x_m"])
            y = float(row["y_m"])
            v = float(row["v_mps"])
            steer = float(row["steering_rad"])
            steer_norm = float(row["steer_norm"])
            target_speed = float(row["target_speed_mps"])
            target_accel = float(row["target_accel_mps2"])
            throttle = float(row["throttle"])
            brake = float(row["brake"])
            nearest = int(row["nearest_index"])
            target_idx = int(row["target_index"])

            nearest = max(0, min(nearest, len(rx) - 1))
            error = math.sqrt((x - rx[nearest]) ** 2 + (y - ry[nearest]) ** 2)

            times.append(t)
            errors.append(error)
            speeds.append(v)
            steering.append(steer)
            steer_norms.append(steer_norm)
            target_speeds.append(target_speed)
            target_accels.append(target_accel)
            throttles.append(throttle)
            brakes.append(brake)
            speed_errors.append(target_speed - v)
            nearest_indices.append(nearest)
            target_indices.append(target_idx)

    if not times:
        raise RuntimeError("Log file is empty.")

    errors_np = np.array(errors)
    speeds_np = np.array(speeds)
    steering_np = np.array(steering)
    steer_norms_np = np.array(steer_norms)
    target_speeds_np = np.array(target_speeds)
    target_accels_np = np.array(target_accels)
    throttles_np = np.array(throttles)
    brakes_np = np.array(brakes)
    speed_errors_np = np.array(speed_errors)

    max_error_i = int(np.argmax(errors_np))

    print()
    print("Tracking summary")
    print("----------------")
    print(f"duration:            {times[-1]:.2f} s")
    print(f"max speed:           {speeds_np.max():.2f} m/s  ({speeds_np.max() * 3.6:.1f} km/h)")
    print(f"mean speed:          {speeds_np.mean():.2f} m/s  ({speeds_np.mean() * 3.6:.1f} km/h)")
    print(f"mean error:          {errors_np.mean():.2f} m")
    print(f"median error:        {np.median(errors_np):.2f} m")
    print(f"95% error:           {np.percentile(errors_np, 95):.2f} m")
    print(f"max error:           {errors_np.max():.2f} m")
    print(f"max |steer_norm|:    {np.max(np.abs(steer_norms_np)):.2f}")
    print(f"final nearest idx:   {nearest_indices[-1]}")
    print(f"final target idx:    {target_indices[-1]}")

    print()
    print("Speed / throttle summary")
    print("------------------------")
    print(f"max target speed:    {target_speeds_np.max():.2f} m/s  ({target_speeds_np.max() * 3.6:.1f} km/h)")
    print(f"mean target speed:   {target_speeds_np.mean():.2f} m/s  ({target_speeds_np.mean() * 3.6:.1f} km/h)")
    print(f"mean speed error:    {speed_errors_np.mean():+.2f} m/s")
    print(f"max speed error:     {speed_errors_np.max():+.2f} m/s")
    print(f"min speed error:     {speed_errors_np.min():+.2f} m/s")
    print(f"max throttle used:   {throttles_np.max():.2f}")
    print(f"mean throttle used:  {throttles_np.mean():.2f}")
    print(f"max brake used:      {brakes_np.max():.2f}")
    print(f"mean brake used:     {brakes_np.mean():.2f}")

    throttle_high = np.mean(throttles_np > 0.70) * 100.0
    brake_active = np.mean(brakes_np > 0.01) * 100.0
    steer_sat = np.mean(np.abs(steer_norms_np) > 0.95) * 100.0

    print(f"time throttle > 0.70:{throttle_high:.1f}%")
    print(f"time brake active:   {brake_active:.1f}%")
    print(f"time steer saturated:{steer_sat:.1f}%")

    print()
    print("Worst tracking point")
    print("--------------------")
    print(f"t:           {times[max_error_i]:.2f} s")
    print(f"error:       {errors[max_error_i]:.2f} m")
    print(f"speed:       {speeds[max_error_i]:.2f} m/s  ({speeds[max_error_i] * 3.6:.1f} km/h)")
    print(f"target spd:  {target_speeds[max_error_i]:.2f} m/s")
    print(f"speed error: {speed_errors[max_error_i]:+.2f} m/s")
    print(f"steering:    {steering[max_error_i]:+.3f} rad")
    print(f"steer_norm:  {steer_norms[max_error_i]:+.3f}")
    print(f"throttle:    {throttles[max_error_i]:.2f}")
    print(f"brake:       {brakes[max_error_i]:.2f}")
    print(f"nearest:     {nearest_indices[max_error_i]}")
    print(f"target:      {target_indices[max_error_i]}")

    for threshold in [1.0, 2.0, 3.0]:
        bad = np.where(errors_np > threshold)[0]
        print()
        if len(bad) == 0:
            print(f"Error never exceeded {threshold:.1f} m")
        else:
            i = int(bad[0])
            print(f"First time error exceeded {threshold:.1f} m:")
            print(
                f"t={times[i]:.2f}s, "
                f"error={errors[i]:.2f}m, "
                f"speed={speeds[i]:.2f}m/s, "
                f"target_speed={target_speeds[i]:.2f}m/s, "
                f"throttle={throttles[i]:.2f}, "
                f"brake={brakes[i]:.2f}, "
                f"steer_norm={steer_norms[i]:+.3f}"
            )


if __name__ == "__main__":
    main()
