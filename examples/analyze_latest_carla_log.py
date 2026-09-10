from pathlib import Path
import csv
import math
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"
LOG_ROOT = PROJECT_ROOT / "outputs" / "carla"


def latest_log_path() -> Path:
    logs = sorted(LOG_ROOT.glob("rapp_smoke_*/run_log.csv"))
    if not logs:
        raise RuntimeError("No run_log.csv found in outputs/carla/rapp_smoke_*")
    return logs[-1]


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
            nearest = int(row["nearest_index"])
            target_idx = int(row["target_index"])

            nearest = max(0, min(nearest, len(rx) - 1))

            error = math.sqrt((x - rx[nearest]) ** 2 + (y - ry[nearest]) ** 2)

            times.append(t)
            errors.append(error)
            speeds.append(v)
            steering.append(steer)
            steer_norms.append(steer_norm)
            nearest_indices.append(nearest)
            target_indices.append(target_idx)

    if not times:
        raise RuntimeError("Log file is empty.")

    errors_np = np.array(errors)
    speeds_np = np.array(speeds)
    steer_norms_np = np.array(steer_norms)

    max_error_i = int(np.argmax(errors_np))

    print()
    print("Tracking summary")
    print("----------------")
    print(f"duration:           {times[-1]:.2f} s")
    print(f"max speed:          {speeds_np.max():.2f} m/s")
    print(f"mean error:         {errors_np.mean():.2f} m")
    print(f"median error:       {np.median(errors_np):.2f} m")
    print(f"95% error:          {np.percentile(errors_np, 95):.2f} m")
    print(f"max error:          {errors_np.max():.2f} m")
    print(f"max |steer_norm|:   {np.max(np.abs(steer_norms_np)):.2f}")
    print(f"final nearest idx:  {nearest_indices[-1]}")
    print(f"final target idx:   {target_indices[-1]}")

    print()
    print("Worst tracking point")
    print("--------------------")
    print(f"t:          {times[max_error_i]:.2f} s")
    print(f"error:      {errors[max_error_i]:.2f} m")
    print(f"speed:      {speeds[max_error_i]:.2f} m/s")
    print(f"steering:   {steering[max_error_i]:+.3f} rad")
    print(f"steer_norm: {steer_norms[max_error_i]:+.3f}")
    print(f"nearest:    {nearest_indices[max_error_i]}")
    print(f"target:     {target_indices[max_error_i]}")

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
                f"steer_norm={steer_norms[i]:+.3f}"
            )


if __name__ == "__main__":
    main()

