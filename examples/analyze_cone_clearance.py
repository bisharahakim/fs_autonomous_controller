from pathlib import Path
import csv
import math
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.hockenheim_fsg_track import build_fsg_hockenheim_cones


LOG_ROOT = PROJECT_ROOT / "outputs" / "carla"


def latest_log_path() -> Path:
    logs = []
    logs += list(LOG_ROOT.glob("rapp_smoke_*/run_log.csv"))
    logs += list(LOG_ROOT.glob("rapp_lap_*/run_log.csv"))

    if not logs:
        raise RuntimeError("No run_log.csv found.")

    return max(logs, key=lambda path: path.stat().st_mtime)


def widen_cones(left, right, target_width_m):
    wide_left = []
    wide_right = []

    for l, r in zip(left, right):
        lx, ly = float(l.x), float(l.y)
        rx, ry = float(r.x), float(r.y)

        cx = 0.5 * (lx + rx)
        cy = 0.5 * (ly + ry)

        vx = lx - rx
        vy = ly - ry
        norm = math.sqrt(vx * vx + vy * vy)

        if norm < 1e-9:
            continue

        ux = vx / norm
        uy = vy / norm

        half_width = 0.5 * target_width_m

        wide_left.append((cx + half_width * ux, cy + half_width * uy, "left"))
        wide_right.append((cx - half_width * ux, cy - half_width * uy, "right"))

    return wide_left + wide_right


def get_float(row, names, default=None):
    for name in names:
        if name in row and row[name] != "":
            return float(row[name])
    if default is not None:
        return default
    raise KeyError(f"None of these fields found in log: {names}")


def load_log(log_path):
    rows = []

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            t = get_float(row, ["t_s", "time_s", "time"])
            x = get_float(row, ["x_m", "x"])
            y = get_float(row, ["y_m", "y"])
            v = get_float(row, ["v_mps", "speed_mps", "speed"], default=0.0)

            if "yaw_rad" in row:
                yaw = float(row["yaw_rad"])
            elif "psi_rad" in row:
                yaw = float(row["psi_rad"])
            elif "yaw_deg" in row:
                yaw = math.radians(float(row["yaw_deg"]))
            else:
                yaw = None

            rows.append([t, x, y, v, yaw])

    if not rows:
        raise RuntimeError("Log is empty.")

    data = np.array(rows, dtype=object)

    # If yaw was not logged, estimate heading from motion.
    if any(r[4] is None for r in rows):
        xs = np.array([float(r[1]) for r in rows])
        ys = np.array([float(r[2]) for r in rows])
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        yaws = np.arctan2(dy, dx)

        for i, r in enumerate(rows):
            r[4] = float(yaws[i])

    return rows


def distance_to_rectangle(local_x, local_y, half_length, half_width):
    """
    Distance from a point to an axis-aligned rectangle centered at origin.
    local_x is forward/backward direction.
    local_y is left/right direction.
    """
    dx = max(abs(local_x) - half_length, 0.0)
    dy = max(abs(local_y) - half_width, 0.0)
    return math.sqrt(dx * dx + dy * dy)


def main():
    # Same width you used in spawn_hockenheim_cones_wide.py
    track_width_m = 5.0

    # Conservative approximate Audi footprint.
    # This does not need to be perfect; it is a safety/clearance check.
    car_length_m = 4.20
    car_width_m = 1.85
    cone_radius_m = 0.20

    half_length = 0.5 * car_length_m
    half_width = 0.5 * car_width_m

    log_path = latest_log_path()
    print("Analyzing:", log_path)

    left, right = build_fsg_hockenheim_cones()
    cones = widen_cones(left, right, track_width_m)

    rows = load_log(log_path)

    hit_count = 0
    near_count = 0
    near_threshold_m = 0.30
    ignore_before_s = 2.0

    best = None

    for t, x, y, v, yaw in rows:
        if t < ignore_before_s:
            continue
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        for cone_x, cone_y, side in cones:
            dx = cone_x - x
            dy = cone_y - y

            # Convert cone position into car local coordinates.
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy

            rect_dist = distance_to_rectangle(local_x, local_y, half_length, half_width)

            # Clearance from car body to cone surface.
            clearance = rect_dist - cone_radius_m

            if clearance <= 0.0:
                hit_count += 1

            if clearance <= near_threshold_m:
                near_count += 1

            if best is None or clearance < best["clearance"]:
                best = {
                    "clearance": clearance,
                    "t": t,
                    "x": x,
                    "y": y,
                    "v": v,
                    "cone_x": cone_x,
                    "cone_y": cone_y,
                    "side": side,
                    "local_x": local_x,
                    "local_y": local_y,
                }

    print()
    print("Cone clearance summary")
    print("----------------------")
    print(f"track width:          {track_width_m:.2f} m")
    print(f"car length assumed:   {car_length_m:.2f} m")
    print(f"car width assumed:    {car_width_m:.2f} m")
    print(f"cone radius assumed:  {cone_radius_m:.2f} m")
    print(f"number of cones:      {len(cones)}")
    print(f"log samples:          {len(rows)}")
    print(f"ignored first:        {ignore_before_s:.1f} s")
    print()

    if hit_count == 0:
        print("Cone hit result:      NO HIT ✅")
    else:
        print("Cone hit result:      POSSIBLE HIT ⚠️")

    print(f"hit samples:          {hit_count}")
    print(f"near samples <=30cm:  {near_count}")

    print()
    print("Closest cone event")
    print("------------------")
    print(f"time:                 {best['t']:.2f} s")
    print(f"speed:                {best['v']:.2f} m/s  ({best['v'] * 3.6:.1f} km/h)")
    print(f"side:                 {best['side']}")
    print(f"clearance:            {best['clearance']:.3f} m")
    print(f"car position:         x={best['x']:.3f}, y={best['y']:.3f}")
    print(f"cone position:        x={best['cone_x']:.3f}, y={best['cone_y']:.3f}")
    print(f"cone in car frame:    forward={best['local_x']:.3f}, side={best['local_y']:.3f}")

    print()
    if best["clearance"] > 0.30:
        print("Interpretation:       Very safe clearance ✅")
    elif best["clearance"] > 0.0:
        print("Interpretation:       Close pass, but no hit ✅")
    else:
        print("Interpretation:       Possible cone contact ⚠️")


if __name__ == "__main__":
    main()
