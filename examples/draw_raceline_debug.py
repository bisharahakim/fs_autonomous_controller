from __future__ import annotations

import sys
from pathlib import Path

import carla
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    data = np.loadtxt(RACELINE_PATH, delimiter=";", comments="#")

    x = data[:, 1]
    y = -data[:, 2]

    life = 100000.0
    z = 0.08

    print("Drawing raceline with", len(x), "points")

    # Green raceline
    for i in range(len(x) - 1):
        world.debug.draw_line(
            carla.Location(x=float(x[i]), y=float(y[i]), z=z),
            carla.Location(x=float(x[i + 1]), y=float(y[i + 1]), z=z),
            thickness=0.08,
            color=carla.Color(0, 255, 0),
            life_time=life,
            persistent_lines=True,
        )

    # Big start marker
    world.debug.draw_point(
        carla.Location(x=float(x[0]), y=float(y[0]), z=1.0),
        size=0.35,
        color=carla.Color(255, 0, 0),
        life_time=life,
        persistent_lines=True,
    )

    world.debug.draw_string(
        carla.Location(x=float(x[0]), y=float(y[0]), z=2.0),
        "RACELINE START",
        draw_shadow=True,
        color=carla.Color(255, 0, 0),
        life_time=life,
        persistent_lines=True,
    )

    # Direction dots every 100 points
    for i in range(0, len(x), 100):
        world.debug.draw_point(
            carla.Location(x=float(x[i]), y=float(y[i]), z=0.5),
            size=0.18,
            color=carla.Color(0, 255, 255),
            life_time=life,
            persistent_lines=True,
        )

    print("Done. Green = raceline, red = start, cyan = direction samples.")


if __name__ == "__main__":
    main()
