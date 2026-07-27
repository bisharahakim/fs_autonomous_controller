from pathlib import Path

import carla
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    data = np.loadtxt(RACELINE_PATH, delimiter=";", comments="#")
    x = data[:, 1]
    y = -data[:, 2]

    life = 100000.0

    for i in range(0, len(x) - 20, 80):
        start = carla.Location(x=float(x[i]), y=float(y[i]), z=1.0)
        end = carla.Location(x=float(x[i + 20]), y=float(y[i + 20]), z=1.0)

        world.debug.draw_arrow(
            start,
            end,
            thickness=0.12,
            arrow_size=0.8,
            color=carla.Color(255, 0, 255),
            life_time=life,
            persistent_lines=True,
        )

    print("Magenta arrows show raceline direction.")


if __name__ == "__main__":
    main()
