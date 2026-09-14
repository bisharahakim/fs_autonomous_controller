from pathlib import Path
import numpy as np
import carla


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    data = np.loadtxt(RACELINE_PATH, delimiter=";", comments="#")

    x = data[:, 1]
    y = -data[:, 2]   # CARLA world is mirrored in Y

    z = 0.35
    step = 2          # draw fewer segments so Unreal does not drop parts

    green = carla.Color(r=0, g=255, b=0)
    red = carla.Color(r=255, g=0, b=0)

    print(f"Raceline points in CSV: {len(x)}")
    print(f"Drawing every {step} points")
    print("This should show the full loop without missing sections.")

    indices = list(range(0, len(x), step))

    # Make sure the final CSV point is included
    if indices[-1] != len(x) - 1:
        indices.append(len(x) - 1)

    # Draw loop through downsampled points
    for k in range(len(indices)):
        i = indices[k]
        j = indices[(k + 1) % len(indices)]

        p1 = carla.Location(x=float(x[i]), y=float(y[i]), z=z)
        p2 = carla.Location(x=float(x[j]), y=float(y[j]), z=z)

        world.debug.draw_line(
            p1,
            p2,
            thickness=0.09,
            color=green,
            life_time=999999.0,
            persistent_lines=True,
        )

    # Draw start point in red
    start = carla.Location(x=float(x[0]), y=float(y[0]), z=z + 0.15)
    world.debug.draw_point(
        start,
        size=0.30,
        color=red,
        life_time=999999.0,
        persistent_lines=True,
    )

    print(f"Drawn segments: {len(indices)}")
    print("Done. Green = full downsampled raceline, red = start/finish.")


if __name__ == "__main__":
    main()
