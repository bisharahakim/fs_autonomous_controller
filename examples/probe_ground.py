"""Find how far the drivable ground of the current CARLA level extends.

    python3 examples/probe_ground.py

Casts rays straight down on a grid around the origin and prints where there is ground.
Read-only: it does not spawn anything or change world settings.
"""

from __future__ import annotations

import sys


def main() -> None:
    import carla

    world = carla.Client("localhost", 2000).get_world()
    pass

    def ground(x: float, y: float):
        start = carla.Location(x=x, y=y, z=50.0)
        try:                                            # CARLA >= 0.9.12
            hit = world.ground_projection(start, 200.0)
            return None if hit is None else hit.location.z
        except AttributeError:
            hits = world.cast_ray(start, carla.Location(x=x, y=y, z=-150.0))
            return hits[0].location.z if hits else None

    for axis, label in ((("x", 1, 0), "+x"), (("x", -1, 0), "-x"), (("y", 0, 1), "+y"), (("y", 0, -1), "-y")):
        _, dx, dy = axis
        last_ok = 0
        for d in range(0, 3001, 50):
            z = ground(dx * d, dy * d)
            if z is None or abs(z) > 5.0:
                break
            last_ok = d
        print(f"  ground along {label}: at least {last_ok} m from the origin")

    print("\nCorners of a square around the origin:")
    for r in (250, 500, 1000, 1500):
        corners = [ground(sx * r, sy * r) for sx in (-1, 1) for sy in (-1, 1)]
        print(f"  +-{r} m: " + ", ".join("ground" if c is not None and abs(c) < 5 else "none" for c in corners))


if __name__ == "__main__":
    sys.exit(main())
