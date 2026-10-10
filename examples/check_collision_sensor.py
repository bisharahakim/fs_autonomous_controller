"""One-off check: does CARLA report a contact when the car drives over a cone?

Places ONE extra cone on the raceline about 12 m ahead of the start, drives
the car straight over it, and prints what the collision sensor reported. The
benchmark's "cones touched" number can only be trusted if this prints YES.

    python3 examples/check_collision_sensor.py

Run with CARLA in Play. The extra cone and the car are removed afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fs_controller.lateral import RAPPLateral  # noqa: E402
from fs_controller.raceline import load_raceline  # noqa: E402
from fs_controller.rapp import RAPPConfig  # noqa: E402
from run_carla_benchmark import MAX_STEER_RAD, RACELINE_PATH, WHEELBASE_M, CarlaPlant  # noqa: E402

CONE_BLUEPRINT = "static.prop.wemas_small_blue"
CONE_DISTANCE_M = 12.0
DRIVE_SECONDS = 8.0


def main() -> None:
    import carla

    raceline = load_raceline(RACELINE_PATH)
    plant = CarlaPlant(dt_s=0.05, sync=True, host="localhost", port=2000, realtime=False)
    cone = None
    try:
        plant.reset(raceline)
        print("Vehicle bounding box half-extent (x, y, z):", plant.info.get("bbox_half_extent_m"))
        print("Bounding box centre offset from the logged position:", plant.info.get("bbox_center_offset_m"))
        print("Collision sensor attached:", plant.info.get("collision_sensor"))

        target = raceline.target_at_s(CONE_DISTANCE_M)
        cone_bp = plant.world.get_blueprint_library().find(CONE_BLUEPRINT)
        cone = plant.world.try_spawn_actor(
            cone_bp,
            carla.Transform(carla.Location(x=target.x_m, y=-target.y_m, z=0.0), carla.Rotation()),
        )
        if cone is None:
            raise RuntimeError("Could not spawn the test cone on the raceline.")
        plant.step()
        before = cone.get_transform().location
        print(f"Test cone placed on the raceline {CONE_DISTANCE_M:.0f} m ahead (actor id {cone.id}).")

        steering = RAPPLateral(raceline, RAPPConfig(wheelbase_m=WHEELBASE_M, max_steer_rad=MAX_STEER_RAD))
        closest = float("inf")
        for _ in range(int(DRIVE_SECONDS / plant.dt_s)):
            x, y, yaw, v, _ = plant.state()
            closest = min(closest, ((x - target.x_m) ** 2 + (y - target.y_m) ** 2) ** 0.5)
            plant.apply(steering.compute(x, y, yaw, v, plant.dt_s).steering, 0.45 if v < 6.0 else 0.0, 0.0)
            plant.step()

        after = cone.get_transform().location
        moved = ((after.x - before.x) ** 2 + (after.y - before.y) ** 2 + (after.z - before.z) ** 2) ** 0.5
        events = list(plant._collisions)
        cone_events = [e for e in events if e[2] == cone.id]

        print(f"\nClosest the car position came to the cone: {closest:.2f} m")
        print(f"Cone moved by: {moved:.2f} m")
        print(f"Collision events in total: {len(events)}")
        for type_id in sorted({e[1] for e in events}):
            print(f"   {type_id}: {sum(1 for e in events if e[1] == type_id)} events")
        if cone_events:
            print("\nRESULT: YES - the simulator reports cone contacts. 'sim' hits in the benchmark can be trusted.")
        elif closest > 1.5:
            print("\nRESULT: INCONCLUSIVE - the car never reached the cone. Paste this output.")
        else:
            print("\nRESULT: NO - the car drove over the cone and no contact was reported. "
                  "'sim' hits in the benchmark are NOT reliable; use the geometric count.")
    finally:
        try:
            if cone is not None:
                cone.destroy()
            plant.stop_vehicle(keep=False)
        finally:
            plant.close()


if __name__ == "__main__":
    main()
