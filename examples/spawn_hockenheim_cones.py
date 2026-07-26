from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import carla


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.hockenheim_fsg_track import build_fsg_hockenheim_cones


def make_flat_opendrive() -> str:
    """
    Creates a very wide, flat OpenDRIVE road surface.
    This is not a real map. It is only a flat surface for cones.
    """
    return """<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="flat_cone_world" version="1.00" date="2026-07-26"
          north="700" south="-700" east="700" west="-700" vendor="fs_autonomous_controller"/>
  <road name="flat_surface" length="1400.0" id="1" junction="-1">
    <link/>
    <type s="0.0" type="rural"/>
    <planView>
      <geometry s="0.0" x="-700.0" y="0.0" hdg="0.0" length="1400.0">
        <line/>
      </geometry>
    </planView>
    <elevationProfile>
      <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
    </elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false">
            <link/>
            <roadMark sOffset="0.0" type="none" weight="standard" color="standard" width="0.12" laneChange="none"/>
          </lane>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <link/>
            <width sOffset="0.0" a="1400.0" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="none" weight="standard" color="standard" width="0.12" laneChange="none"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


def load_flat_world(client: carla.Client) -> carla.World:
    print("Generating flat OpenDRIVE world...")

    params = carla.OpendriveGenerationParameters()
    params.vertex_distance = 2.0
    params.max_road_length = 500.0
    params.wall_height = 0.0
    params.additional_width = 0.6
    params.smooth_junctions = True
    params.enable_mesh_visibility = True

    xodr = make_flat_opendrive()

    try:
        world = client.generate_opendrive_world(xodr, params, reset_settings=True)
    except TypeError:
        world = client.generate_opendrive_world(xodr, params)

    time.sleep(2.0)
    return world


def get_blueprint(world: carla.World, pattern: str) -> carla.ActorBlueprint:
    matches = list(world.get_blueprint_library().filter(pattern))
    if not matches:
        raise RuntimeError(f"No blueprint found for pattern: {pattern}")
    return matches[0]


def point_to_carla_location(point, z: float, flip_y: bool) -> carla.Location:
    y = -point.y if flip_y else point.y
    return carla.Location(x=float(point.x), y=float(y), z=float(z))


def spawn_cones(
    world: carla.World,
    left_cones,
    right_cones,
    z: float,
    flip_y: bool,
    limit_per_side: int | None,
):
    left_bp = get_blueprint(world, "static.prop.wemas_small_blue")
    right_bp = get_blueprint(world, "static.prop.wemas_small_yellow")

    if limit_per_side is not None:
        left_cones = left_cones[:limit_per_side]
        right_cones = right_cones[:limit_per_side]

    spawned = []

    print(f"Spawning {len(left_cones)} left cones using {left_bp.id}")
    for point in left_cones:
        loc = point_to_carla_location(point, z=z, flip_y=flip_y)
        transform = carla.Transform(loc, carla.Rotation())
        actor = world.try_spawn_actor(left_bp, transform)
        if actor is not None:
            spawned.append(actor)

    print(f"Spawning {len(right_cones)} right cones using {right_bp.id}")
    for point in right_cones:
        loc = point_to_carla_location(point, z=z, flip_y=flip_y)
        transform = carla.Transform(loc, carla.Rotation())
        actor = world.try_spawn_actor(right_bp, transform)
        if actor is not None:
            spawned.append(actor)

    return spawned


def set_birdseye_view(world: carla.World, z: float = 350.0) -> None:
    spectator = world.get_spectator()
    transform = carla.Transform(
        carla.Location(x=80.0, y=40.0, z=z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )
    spectator.set_transform(transform)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty", action="store_true", help="Generate a flat OpenDRIVE world first.")
    parser.add_argument("--flip-y", action="store_true", help="Flip y coordinate if layout looks mirrored.")
    parser.add_argument("--z", type=float, default=0.00, help="Cone spawn height.")
    parser.add_argument("--limit-per-side", type=int, default=None, help="Spawn only N cones per side.")
    parser.add_argument("--keep", action="store_true", help="Leave cones in the world after script exits.")
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    print("Connected to CARLA")
    print("Client version:", client.get_client_version())
    print("Server version:", client.get_server_version())

    if args.empty:
        world = load_flat_world(client)
    else:
        world = client.get_world()

    try:
        print("Current map:", world.get_map().name)
    except RuntimeError as e:
        print("Current map: <custom Unreal level / no OpenDRIVE map available>")
        print("Map warning:", e)

    left_cones, right_cones = build_fsg_hockenheim_cones()
    print("Generated cone layout:")
    print("  left cones:", len(left_cones))
    print("  right cones:", len(right_cones))

    xs = [p.x for p in left_cones + right_cones]
    ys = [p.y for p in left_cones + right_cones]
    print(f"Track bounds: x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}]")

    spawned = spawn_cones(
        world=world,
        left_cones=left_cones,
        right_cones=right_cones,
        z=args.z,
        flip_y=args.flip_y,
        limit_per_side=args.limit_per_side,
    )

    print(f"Actually spawned actors: {len(spawned)}")
    set_birdseye_view(world)

    if args.keep:
        print("Keeping cones in the world. Reload the world later to clear them.")
        return

    input("Look at CARLA now. Press Enter here to destroy the cones and exit...")

    print("Destroying spawned cones...")
    for actor in spawned:
        actor.destroy()

    print("Done.")


if __name__ == "__main__":
    main()
