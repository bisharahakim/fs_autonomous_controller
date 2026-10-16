"""Check what CARLA's RGB camera actually returns (diagnoses black video frames).

    python3 examples/check_camera.py

Puts a camera where the spectator (your CARLA window view) is looking, ticks the world
in synchronous mode, and prints the pixel brightness of each frame for a few camera
settings. Saves one PNG per setting to outputs/videos/camera_check_<setting>.png.
Leaves the world in asynchronous mode at the end.
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "videos"


def main() -> None:
    import carla
    from PIL import Image

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    original = world.get_settings()
    original.synchronous_mode = False               # always leave CARLA free-running afterwards
    original.fixed_delta_seconds = None
    OUT.mkdir(parents=True, exist_ok=True)

    lib = world.get_blueprint_library()
    bp0 = lib.find("sensor.camera.rgb")
    print("Camera attributes in this CARLA version:")
    for attr in bp0:
        try:
            value = attr.as_str()
        except Exception:
            value = "(not readable)"
        print(f"  {attr.id} = {value}")

    spectator = world.get_spectator().get_transform()
    print(f"\nSpectator at x={spectator.location.x:.1f} y={spectator.location.y:.1f} z={spectator.location.z:.1f} "
          f"pitch={spectator.rotation.pitch:.0f} yaw={spectator.rotation.yaw:.0f}")

    variants = {"default": {}}
    if bp0.has_attribute("exposure_mode"):
        variants["manual_exposure"] = {"exposure_mode": "manual"}
    if bp0.has_attribute("enable_postprocess_effects"):
        variants["no_postprocess"] = {"enable_postprocess_effects": "False"}

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    try:
        world.apply_settings(settings)
        for name, attrs in variants.items():
            bp = lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", "640")
            bp.set_attribute("image_size_y", "360")
            bp.set_attribute("fov", "90")
            for k, v in attrs.items():
                bp.set_attribute(k, v)
            cam = world.spawn_actor(bp, spectator)
            q: queue.Queue = queue.Queue()
            cam.listen(q.put)
            print(f"\n[{name}] {attrs or ''}")
            last = None
            for i in range(40):
                frame = world.tick()
                image = q.get(timeout=10.0)
                while image.frame < frame:
                    image = q.get(timeout=10.0)
                arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
                if i in (0, 1, 5, 10, 20, 39):
                    b, g, r, a = (arr[:, :, k] for k in range(4))
                    print(f"  tick {i:2d}: mean R {r.mean():6.1f} G {g.mean():6.1f} B {b.mean():6.1f} "
                          f"| max {arr[:, :, :3].max():3d} | alpha mean {a.mean():6.1f}")
                last = arr
            Image.fromarray(np.ascontiguousarray(last[:, :, 2::-1])).save(OUT / f"camera_check_{name}.png")
            print(f"  saved {OUT / f'camera_check_{name}.png'}")
            cam.stop()
            cam.destroy()
            world.tick()
    finally:
        world.apply_settings(original)
        print("\nWorld settings restored.")


if __name__ == "__main__":
    sys.exit(main())
