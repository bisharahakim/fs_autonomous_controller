"""Record a CARLA lap as a video: chase camera and top-down camera side by side.

Used by run_carla_benchmark.py --record. Every simulation step gives one frame, so the
video plays at true speed (20 fps for dt = 0.05 s) however slowly the PC renders.

Drawn on top of the camera images (projected from track coordinates, so it does not
depend on CARLA's debug drawing being visible to cameras):
    green line   racing line
    blue line    path the car has driven (its trail)
    red dot      lookahead target point, with a line from the car and the distance
    red ring     cone hit (flashes for a second, then stays marked)
    bottom bar   controller, delay, time, speed, tracking error, cones hit

Encoding: ffmpeg (H.264, plays in PowerPoint and browsers) if installed, otherwise
OpenCV (mp4v), otherwise numbered PNG frames in a folder.
"""

from __future__ import annotations

import atexit
import math
import queue
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RACING_LINE = (29, 158, 117)
TRAIL = (42, 120, 214)
TARGET = (226, 73, 72)
HIT = (226, 73, 72)
WHITE = (255, 255, 255)
BAR_BG = (24, 24, 23)
BAR_TEXT = (240, 239, 236)
BAR_MUTED = (195, 194, 183)

FLASH_S = 1.0


def _font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        for folder in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
            path = Path(folder) / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


# ----------------------------------------------------------------------------- geometry
def ue_matrix(x: float, y: float, z: float, pitch: float, yaw: float, roll: float) -> np.ndarray:
    """4x4 sensor-to-world matrix for an Unreal/CARLA transform (angles in degrees)."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def project(points_world: np.ndarray, cam_to_world: np.ndarray, width: int, height: int,
            fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of world points (CARLA frame, N x 3) and a mask of those in front."""
    world_to_cam = np.linalg.inv(cam_to_world)
    homog = np.hstack((points_world, np.ones((len(points_world), 1))))
    cam = (world_to_cam @ homog.T).T                       # x forward, y right, z up
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    ahead = cam[:, 0] > 0.3
    depth = np.where(ahead, cam[:, 0], 1.0)
    u = width / 2.0 + focal * cam[:, 1] / depth
    v = height / 2.0 - focal * cam[:, 2] / depth
    return np.column_stack((u, v)), ahead


def track_to_carla(xy: np.ndarray, z: float) -> np.ndarray:
    """Track frame (x, y) -> CARLA frame (x, -y, z); see CarlaPlant."""
    xy = np.atleast_2d(xy)
    return np.column_stack((xy[:, 0], -xy[:, 1], np.full(len(xy), z)))


def polyline_runs(pixels: np.ndarray, valid: np.ndarray, width: int, height: int):
    """Split a projected polyline into drawable pieces (in front of the camera, near the image)."""
    margin = 2 * max(width, height)
    ok = valid & (np.abs(pixels[:, 0] - width / 2) < margin) & (np.abs(pixels[:, 1] - height / 2) < margin)
    run = []
    for keep, p in zip(ok, pixels):
        if keep:
            run.append((float(p[0]), float(p[1])))
        elif run:
            if len(run) > 1:
                yield run
            run = []
    if len(run) > 1:
        yield run


# ----------------------------------------------------------------------------- output
class VideoSink:
    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        self.path, self.size, self.fps = Path(path), size, fps
        self.kind = None
        self._proc = self._writer = None
        self._count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            self._proc = subprocess.Popen(
                [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:g}", "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(self.path)],
                stdin=subprocess.PIPE,
            )
            self.kind = "ffmpeg"
            return
        try:
            import cv2
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if writer.isOpened():
                self._writer, self.kind = writer, "opencv"
                return
        except ImportError:
            pass
        self.frames_dir = self.path.with_suffix("")
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.kind = "png"

    def write(self, rgb: np.ndarray) -> None:
        if self.kind == "ffmpeg":
            self._proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        elif self.kind == "opencv":
            self._writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        else:
            Image.fromarray(rgb).save(self.frames_dir / f"frame_{self._count:05d}.png")
        self._count += 1

    def close(self) -> str:
        if self.kind == "ffmpeg":
            self._proc.stdin.close()
            self._proc.wait()
            return str(self.path)
        if self.kind == "opencv":
            self._writer.release()
            return str(self.path)
        return f"{self.frames_dir}/ (PNG frames; no ffmpeg or OpenCV found)"


# ----------------------------------------------------------------------------- drawing
class Overlay:
    """Draws one camera view. Pure numpy/PIL so it can be tested without CARLA."""

    def __init__(self, width: int, height: int, fov_deg: float, line_width: int, label: str) -> None:
        self.w, self.h, self.fov, self.lw, self.label = width, height, fov_deg, line_width, label
        self.font = _font(max(14, height // 30), bold=True)
        self.small = _font(max(12, height // 38))

    def render(self, rgb: np.ndarray, cam_to_world: np.ndarray, ground_z: float, scene: dict) -> Image.Image:
        img = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        def lines(xy, colour, width):
            if xy is None or len(xy) < 2:
                return
            px, ok = project(track_to_carla(xy, ground_z), cam_to_world, self.w, self.h, self.fov)
            for run in polyline_runs(px, ok, self.w, self.h):
                draw.line(run, fill=colour, width=width, joint="curve")

        def point(xy):
            px, ok = project(track_to_carla(np.asarray(xy, float), ground_z), cam_to_world, self.w, self.h, self.fov)
            return (float(px[0, 0]), float(px[0, 1])) if ok[0] else None

        for piece in scene["line_runs"]:
            lines(piece, RACING_LINE + (230,), self.lw)
        lines(scene["trail_xy"], TRAIL + (255,), self.lw)

        car = point(scene["car_xy"])
        target = point(scene["target_xy"])
        if car and target:
            draw.line([car, target], fill=WHITE + (220,), width=max(2, self.lw - 1))
        if target:
            r = self.lw + 4
            draw.ellipse([target[0] - r, target[1] - r, target[0] + r, target[1] + r],
                         fill=TARGET + (255,), outline=WHITE, width=2)
            draw.text((target[0] + r + 4, target[1] - r - 2), f"L = {scene['lookahead_m']:.1f} m",
                      font=self.small, fill=WHITE, stroke_width=2, stroke_fill=(0, 0, 0))

        for cone_xy, age in scene["hits"]:
            c = point(cone_xy)
            if c is None:
                continue
            flashing = age < FLASH_S
            if flashing and int(age / 0.1) % 2 == 1:
                continue
            r = (self.lw * 5) if flashing else (self.lw * 3)
            draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=HIT + (255,),
                         width=self.lw + (2 if flashing else 0))

        draw.text((14, 10), self.label, font=self.font, fill=WHITE, stroke_width=3, stroke_fill=(0, 0, 0))
        return img


def hud_bar(width: int, height: int, scene: dict) -> Image.Image:
    bar = Image.new("RGB", (width, height), BAR_BG)
    draw = ImageDraw.Draw(bar)
    big, small = _font(int(height * 0.36), bold=True), _font(int(height * 0.28))
    y = int(height * 0.30)
    draw.text((20, y - 6), scene["title"], font=big, fill=BAR_TEXT)
    draw.text((20 + draw.textlength(scene["title"], font=big) + 18, y), scene["subtitle"], font=small, fill=BAR_MUTED)
    stats = (f"t {scene['t']:6.1f} s     v {scene['v']:5.1f} m/s ({scene['v'] * 3.6:5.1f} km/h)     "
             f"error {abs(scene['xte']):.2f} m     cones {len(scene['hits'])}")
    draw.text((width - draw.textlength(stats, font=small) - 20, y), stats, font=small,
              fill=HIT if scene["hits"] else BAR_TEXT)
    return bar


def legend(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(max(12, img.height // 40))
    items = [("racing line", RACING_LINE), ("driven path", TRAIL), ("target point", TARGET), ("cone hit", HIT)]
    x, y = 14, img.height - 26
    total = sum(28 + int(draw.textlength(text, font=font)) + 22 for text, _ in items)
    draw.rounded_rectangle([x - 8, y - 8, x + total - 10, y + 20], radius=6, fill=(0, 0, 0, 150))
    for text, colour in items:
        draw.rectangle([x, y + 5, x + 22, y + 11], fill=colour + (255,))
        draw.text((x + 28, y - 2), text, font=font, fill=WHITE, stroke_width=2, stroke_fill=(0, 0, 0))
        x += 28 + int(draw.textlength(text, font=font)) + 22


def compose(chase: Image.Image, top: Image.Image, bar: Image.Image) -> np.ndarray:
    w = chase.width + top.width
    out = Image.new("RGB", (w, chase.height + bar.height), (0, 0, 0))
    out.paste(chase, (0, 0))
    out.paste(top, (chase.width, 0))
    out.paste(bar, (0, chase.height))
    return np.asarray(out)


# ----------------------------------------------------------------------------- CARLA side
class LapRecorder:
    """Spawns the two cameras on the ego car and writes one composed frame per step."""

    def __init__(self, plant, out_path: Path, title: str, subtitle: str, line_xy: np.ndarray,
                 cones_xy: np.ndarray, size=(960, 540), fps: float = 20.0,
                 top_height_m: float = 32.0, chase_fov: float = 90.0, top_fov: float = 70.0,
                 gamma: float = 1.0) -> None:
        import summarize_benchmark as sb                # cone_clearance, same check as the summary
        self._sb = sb
        self.plant, self.carla, self.world = plant, plant._carla, plant.world
        self.vehicle = plant.vehicle
        self.w, self.h = size
        self.title, self.subtitle = title, subtitle
        self.line_xy, self.cones = line_xy, cones_xy
        self.top_height = top_height_m
        self.ext = plant.info.get("bbox_half_extent_m") or [2.09, 1.0, 0.7]
        self.off = plant.info.get("bbox_center_offset_m") or [0.0, 0.0, 0.0]
        # Brightness curve for the camera image (< 1 lifts the dark asphalt; 1 = unchanged).
        self.lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8) if gamma != 1.0 else None
        self.trail: list[tuple[float, float]] = []
        self.hits: dict[int, float] = {}               # cone index -> time it was hit
        self.t = 0.0
        self.last = None

        carla = self.carla
        lib = self.world.get_blueprint_library()

        def camera(fov):
            bp = lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(self.w))
            bp.set_attribute("image_size_y", str(self.h))
            bp.set_attribute("fov", str(fov))
            # CARLA 0.10 (UE5): with post-processing on, the camera returns black frames in
            # this level; with it off the scene renders (found with examples/check_camera.py).
            if bp.has_attribute("enable_postprocess_effects"):
                bp.set_attribute("enable_postprocess_effects", "False")
            return bp

        self.chase_fov, self.top_fov = chase_fov, top_fov
        self.chase = self.world.spawn_actor(
            camera(chase_fov),
            carla.Transform(carla.Location(x=-7.5, z=3.2), carla.Rotation(pitch=-13.0)),
            attach_to=self.vehicle,
        )
        self.top = self.world.spawn_actor(camera(top_fov), self._top_transform())
        self.q_chase, self.q_top = queue.Queue(), queue.Queue()
        self._prev = [None, None]
        self._missing = [0, 0]
        self.skipped = 0
        atexit.register(self._destroy)                 # no cameras left behind if the run crashes
        self.chase.listen(self.q_chase.put)
        self.top.listen(self.q_top.put)
        self.ov_chase = Overlay(self.w, self.h, chase_fov, 4, "Chase camera")
        self.ov_top = Overlay(self.w, self.h, top_fov, 3, "Top-down")
        self.bar_h = max(56, self.h // 9)
        self.sink = VideoSink(out_path, (2 * self.w, self.h + self.bar_h), fps)

    def _top_transform(self, lead_s: float = 0.0):
        carla = self.carla
        tf = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        # yaw -90: image up = +y in the track frame, image right = +x, same as the report figures.
        return carla.Transform(
            carla.Location(x=tf.location.x + vel.x * lead_s, y=tf.location.y + vel.y * lead_s,
                           z=tf.location.z + self.top_height),
            carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
        )

    def pre_tick(self) -> None:
        """Call just before the world ticks: keeps the top camera over the car."""
        self.top.set_transform(self._top_transform(lead_s=self.plant.dt_s))

    def _drain_to(self, frame: int):
        """The newest image from each camera for this tick.

        In synchronous mode nothing ticks while we wait, so a camera that skipped this
        frame would block forever. Instead: wait briefly, take the newest image there is,
        and if a camera sent nothing, reuse its previous image (counted in self.skipped).
        """
        out = []
        for k, q in enumerate((self.q_chase, self.q_top)):
            image = None
            try:
                image = q.get(timeout=2.0)
                while image.frame < frame:
                    image = q.get(timeout=0.5)
            except queue.Empty:
                pass
            while True:                                # anything newer that is already waiting
                try:
                    image = q.get_nowait()
                except queue.Empty:
                    break
            self._missing[k] = self._missing[k] + 1 if image is None else 0
            if self._missing[k] > 40:                  # 2 s of simulation without a single image
                raise RuntimeError(f"The {('chase', 'top-down')[k]} camera stopped sending images.")
            if image is None:
                image = self._prev[k]
                self.skipped += 1
                if self.skipped in (1, 20, 200):
                    print(f"WARNING: a camera skipped frames ({self.skipped} so far); reusing its last image.")
            if image is None:
                raise RuntimeError("The recording cameras are not sending images.")
            self._prev[k] = image
            out.append(image)
        return out

    def _destroy(self) -> None:
        for cam in (getattr(self, "chase", None), getattr(self, "top", None)):
            if cam is None:
                continue
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        self.chase = self.top = None

    @staticmethod
    def _to_rgb(image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        return arr[:, :, 2::-1]                         # BGRA -> RGB

    @staticmethod
    def _matrix(image) -> np.ndarray:
        tf = image.transform
        return ue_matrix(tf.location.x, tf.location.y, tf.location.z,
                         tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll)

    def capture(self, t: float, v: float, xte: float, target_xy, lookahead_m: float) -> None:
        """Call right after the world ticked."""
        frame = int(self.world.get_snapshot().frame)
        img_chase, img_top = self._drain_to(frame)
        tf = self.vehicle.get_transform()
        x, y = float(tf.location.x), -float(tf.location.y)          # track frame
        yaw = -math.radians(float(tf.rotation.yaw))
        self.t = t
        self.trail.append((x, y))
        clearance = self._sb.cone_clearance(np.array([[x, y]]), np.array([yaw]), self.cones,
                                            self.ext[0], self.ext[1], self.off[0])[0]
        for i in np.where(clearance <= 0.0)[0]:
            self.hits.setdefault(int(i), t)

        near = np.hypot(self.line_xy[:, 0] - x, self.line_xy[:, 1] - y) < 70.0
        scene = {
            "line_runs": self._runs(self.line_xy, near),
            "trail_xy": np.array(self.trail[-2400:]),
            "car_xy": (x, y), "target_xy": tuple(target_xy), "lookahead_m": lookahead_m,
            "hits": [(self.cones[i], t - t_hit) for i, t_hit in self.hits.items()],
            "title": self.title, "subtitle": self.subtitle, "t": t, "v": v, "xte": xte,
        }
        ground = float(tf.location.z) + 0.05
        rgb_chase, rgb_top = self._to_rgb(img_chase), self._to_rgb(img_top)
        if self.lut is not None:
            rgb_chase, rgb_top = self.lut[rgb_chase], self.lut[rgb_top]
        chase = self.ov_chase.render(rgb_chase, self._matrix(img_chase), ground, scene)
        top = self.ov_top.render(rgb_top, self._matrix(img_top), ground, scene)
        legend(top)
        self.last = (chase, top, scene)
        self.sink.write(compose(chase, top, hud_bar(2 * self.w, self.bar_h, scene)))

    @staticmethod
    def _runs(line_xy: np.ndarray, near: np.ndarray) -> list[np.ndarray]:
        """The racing line near the car, as separate continuous pieces.

        Parts of the track that pass close to each other must not be joined by a straight
        jump, and the piece crossing the start/finish is stitched back together.
        """
        idx = np.where(near)[0]
        if len(idx) == 0:
            return []
        breaks = np.where(np.diff(idx) > 1)[0]
        pieces = np.split(idx, breaks + 1)
        if len(pieces) > 1 and pieces[0][0] == 0 and pieces[-1][-1] == len(line_xy) - 1:
            pieces = [np.concatenate((pieces[-1], pieces[0]))] + pieces[1:-1]
        return [line_xy[p] for p in pieces if len(p) > 1]

    def finish(self, result_text: str, hold_s: float = 2.5) -> str:
        if self.last is not None:
            chase, top, scene = self.last
            scene = dict(scene, subtitle=f"{scene['subtitle']}   |   {result_text}")
            frame = compose(chase, top, hud_bar(2 * self.w, self.bar_h, scene))
            for _ in range(int(hold_s * self.sink.fps)):
                self.sink.write(frame)
        self._destroy()
        if self.skipped:
            print(f"Note: {self.skipped} camera frames were skipped and filled with the previous image.")
        return self.sink.close()
