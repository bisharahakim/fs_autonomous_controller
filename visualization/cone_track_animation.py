from __future__ import annotations

import math
import tkinter as tk
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from track_registry import TrackDefinition, get_track
    from autocross_track import TrackPoint
except ModuleNotFoundError:
    from visualization.track_registry import TrackDefinition, get_track
    from visualization.autocross_track import TrackPoint

from fs_controller import (
    MAX_SPEED_30_KMH_MPS,
    PowertrainConfig,
    PowertrainModel,
    apply_friction_circle,
    requested_acceleration_from_actuators,
)
from fs_controller.raceline import Raceline, load_raceline, wrap_angle


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


class ConeTrackAnimation:
    def __init__(self, track: TrackDefinition, raceline_path: Path | None = None) -> None:
        self.root = tk.Tk()
        self.root.title("Formula Student Pure Pursuit Animation")
        self.root.geometry("1260x800")
        self.root.minsize(980, 650)

        self.track = track
        self.wheelbase_m = 1.55
        self.car_length_m = 2.8
        self.car_width_m = 1.1
        self.track_width_m = track.track_width_m
        self.cone_spacing_m = track.cone_spacing_m
        self.min_lookahead_m = 3.0
        self.max_lookahead_m = 18.0
        self.max_steer_rad = 0.5
        self.dt_s = 0.02
        self.powertrain = PowertrainModel()
        self.vehicle_config = PowertrainConfig()
        self.raceline: Raceline | None = load_raceline(raceline_path) if raceline_path is not None else None
        self.raceline_points = (
            [TrackPoint(float(x), float(y)) for x, y in zip(self.raceline.x_m, self.raceline.y_m)]
            if self.raceline is not None
            else []
        )
        self.safe_braking_accel_mps2 = 8.0
        self.safe_throttle_accel_mps2 = 5.0
        self.raceline_speed_kp = 2.0
        self.raceline_speed_ki = 0.5
        self.raceline_integral_limit_mps2 = 5.0
        self.max_jerk_throttle_mps3 = 8.0
        self.max_jerk_brake_mps3 = 20.0

        self.path = track.build_centerline()
        self.left_cones, self.right_cones = track.build_cones()
        self.running = True
        self.speed_scale = tk.DoubleVar(value=1.0)
        self.lookahead_gain = tk.DoubleVar(value=0.2)
        self.zoom = tk.DoubleVar(value=1.0)
        self.follow_car = tk.BooleanVar(value=True)
        self.view_center_x = self.path[0].x
        self.view_center_y = self.path[0].y
        self._drag_start: tuple[float, float, float, float] | None = None
        self.trajectory: list[TrackPoint] = []
        self.target_marker = self.path[0]

        self._build_ui()
        self.reset()
        self._tick()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = tk.Frame(self.root, bg="#151a1f", padx=12, pady=10)
        toolbar.grid(row=0, column=0, sticky="ew")

        title = tk.Label(
            toolbar,
            text=f"{self.track.name} {'Raceline' if self.raceline is not None else 'Pure Pursuit'}",
            bg="#151a1f",
            fg="#f3f5f4",
            font=("Arial", 18, "bold"),
        )
        title.pack(side="left", padx=(0, 18))

        self.play_button = tk.Button(toolbar, text="Pause", width=8, command=self.toggle)
        self.play_button.pack(side="left", padx=4)

        reset_button = tk.Button(toolbar, text="Reset", width=8, command=self.reset)
        reset_button.pack(side="left", padx=4)

        follow_button = tk.Button(toolbar, text="Follow", width=8, command=self.follow_vehicle)
        follow_button.pack(side="left", padx=4)

        fit_button = tk.Button(toolbar, text="Fit", width=8, command=self.fit_track)
        fit_button.pack(side="left", padx=4)

        tk.Label(toolbar, text="Speed", bg="#151a1f", fg="#c7d0d6").pack(
            side="left", padx=(22, 6)
        )
        tk.Scale(
            toolbar,
            from_=0.25,
            to=2.5,
            resolution=0.05,
            orient="horizontal",
            variable=self.speed_scale,
            length=150,
            showvalue=False,
            bg="#151a1f",
            fg="#f3f5f4",
            highlightthickness=0,
        ).pack(side="left")

        tk.Label(toolbar, text="Lookahead", bg="#151a1f", fg="#c7d0d6").pack(
            side="left", padx=(22, 6)
        )
        tk.Scale(
            toolbar,
            from_=0.15,
            to=0.8,
            resolution=0.01,
            orient="horizontal",
            variable=self.lookahead_gain,
            length=150,
            showvalue=False,
            bg="#151a1f",
            fg="#f3f5f4",
            highlightthickness=0,
        ).pack(side="left")

        tk.Label(toolbar, text="Zoom", bg="#151a1f", fg="#c7d0d6").pack(
            side="left", padx=(22, 6)
        )
        tk.Scale(
            toolbar,
            from_=0.2,
            to=4.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.zoom,
            length=130,
            showvalue=False,
            bg="#151a1f",
            fg="#f3f5f4",
            highlightthickness=0,
        ).pack(side="left")

        content = tk.Frame(self.root, bg="#101316")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(content, bg="#1f261f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._begin_pan)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<MouseWheel>", self._zoom_with_wheel)
        self.canvas.bind("<Button-4>", self._zoom_with_wheel)
        self.canvas.bind("<Button-5>", self._zoom_with_wheel)

        panel = tk.Frame(content, width=280, bg="#151a1f", padx=16, pady=14)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        self.metrics: dict[str, tk.Label] = {}
        metric_names = [
            ("Track width", "track_width"),
            ("Car width", "car_width"),
            ("Side clearance", "clearance"),
            ("Speed", "speed"),
            ("Target speed", "target_speed"),
            ("Requested accel", "requested_accel"),
            ("Actual accel", "actual_accel"),
            ("Steering", "steering"),
            ("Lookahead", "lookahead"),
            ("Target point", "target_index"),
            ("View", "view"),
        ]
        for label_text, key in metric_names:
            row = tk.Frame(panel, bg="#151a1f")
            row.pack(fill="x", pady=7)
            tk.Label(row, text=label_text, bg="#151a1f", fg="#aeb9c1").pack(side="left")
            value = tk.Label(row, text="-", bg="#151a1f", fg="#f3f5f4", font=("Arial", 12, "bold"))
            value.pack(side="right")
            self.metrics[key] = value

        self.metrics["track_width"].config(text=f"{self.track_width_m:.2f} m")
        self.metrics["car_width"].config(text=f"{self.car_width_m:.2f} m")
        self.metrics["clearance"].config(
            text=f"{((self.track_width_m - self.car_width_m) / 2):.2f} m"
        )

        legend = tk.Label(
            panel,
            text=(
                "Orange cones: left boundary\n"
                "Blue cones: right boundary\n"
                "Red line: planned raceline\n"
                "Pink line: driven trajectory\n"
                "Green dot: pure pursuit target\n"
                "White body: 1.1 m wide car"
            ),
            justify="left",
            bg="#151a1f",
            fg="#c7d0d6",
            pady=12,
        )
        legend.pack(anchor="w")

    def reset(self) -> None:
        if self.raceline is None:
            first = self.path[0]
            self.state = VehicleState(first.x, first.y, first.yaw, 0.5)
        else:
            start_index, start_yaw = self._prepare_raceline_start()
            self.state = VehicleState(
                float(self.raceline.x_m[start_index]),
                float(self.raceline.y_m[start_index]),
                start_yaw,
                0.5,
            )
            self.last_target_index = start_index
        if self.raceline is None:
            self.last_target_index = 0
        self.integrator = 0.0
        self.previous_accel_cmd_mps2 = 0.0
        self.sim_time_s = 0.0
        self.trajectory = [TrackPoint(self.state.x, self.state.y, self.state.yaw, self.state.speed)]
        self.target_marker = self.path[0]
        self.follow_vehicle()

    def _prepare_raceline_start(self) -> tuple[int, float]:
        if self.raceline is None:
            return 0, self.path[0].yaw

        start_index = self.raceline.closest_index_to(0.0, 0.0)
        start_dist = math.hypot(float(self.raceline.x_m[start_index]), float(self.raceline.y_m[start_index]))
        if start_dist > 2.0:
            raise ValueError(
                f"No raceline point near origin (closest is {start_dist:.2f} m at index {start_index})."
            )

        yaw = float(self.raceline.psi_rad[start_index])
        yaw_deg = self._normalize_degrees(math.degrees(yaw))
        if abs(yaw_deg) > 90.0:
            geometric_yaw = self.raceline.geometric_heading_at(start_index)
            psi_error = wrap_angle(yaw - geometric_yaw)
            if abs(abs(math.degrees(psi_error)) - 90.0) < 15.0:
                self.raceline.rotate_psi(-psi_error)
            else:
                self.raceline.reverse_direction()

            start_index = self.raceline.closest_index_to(0.0, 0.0)
            yaw = float(self.raceline.psi_rad[start_index])
            yaw_deg = self._normalize_degrees(math.degrees(yaw))
            if abs(yaw_deg) > 90.0:
                raise ValueError(
                    f"Raceline at start points {yaw_deg:.1f} degrees from +x axis. "
                    "Expected near 0 degrees."
                )

        print(f"Start index in raceline: {start_index}")
        print(f"Start position: ({self.raceline.x_m[start_index]:.3f}, {self.raceline.y_m[start_index]:.3f})")
        print(f"Start yaw: {yaw:.3f} rad ({yaw_deg:.1f} degrees)")
        return start_index, yaw

    @staticmethod
    def _normalize_degrees(angle_deg: float) -> float:
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg <= -180.0:
            angle_deg += 360.0
        return angle_deg

    def toggle(self) -> None:
        self.running = not self.running
        self.play_button.config(text="Pause" if self.running else "Play")

    def follow_vehicle(self) -> None:
        self.follow_car.set(True)
        self.view_center_x = self.state.x
        self.view_center_y = self.state.y

    def fit_track(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        min_x, max_x, min_y, max_y = self._track_bounds()
        track_width = max(max_x - min_x, 1.0)
        track_height = max(max_y - min_y, 1.0)
        base_scale = self._base_scale(width, height)
        fit_zoom = min(width * 0.88 / track_width, height * 0.88 / track_height) / base_scale
        self.zoom.set(self._clamp(fit_zoom, 0.2, 4.0))
        self.view_center_x = 0.5 * (min_x + max_x)
        self.view_center_y = 0.5 * (min_y + max_y)
        self.follow_car.set(False)

    def run(self) -> None:
        self.root.mainloop()

    def _tick(self) -> None:
        if self.running:
            steps = max(1, round(self.speed_scale.get()))
            for _ in range(steps):
                self._update_vehicle(self.dt_s * self.speed_scale.get())
        self._draw()
        self.root.after(16, self._tick)

    def _update_vehicle(self, dt_s: float) -> None:
        high = self._raceline_pure_pursuit() if self.raceline is not None else self._pure_pursuit()
        if self.raceline is not None:
            low = self._speed_feedforward_pi(high["target_speed"], high["target_accel"], dt_s)
            requested_accel = low["requested_accel"]
        else:
            low = self._speed_pi(high["target_speed"], dt_s)
            requested_accel = requested_acceleration_from_actuators(low["throttle"], low["brake"])
        powertrain_accel = self.powertrain.actual_acceleration(self.state.speed, requested_accel)
        speed_before_lateral = max(0.0, self.state.speed + powertrain_accel * dt_s)
        commanded_lateral_accel = (
            speed_before_lateral * speed_before_lateral * math.tan(high["steering"]) / self.wheelbase_m
        )
        actual_long_force, actual_lateral_force, _ = apply_friction_circle(
            self.vehicle_config.mass_kg * powertrain_accel,
            self.vehicle_config.mass_kg * commanded_lateral_accel,
            self.vehicle_config.mass_kg,
        )
        actual_accel = actual_long_force / self.vehicle_config.mass_kg
        achievable_lateral_accel = actual_lateral_force / self.vehicle_config.mass_kg
        self.state.speed = max(0.0, self.state.speed + actual_accel * dt_s)
        if self.raceline is None:
            self.state.speed = min(MAX_SPEED_30_KMH_MPS, self.state.speed)
        yaw_rate = achievable_lateral_accel / max(self.state.speed, 1e-3)
        self.state.yaw += yaw_rate * dt_s
        self.state.x += self.state.speed * math.cos(self.state.yaw) * dt_s
        self.state.y += self.state.speed * math.sin(self.state.yaw) * dt_s
        self.sim_time_s += dt_s
        self._append_trajectory_point()

        self.metrics["speed"].config(text=f"{self.state.speed:.2f} m/s")
        self.metrics["target_speed"].config(text=f"{high['target_speed']:.2f} m/s")
        self.metrics["requested_accel"].config(text=f"{requested_accel:.2f} m/s^2")
        self.metrics["actual_accel"].config(text=f"{actual_accel:.2f} m/s^2")
        self.metrics["steering"].config(text=f"{high['steering']:.2f} rad")
        self.metrics["lookahead"].config(text=f"{high['lookahead']:.2f} m")
        self.metrics["target_index"].config(text=str(high["target_index"]))
        self.metrics["view"].config(text="follow" if self.follow_car.get() else "manual")

    def _pure_pursuit(self) -> dict[str, float]:
        nearest = self._nearest_index()
        lookahead = self._clamp(
            self.min_lookahead_m + self.lookahead_gain.get() * self.state.speed,
            self.min_lookahead_m,
            self.max_lookahead_m,
        )

        target_index = nearest
        for offset in range(len(self.path)):
            index = (nearest + offset) % len(self.path)
            point = self.path[index]
            if math.hypot(point.x - self.state.x, point.y - self.state.y) >= lookahead:
                target_index = index
                break

        self.last_target_index = target_index
        target = self.path[target_index]
        self.target_marker = target
        dx = target.x - self.state.x
        dy = target.y - self.state.y
        local_x = math.cos(self.state.yaw) * dx + math.sin(self.state.yaw) * dy
        local_y = -math.sin(self.state.yaw) * dx + math.cos(self.state.yaw) * dy
        distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
        curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
        steering = self._clamp(
            math.atan(self.wheelbase_m * curvature),
            -self.max_steer_rad,
            self.max_steer_rad,
        )

        return {
            "steering": steering,
            "target_speed": target.speed,
            "target_accel": 0.0,
            "target_index": target_index,
            "lookahead": lookahead,
        }

    def _raceline_pure_pursuit(self) -> dict[str, float]:
        if self.raceline is None:
            return self._pure_pursuit()

        nearest = self.raceline.nearest_index(self.state.x, self.state.y)
        s_now = float(self.raceline.s_m[nearest])
        lookahead, _, _, _ = self._compute_raceline_lookahead(self.state.speed, s_now)
        target = self.raceline.target_at_s(s_now + lookahead)
        brake_distance = max(3.0, self.state.speed * self.state.speed / (2.0 * 8.0))
        speed_target = self.raceline.target_at_s(s_now + brake_distance)
        self.last_target_index = target.index
        self.target_marker = TrackPoint(target.x_m, target.y_m)

        dx = target.x_m - self.state.x
        dy = target.y_m - self.state.y
        local_x = math.cos(self.state.yaw) * dx + math.sin(self.state.yaw) * dy
        local_y = -math.sin(self.state.yaw) * dx + math.cos(self.state.yaw) * dy
        distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
        curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
        steering = self._clamp(
            math.atan(self.wheelbase_m * curvature),
            -self.max_steer_rad,
            self.max_steer_rad,
        )

        return {
            "steering": steering,
            "target_speed": speed_target.v_target_mps,
            "target_accel": speed_target.a_target_mps2,
            "target_index": target.index,
            "lookahead": lookahead,
        }

    def _compute_raceline_lookahead(
        self,
        v_actual: float,
        s_now: float,
        horizon: float = 10.0,
    ) -> tuple[float, float, float, float]:
        if self.raceline is None:
            lookahead = self._clamp(v_actual * 0.45, 2.0, 6.0)
            return lookahead, lookahead, lookahead, 0.0

        sample_count = 20
        s_samples = [
            (s_now + horizon * i / (sample_count - 1)) % self.raceline.total_s
            for i in range(sample_count)
        ]
        kappa_max = max(abs(self.raceline.kappa_at(s)) for s in s_samples)
        lookahead_speed = v_actual * 0.45
        lookahead_curv = 1.0 / max(kappa_max, 1e-3)
        lookahead = self._clamp(min(lookahead_speed, lookahead_curv), 2.0, 6.0)
        return lookahead, lookahead_speed, lookahead_curv, kappa_max

    def _speed_pi(self, target_speed: float, dt_s: float) -> dict[str, float]:
        error = target_speed - self.state.speed
        self.integrator = self._clamp(self.integrator + error * dt_s, -3.0, 3.0)
        command = self._clamp(0.35 * error + 0.12 * self.integrator, -1.0, 1.0)
        return {
            "throttle": self._clamp(command, 0.0, 1.0),
            "brake": self._clamp(-1.3 * command, 0.0, 1.0),
        }

    def _speed_feedforward_pi(self, target_speed: float, target_accel: float, dt_s: float) -> dict[str, float]:
        error = target_speed - self.state.speed
        self.integrator = self._clamp(
            self.integrator + self.raceline_speed_ki * error * dt_s,
            -self.raceline_integral_limit_mps2,
            self.raceline_integral_limit_mps2,
        )
        raw_accel = target_accel + self.raceline_speed_kp * error + self.integrator
        lower = self.previous_accel_cmd_mps2 - self.max_jerk_brake_mps3 * dt_s
        upper = self.previous_accel_cmd_mps2 + self.max_jerk_throttle_mps3 * dt_s
        accel_cmd = self._clamp(raw_accel, lower, upper)
        self.previous_accel_cmd_mps2 = accel_cmd
        return {
            "throttle": self._clamp(accel_cmd / self.safe_throttle_accel_mps2, 0.0, 1.0),
            "brake": self._clamp(-accel_cmd / self.safe_braking_accel_mps2, 0.0, 1.0),
            "requested_accel": accel_cmd,
        }

    def _nearest_index(self) -> int:
        best_index = self.last_target_index
        best_distance = float("inf")
        for offset in range(len(self.path)):
            index = (self.last_target_index + offset) % len(self.path)
            point = self.path[index]
            distance = (self.state.x - point.x) ** 2 + (self.state.y - point.y) ** 2
            if distance < best_distance:
                best_distance = distance
                best_index = index
            if offset > 180 and distance > best_distance * 6:
                break
        return best_index

    def _draw(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")

        scale = self._base_scale(width, height) * self.zoom.get()
        if self.follow_car.get():
            self.view_center_x = self.state.x
            self.view_center_y = self.state.y
        camera_x = width / 2.0 - self.view_center_x * scale
        camera_y = height / 2.0 + self.view_center_y * scale

        def screen(point: TrackPoint | VehicleState) -> tuple[float, float]:
            return camera_x + point.x * scale, camera_y - point.y * scale

        self._draw_grid(width, height, camera_x, camera_y, scale)
        self._draw_polyline(self.path, screen, "#6d7a71", 2, True)
        if self.raceline_points:
            self._draw_polyline(self.raceline_points, screen, "#f04444", 2, True)
        self._draw_polyline(self.left_cones, screen, "#6b4b28", 2, True)
        self._draw_polyline(self.right_cones, screen, "#254e70", 2, True)
        self._draw_polyline(self.trajectory, screen, "#e35d5b", 3, False)
        self._draw_cones(self.left_cones, screen, "#f4a23a", scale)
        self._draw_cones(self.right_cones, screen, "#3aa0f4", scale)
        self._draw_target(screen, scale)
        self._draw_car(screen, scale)

    def _draw_grid(
        self,
        width: int,
        height: int,
        camera_x: float,
        camera_y: float,
        scale: float,
    ) -> None:
        step = scale * 5.0
        x = camera_x % step
        while x < width:
            self.canvas.create_line(x, 0, x, height, fill="#2a332d")
            x += step
        y = camera_y % step
        while y < height:
            self.canvas.create_line(0, y, width, y, fill="#2a332d")
            y += step

    def _begin_pan(self, event: tk.Event) -> None:
        self._drag_start = (float(event.x), float(event.y), self.view_center_x, self.view_center_y)
        self.follow_car.set(False)

    def _pan(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        start_x, start_y, center_x, center_y = self._drag_start
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = self._base_scale(width, height) * self.zoom.get()
        self.view_center_x = center_x - (float(event.x) - start_x) / scale
        self.view_center_y = center_y + (float(event.y) - start_y) / scale

    def _zoom_with_wheel(self, event: tk.Event) -> None:
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            factor = 0.9
        else:
            factor = 1.1
        self.zoom.set(self._clamp(self.zoom.get() * factor, 0.2, 4.0))

    def _draw_polyline(
        self,
        points: list[TrackPoint],
        screen,
        color: str,
        width: int,
        closed: bool,
    ) -> None:
        if len(points) < 2:
            return
        coords: list[float] = []
        for point in points:
            x, y = screen(point)
            coords.extend([x, y])
        if closed and points:
            x, y = screen(points[0])
            coords.extend([x, y])
        self.canvas.create_line(*coords, fill=color, width=width, smooth=True)

    def _draw_cones(self, cones: list[TrackPoint], screen, color: str, scale: float) -> None:
        radius = max(3.5, 0.16 * scale)
        for cone in cones:
            x, y = screen(cone)
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="#101316",
                width=2,
            )

    def _draw_target(self, screen, scale: float) -> None:
        target = self.target_marker
        car_x, car_y = screen(self.state)
        target_x, target_y = screen(target)
        radius = max(6.0, 0.25 * scale)
        self.canvas.create_line(
            car_x,
            car_y,
            target_x,
            target_y,
            fill="#7cc7a0",
            width=2,
            dash=(7, 7),
        )
        self.canvas.create_oval(
            target_x - radius,
            target_y - radius,
            target_x + radius,
            target_y + radius,
            fill="#7cc7a0",
            outline="",
        )

    def _draw_car(self, screen, scale: float) -> None:
        center_x, center_y = screen(self.state)
        half_length = self.car_length_m * scale / 2.0
        half_width = self.car_width_m * scale / 2.0

        corners = [
            (half_length, 0.0),
            (half_length * 0.62, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length * 0.62, half_width),
        ]
        body = self._rotate_translate(corners, center_x, center_y, -self.state.yaw)
        self.canvas.create_polygon(body, fill="#e9eef2", outline="#101316", width=2)

        nose = self._rotate_translate(
            [
                (half_length, 0.0),
                (half_length * 0.62, -half_width * 0.6),
                (half_length * 0.62, half_width * 0.6),
            ],
            center_x,
            center_y,
            -self.state.yaw,
        )
        self.canvas.create_polygon(nose, fill="#e35d5b", outline="")

    @staticmethod
    def _rotate_translate(
        points: list[tuple[float, float]],
        cx: float,
        cy: float,
        angle: float,
    ) -> list[float]:
        coords: list[float] = []
        c = math.cos(angle)
        s = math.sin(angle)
        for x, y in points:
            coords.extend([cx + c * x - s * y, cy + s * x + c * y])
        return coords

    def _build_track(self) -> list[TrackPoint]:
        return self.path

    def _build_cones(self) -> tuple[list[TrackPoint], list[TrackPoint]]:
        return self.left_cones, self.right_cones

    def _append_trajectory_point(self) -> None:
        if not self.trajectory:
            self.trajectory.append(TrackPoint(self.state.x, self.state.y, self.state.yaw, self.state.speed))
            return
        previous = self.trajectory[-1]
        if math.hypot(self.state.x - previous.x, self.state.y - previous.y) >= 0.2:
            self.trajectory.append(TrackPoint(self.state.x, self.state.y, self.state.yaw, self.state.speed))
        if len(self.trajectory) > 6000:
            self.trajectory = self.trajectory[-6000:]

    def _track_bounds(self) -> tuple[float, float, float, float]:
        points = [*self.path, *self.raceline_points, *self.left_cones, *self.right_cones]
        return (
            min(point.x for point in points),
            max(point.x for point in points),
            min(point.y for point in points),
            max(point.y for point in points),
        )

    @staticmethod
    def _base_scale(width: int, height: int) -> float:
        return min(width / 132.0, height / 86.0)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cone track pure pursuit animation.")
    parser.add_argument("--track", default="autocross", help="Track name: autocross or hockenheim_fsg")
    parser.add_argument("--raceline", type=Path, help="Optional optimizer trajectory CSV to animate")
    args = parser.parse_args()
    ConeTrackAnimation(get_track(args.track), args.raceline).run()


if __name__ == "__main__":
    main()
