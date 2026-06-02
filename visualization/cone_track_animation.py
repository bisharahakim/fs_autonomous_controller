from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass

from autocross_track import TrackPoint, build_cone_boundaries, build_fsd_autocross_track


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


class ConeTrackAnimation:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Formula Student Pure Pursuit Animation")
        self.root.geometry("1260x800")
        self.root.minsize(980, 650)

        self.wheelbase_m = 1.55
        self.car_length_m = 2.8
        self.car_width_m = 1.5
        self.track_width_m = 3.0
        self.cone_spacing_m = 2.5
        self.min_lookahead_m = 3.0
        self.max_lookahead_m = 18.0
        self.max_steer_rad = 0.5
        self.dt_s = 0.02
        self.max_graph_samples = 420

        self.path = self._build_track()
        self.left_cones, self.right_cones = self._build_cones()
        self.running = True
        self.speed_scale = tk.DoubleVar(value=1.0)
        self.lookahead_gain = tk.DoubleVar(value=0.2)

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
            text="FSD Autocross Pure Pursuit",
            bg="#151a1f",
            fg="#f3f5f4",
            font=("Arial", 18, "bold"),
        )
        title.pack(side="left", padx=(0, 18))

        self.play_button = tk.Button(toolbar, text="Pause", width=8, command=self.toggle)
        self.play_button.pack(side="left", padx=4)

        reset_button = tk.Button(toolbar, text="Reset", width=8, command=self.reset)
        reset_button.pack(side="left", padx=4)

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

        content = tk.Frame(self.root, bg="#101316")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(content, bg="#1f261f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

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
            ("Steering", "steering"),
            ("Lookahead", "lookahead"),
            ("Target point", "target_index"),
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
            text="Orange cones: left boundary\nBlue cones: right boundary\nGreen dot: pure pursuit target\nWhite body: 1.5 m wide car",
            justify="left",
            bg="#151a1f",
            fg="#c7d0d6",
            pady=12,
        )
        legend.pack(anchor="w")

        graph_title = tk.Label(
            panel,
            text="Live graph",
            bg="#151a1f",
            fg="#f3f5f4",
            font=("Arial", 13, "bold"),
        )
        graph_title.pack(anchor="w", pady=(8, 6))

        self.graph = tk.Canvas(panel, height=235, bg="#101316", highlightthickness=1, highlightbackground="#293138")
        self.graph.pack(fill="x")

        graph_legend = tk.Label(
            panel,
            text="Green: speed\nGray: target speed\nRed: steering",
            justify="left",
            bg="#151a1f",
            fg="#c7d0d6",
            pady=8,
        )
        graph_legend.pack(anchor="w")

    def reset(self) -> None:
        first = self.path[0]
        self.state = VehicleState(first.x - 1.2, first.y - 0.15, first.yaw, 0.0)
        self.last_target_index = 0
        self.integrator = 0.0
        self.sim_time_s = 0.0
        self.history: list[dict[str, float]] = []

    def toggle(self) -> None:
        self.running = not self.running
        self.play_button.config(text="Pause" if self.running else "Play")

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
        high = self._pure_pursuit()
        low = self._speed_pi(high["target_speed"], dt_s)

        accel = 5.0 * low["throttle"] - 6.0 * low["brake"] - 0.08 * self.state.speed
        self.state.speed = max(0.0, self.state.speed + accel * dt_s)
        yaw_rate = self.state.speed / self.wheelbase_m * math.tan(high["steering"])
        self.state.yaw += yaw_rate * dt_s
        self.state.x += self.state.speed * math.cos(self.state.yaw) * dt_s
        self.state.y += self.state.speed * math.sin(self.state.yaw) * dt_s
        self.sim_time_s += dt_s

        self.history.append(
            {
                "time": self.sim_time_s,
                "speed": self.state.speed,
                "target_speed": high["target_speed"],
                "steering": high["steering"],
            }
        )
        if len(self.history) > self.max_graph_samples:
            self.history = self.history[-self.max_graph_samples :]

        self.metrics["speed"].config(text=f"{self.state.speed:.2f} m/s")
        self.metrics["target_speed"].config(text=f"{high['target_speed']:.2f} m/s")
        self.metrics["steering"].config(text=f"{high['steering']:.2f} rad")
        self.metrics["lookahead"].config(text=f"{high['lookahead']:.2f} m")
        self.metrics["target_index"].config(text=str(high["target_index"]))

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
            "target_index": target_index,
            "lookahead": lookahead,
        }

    def _speed_pi(self, target_speed: float, dt_s: float) -> dict[str, float]:
        error = target_speed - self.state.speed
        self.integrator = self._clamp(self.integrator + error * dt_s, -3.0, 3.0)
        command = self._clamp(0.35 * error + 0.12 * self.integrator, -1.0, 1.0)
        return {
            "throttle": self._clamp(command, 0.0, 1.0),
            "brake": self._clamp(-1.3 * command, 0.0, 1.0),
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

        scale = min(width / 132.0, height / 86.0)
        camera_x = width / 2.0 - self.state.x * scale
        camera_y = height / 2.0 + self.state.y * scale

        def screen(point: TrackPoint | VehicleState) -> tuple[float, float]:
            return camera_x + point.x * scale, camera_y - point.y * scale

        self._draw_grid(width, height, camera_x, camera_y, scale)
        self._draw_polyline(self.path, screen, "#6d7a71", 2, True)
        self._draw_polyline(self.left_cones, screen, "#6b4b28", 2, True)
        self._draw_polyline(self.right_cones, screen, "#254e70", 2, True)
        self._draw_cones(self.left_cones, screen, "#f4a23a", scale)
        self._draw_cones(self.right_cones, screen, "#3aa0f4", scale)
        self._draw_target(screen, scale)
        self._draw_car(screen, scale)
        self._draw_live_graph()

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

    def _draw_polyline(
        self,
        points: list[TrackPoint],
        screen,
        color: str,
        width: int,
        closed: bool,
    ) -> None:
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
        high = self._pure_pursuit()
        target = self.path[int(high["target_index"])]
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

    def _draw_live_graph(self) -> None:
        width = max(1, self.graph.winfo_width())
        height = max(1, self.graph.winfo_height())
        self.graph.delete("all")

        margin_left = 34
        margin_right = 10
        margin_top = 14
        margin_bottom = 28
        plot_x = margin_left
        plot_y = margin_top
        plot_w = max(1, width - margin_left - margin_right)
        plot_h = max(1, height - margin_top - margin_bottom)

        self.graph.create_rectangle(
            plot_x,
            plot_y,
            plot_x + plot_w,
            plot_y + plot_h,
            outline="#293138",
            fill="#11171b",
        )

        for i in range(1, 4):
            y = plot_y + plot_h * i / 4
            self.graph.create_line(plot_x, y, plot_x + plot_w, y, fill="#253039")

        self.graph.create_text(8, plot_y + 2, text="25 m/s", anchor="nw", fill="#aeb9c1", font=("Arial", 9))
        self.graph.create_text(8, plot_y + plot_h - 12, text="0", anchor="nw", fill="#aeb9c1", font=("Arial", 9))
        self.graph.create_text(
            plot_x,
            height - 18,
            text="rolling controller history",
            anchor="w",
            fill="#aeb9c1",
            font=("Arial", 9),
        )

        if len(self.history) < 2:
            return

        self._draw_graph_series("speed", "#7cc7a0", 0.0, 25.0, plot_x, plot_y, plot_w, plot_h)
        self._draw_graph_series(
            "target_speed",
            "#cfd6dc",
            0.0,
            25.0,
            plot_x,
            plot_y,
            plot_w,
            plot_h,
            dash=(5, 4),
        )
        self._draw_graph_series("steering", "#e35d5b", -0.5, 0.5, plot_x, plot_y, plot_w, plot_h)

    def _draw_graph_series(
        self,
        key: str,
        color: str,
        value_min: float,
        value_max: float,
        plot_x: float,
        plot_y: float,
        plot_w: float,
        plot_h: float,
        dash: tuple[int, int] | None = None,
    ) -> None:
        coords: list[float] = []
        count = len(self.history)
        for i, sample in enumerate(self.history):
            x = plot_x + plot_w * i / max(1, count - 1)
            normalized = (sample[key] - value_min) / (value_max - value_min)
            normalized = self._clamp(normalized, 0.0, 1.0)
            y = plot_y + plot_h * (1.0 - normalized)
            coords.extend([x, y])

        kwargs = {"fill": color, "width": 2}
        if dash is not None:
            kwargs["dash"] = dash
        self.graph.create_line(*coords, **kwargs)

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
        return build_fsd_autocross_track()

    def _build_cones(self) -> tuple[list[TrackPoint], list[TrackPoint]]:
        return build_cone_boundaries(
            self.path,
            track_width_m=self.track_width_m,
            cone_spacing_m=self.cone_spacing_m,
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)


if __name__ == "__main__":
    ConeTrackAnimation().run()
