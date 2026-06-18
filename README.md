# fs_autonomous_controller

`fs_autonomous_controller` is a small Python starting point for a Formula
Student Driverless controller:

- High-level lateral control: pure pursuit.
- Low-level longitudinal control: PI speed control.
- Synthetic FSD autocross track made from cones.
- Python animation for watching the car move.
- Static SVG graphs for inspecting controller behavior.

The project name is `fs_autonomous_controller`. The reusable Python controller
package inside the project is still named `fs_controller`.

## Quick Start

From the project folder:

```bash
cd "/Users/bisharahakim/Documents/fs_autonomous_controller"
```

Run the moving Python animation:

```bash
python3 visualization/cone_track_animation.py
```

Generate the main static graph:

```bash
python3 visualization/static_controller_graph.py
```

Open the generated graph:

```text
outputs/controller_static_graph.svg
```

Generate the controller comparison graph:

```bash
python3 examples/compare_controller_settings.py
```

Open:

```text
outputs/controller_settings_comparison.svg
```

Compare regular pure pursuit against adaptive pure pursuit:

```bash
python3 examples/compare_pure_pursuit.py
```

Open:

```text
outputs/pure_pursuit_comparison.svg
```

## What Each File Does

```text
fs_controller/pure_pursuit.py
```

Pure pursuit steering controller. It receives the vehicle state and a path, then
outputs a steering angle.

```text
fs_controller/low_level.py
```

PI speed controller. It compares target speed against actual speed, then outputs
normalized throttle and brake.

```text
fs_controller/controller_stack.py
```

Combines pure pursuit and PI into one controller call.

```text
visualization/autocross_track.py
```

Creates the synthetic FSD autocross track and assigns target speeds using a
simple curvature-based speed planner. Straights can request up to about
90 km/h, while tight corners slow toward about 27-40 km/h.

```text
visualization/cone_track_animation.py
```

Best file to run when you want to see something moving. It opens a Tkinter
window with the cone track, car, pure pursuit target, live values, and a small
rolling graph.

```text
visualization/static_controller_graph.py
```

Runs one full simulation and saves a static SVG graph showing trajectory, speed,
tracking error, steering, throttle/brake, and lookahead.

```text
examples/simulate_controller.py
```

Terminal-only smoke test. It proves the controller stack imports and runs, but
it only prints numbers. You are not supposed to see a visual graph from this one.

```text
examples/compare_controller_settings.py
```

Visual example. It runs the same autocross track with multiple controller
settings and saves a comparison SVG. This is the useful example for seeing how
different lookahead and PI gains affect the path.

```text
examples/compare_pure_pursuit.py
```

Analytical comparison of regular fixed-lookahead pure pursuit and adaptive
speed-based pure pursuit. It saves a readable SVG with trajectory, tracking
error, lookahead, steering, speed tracking, and summary metrics.

## What You Should See

When you run:

```bash
python3 visualization/cone_track_animation.py
```

You should see a window with:

- orange and blue cones
- a white car
- a green pure pursuit target point
- live speed, target speed, steering, and lookahead values
- a small rolling graph

When you run:

```bash
python3 visualization/static_controller_graph.py
```

You should get:

```text
outputs/controller_static_graph.svg
```

This graph shows one controller setup in detail.

When you run:

```bash
python3 examples/compare_controller_settings.py
```

You should get:

```text
outputs/controller_settings_comparison.svg
```

This graph compares:

- baseline tuning
- shorter lookahead
- longer lookahead
- more aggressive PI gains

Use this one to understand why tuning matters.

## Controller Loop

At each control tick:

1. Read vehicle state: `x`, `y`, `yaw`, and `speed`.
2. Provide a path made of `PathPoint(x, y, speed)` values.
3. Call `ControllerStack.update(state, path, dt_s)`.
4. Send `steering_rad`, `throttle`, and `brake` to your vehicle interface.

Example:

```python
from fs_controller import ControllerStack, ControllerStackConfig
from fs_controller import PIConfig, PurePursuitConfig, VehicleState, PathPoint

controller = ControllerStack(
    ControllerStackConfig(
        pure_pursuit=PurePursuitConfig(wheelbase_m=1.55),
        speed_pi=PIConfig(
            kp=0.35,
            ki=0.12,
            output_min=-1.0,
            output_max=1.0,
            integrator_min=-3.0,
            integrator_max=3.0,
        ),
    )
)

path = [
    PathPoint(0.0, 0.0, 4.0),
    PathPoint(5.0, 0.5, 5.0),
    PathPoint(10.0, 2.0, 5.0),
]

state = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.0)
command = controller.update(state, path, dt_s=0.02)
```

## Tuning Notes

Pure pursuit:

- Increase lookahead for smoother steering.
- Decrease lookahead if the car cuts corners or reacts too late.
- Too much lookahead can make the car miss tight turns.
- Too little lookahead can make steering twitchy.

PI speed controller:

- Higher `kp` reacts faster to speed error.
- Higher `ki` removes long-term speed error.
- Too much `kp` or `ki` can cause brake/throttle oscillation.
- Normalized throttle/brake values in this simulation are not real actuator
  commands yet.

## Real Car Safety

Before driving a real vehicle, add independent checks for emergency stop,
command timeout, steering/speed limits, sensor validity, actuator feedback, and
state-estimation confidence. This code is a controller starting point, not a
complete safety system.
