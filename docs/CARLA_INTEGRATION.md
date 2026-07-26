# CARLA Integration — Living Context

**Purpose:** Context handoff for the CARLA integration side of the Formula Student Driverless AV final project. Update this file as milestones complete.

**Owner:** Mate (`fs_autonomous_controller` repo)
**Last updated:** _[fill in date when editing]_

---

## Environment overview

**Development machine:** Lab PC, Ubuntu 24 (native Linux, not a VM). All CARLA work is done directly on this machine. My Mac is used only for the pure-Python side of the project — the RAPP module lives there, but gets copied/pulled to the lab PC for CARLA integration.

**Simulator:** CARLA (version TBD, running natively on the lab PC — team got it working on Ubuntu 24 despite it being officially unsupported for CARLA).

**Middleware:** None. Direct CARLA Python API only. **No ROS 2 in this project.** ROS 2 integration is explicitly deferred to a future project (whole-car pipeline including perception, planning, control across nodes). For this final project, everything runs in a single Python process talking directly to the CARLA server.

**Perception:** Out of scope. No SLAM, no cone detection, no YOLOv8. Track geometry and vehicle state are both ground-truth from the simulator. This is a controller demonstration, not a full AV stack.

---

## Project scope for CARLA

Port the pure-Python RAPP controller (tag `v0.4-rapp`, 101 s clean lap on Hockenheim FSG) into CARLA to demonstrate the same controller drives the same racing line in a 3D simulation environment.

**Deliverable:** A CARLA vehicle driving one clean lap of the Hockenheim FSG cone layout in an empty CARLA world using RAPP through the plain CARLA Python API. Measured lap time, tracking error, incidents logged. Video recordings from two camera angles (chase and bird's-eye) for the report and defense.

**Explicit non-goals:**
- No perception integration
- No ROS 2
- No custom vehicle mesh
- No custom high-fidelity map — cones on a flat empty plane is fine
- No matching CARLA lap time to pure-Python lap time exactly (a residual gap is expected and documented)

---

## Current state of the pure-Python code (as of v0.4-rapp)

Pipeline in `~/Documents/fs_autonomous_controller/`:

```
Track (visualization/hockenheim_fsg_track.py)
    ↓
Vehicle physics (fs_controller/powertrain.py, fs_controller/tire_grip.py)
                                                    ← 250 kg driverless FS car,
                                                       calibrated from team's
                                                       2024 MATLAB source
    ↓
Raceline CSV (inputs/racelines/hockenheim_fsg_benjamin24_safe.csv)
    ↓ columns: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
Raceline loader (fs_controller/raceline.py)     ← kappa_at(s), target_at_s()
    ↓
RAPP controller (fs_controller/rapp.py)         ← 121 lines
    input:  (x, y, yaw, v)
    output: RAPPCommand(steering, target_speed, target_accel, + diagnostics)
    config: RAPPConfig(speed_gain, lookahead_min/max, horizon_ahead/behind,
                      brake_decel, brake_dist_min)
    ↓
Simulator loop (examples/run_track_baseline.py) ← this is what CARLA replaces
```

**Result on Hockenheim:** 101.00 s lap, 0 cone hits, 0 off-track (optimizer prediction: 88.44 s, so 14% controller tracking gap).

**Key architectural point:** `rapp.py` is a pure Python module with no dependencies on the simulator. It gets reused as-is inside the CARLA script — the CARLA client handles state extraction and command application, `rapp.py` handles the control logic.

---

## Architecture

### Track approach — empty world with cones on a flat plane

An empty CARLA world (loaded either from a minimal generated OpenDRIVE description or from the closest-to-empty existing map, TBD in M2). Static traffic cones spawned at the exact positions from `hockenheim_fsg_track.py`. No town, no buildings, no road markings, no other traffic. The cones **are** the track.

Rationale: cleanest visuals for the report, no ambiguity about what the car is or isn't hitting, deterministic setup, no dependency on which maps happen to be installed on the lab PC.

### Vehicle approach — CARLA default vehicle, tuned toward FS car model (Option Y, staged)

**Stage 1 (M3):** Use a stock CARLA vehicle blueprint (candidates: `vehicle.audi.tt`, `vehicle.mini.cooper_s` — pick after inspecting what's available). Do not modify physics. Point of this stage: prove the control loop closes (state read → RAPP → command apply → vehicle moves). Expect lap time and behavior to be off — that's fine.

**Stage 2 (M5):** Tune CARLA's `PhysicsControl` to approximate the Python FS car specs:
- `mass = 250.0` (exact match)
- `torque_curve` matched to Benjamin24 motor curve (135 Nm to peak rpm, then power-limited)
- `drag_coefficient` derived from Cd, area, air density in `powertrain.py`
- `tire_friction` set to effective mu at static wheel load from `tire_grip.py` poly2
- `downforce` if CARLA version supports it, matched to Cl, area
- Wheelbase, wheel positions, CoG height set from Benjamin24 geometry

**Critical direction:** we tune CARLA to match the Python model. We do NOT retune the Python model to match CARLA. The Python model represents the team's real car and stays the source of truth throughout the pipeline.

### Controller wiring

Single Python process. One file: `examples/run_carla_baseline.py`. Structure:

```python
# Setup phase (once)
client = carla.Client('localhost', 2000)
world = load_empty_world(client)          # M2
cones = spawn_hockenheim_cones(world)     # M2
vehicle = spawn_ego_vehicle(world)        # M3
apply_fs_physics(vehicle)                 # M5, no-op in M3
raceline = load_raceline('inputs/racelines/hockenheim_fsg_benjamin24_safe.csv')
rapp = RAPP(raceline, RAPPConfig(...))
chase_cam, birdseye_cam = setup_cameras(vehicle, world)  # M6

# Control loop
world.set_synchronous_mode(True)  # deterministic timing
while running:
    state = extract_state(vehicle)
    cmd = rapp.compute_control(state.x, state.y, state.yaw, state.v)
    control = translate_command_to_carla(cmd, vehicle)
    vehicle.apply_control(control)
    log(t, state, cmd, control)
    world.tick()
```

### Command translation

- **Steering:** RAPP outputs radians. CARLA's `VehicleControl.steer` is normalized `[-1, 1]`. Divide by max steering angle from `PhysicsControl.wheels[0].max_steer_angle` (which is in degrees — convert).
- **Longitudinal:** RAPP outputs target acceleration in m/s². CARLA takes `throttle [0, 1]` and `brake [0, 1]` separately. Positive target → throttle, negative → brake. Simple linear map initially; tune coefficient in M5 based on measured response.

### Recording

Two cameras attached at startup:
1. **Chase camera** — attached to the vehicle, ~5 m behind, ~2 m above, looking forward. Dramatic for demo video.
2. **Bird's-eye camera** — stationary in world coordinates at `(x=50, y=100, z=200)` looking down, wide FOV. Shows full track and racing line.

Only one is active at a time (recording to disk). Keyboard toggle (`c` for chase, `b` for bird's-eye) switches during the run. Or run twice, once with each. Both approaches acceptable.

---

## Lab environment (fill in when confirmed)

- **CARLA version:** _[e.g. 0.9.15]_
- **Ubuntu version:** 24.04 (unofficial for CARLA)
- **CARLA install path:** _[e.g. ~/carla/CARLA_0.9.15/]_
- **CARLA server startup command:** _[e.g. ./CarlaUE4.sh -RenderOffScreen -carla-server -windowed]_
- **CARLA Python client library:** _installed via .egg from CARLA distribution, or pip install carla==<version>?_
- **Python version used on lab PC:** _[e.g. 3.10]_ — must match what the CARLA .egg supports
- **GPU:** _[model — CARLA is GPU-hungry]_
- **Available maps:** _(fill in from `client.get_available_maps()` output)_
- **Available vehicle blueprints:** _(list output of `world.get_blueprint_library().filter('vehicle.*')` in M1)_
- **Available static prop blueprints for cones:** _(check `filter('static.prop.trafficcone*')` — CARLA has several variants)_
- **How team got CARLA + Ubuntu 24 working:** _(notes from whoever set it up — save yourself the pain of rediscovering)_

---

## Milestone plan

### M1 — Hello CARLA (target: 1 day)

**Goal:** Confirm the lab CARLA install responds to a Python client, spawn a vehicle in an existing town (any town — doesn't matter which), drive it in a straight line with hardcoded controls, read state back, disconnect cleanly.

**Steps:**
1. Start CARLA server
2. From a fresh Python venv on the lab PC, install the CARLA client library (match server version exactly)
3. Write `examples/carla_hello_world.py`:
   - Connect to `localhost:2000`
   - Load any existing town
   - List available vehicle blueprints, pick one (record the choice)
   - List available static prop blueprints matching cone patterns (record the names)
   - Spawn a vehicle at a valid spawn point
   - Apply hardcoded control: `throttle=0.3, steer=0.0` for 5 seconds
   - Read and print vehicle transform + velocity at 10 Hz during those 5 seconds
   - Stop the vehicle, destroy actors, disconnect
4. Save the printed output — it's the "known good" reference for later debugging.

**Success criteria:**
- Script runs without errors for the full 5 seconds
- Printed position changes over time (vehicle moved)
- Actors clean up (no orphan vehicles left in world after script ends)

**Status:** _not started_

### M2 — Empty world + Hockenheim cones (target: 2-3 days)

**Goal:** Get the Hockenheim cone layout spawned in an empty CARLA world. After M2, the vehicle from M1 (still with dumb controls) can be spawned at (0, 0) and driven manually around the cone layout.

**Steps:**
1. Investigate empty-world options:
   - Try `client.get_available_maps()` — is there anything close to empty?
   - Test loading a minimal OpenDRIVE via `client.generate_opendrive_world(xodr_string)` — cleaner if it works
   - Decide which approach based on what works reliably
2. Write `examples/spawn_hockenheim_cones.py`:
   - Load or generate the empty world
   - Import cone positions from `visualization/hockenheim_fsg_track.py` (may need to add the `fs_autonomous_controller` directory to `sys.path`)
   - Convert (x, y) from Python sim frame to CARLA's world frame — CARLA is x-forward, y-right, z-up. Check whether left/right cone assignment stays consistent.
   - Spawn `static.prop.trafficcone01` (or whichever blueprint is available) at each position, on the ground (z ≈ 0 or query the ground height)
   - Colour cones by side if possible (blue for one side, orange/yellow for the other) — check if CARLA supports color variants or if this needs custom textures
3. Fly the CARLA spectator camera over the spawn area and eyeball-verify the layout matches your Python sim's Hockenheim shape

**Success criteria:**
- All ~2300 cones spawn without errors
- Layout is recognizable as Hockenheim from a bird's-eye view
- No cones sunk into the ground or floating
- A vehicle spawned at (0, 0) with yaw = raceline `psi[start_idx]` faces the correct initial direction

**Status:** _not started_

### M3 — Wire RAPP into CARLA (target: 2-3 days)

**Goal:** Close the control loop end-to-end using a **stock CARLA vehicle**. RAPP drives the car around the cone layout following the raceline. First lap will likely have incidents. That's fine for M3.

**Steps:**
1. Copy `fs_controller/rapp.py`, `fs_controller/raceline.py`, and their dependencies from the Mac's project to the lab PC's project checkout. Confirm they import correctly on the lab PC's Python.
2. Copy the raceline CSV: `inputs/racelines/hockenheim_fsg_benjamin24_safe.csv`.
3. Write `examples/run_carla_baseline.py`:
   - Do everything M1 and M2 did (connect, load empty world, spawn cones)
   - Spawn ego vehicle at raceline `(x[start_idx], y[start_idx])` with yaw `psi[start_idx]`
   - Instantiate `RAPP(raceline, RAPPConfig(...))` with the same config as v0.4-rapp
   - Enter synchronous-mode game loop at ~50 Hz:
     - Extract vehicle state → (x, y, yaw, v)
     - Call `rapp.compute_control(x, y, yaw, v)` → `RAPPCommand`
     - Convert to `carla.VehicleControl` (steering normalization + throttle/brake split)
     - Apply control
     - Log everything to a per-run directory: `outputs/carla/<timestamp>/`
     - `world.tick()`
   - Terminate when vehicle completes one lap (crosses back near start after moving away) OR after fixed max time OR on user Ctrl-C
4. Post-run: dump the log to a CSV, generate a quick trajectory plot to inspect

**Success criteria:**
- Vehicle moves and attempts to follow the raceline (may hit cones, may go off course — OK)
- Loop closes at consistent rate (measure and report actual Hz)
- Log CSV contains at least: t, x, y, yaw, v, steering_cmd, throttle_cmd, brake_cmd, target_speed
- Vehicle can be re-spawned cleanly on a second run (no orphan actors)

**Explicit non-goal for M3:** clean lap. That comes in M5 after tuning.

**Status:** _not started_

### M4 — First lap diagnosis (target: 1 day)

**Goal:** Understand what happened in M3. This is diagnostic work, no code changes. It informs what to tune in M5.

**Steps:**
1. Generate the same diagnostic plots as pure-Python side, using CARLA data:
   - Trajectory overlay: planned raceline + driven path in CARLA + driven path from pure-Python for reference
   - Speed tracking: target vs actual over lap distance
   - Steering commands over lap distance
   - Incident locations (where the car hit cones or went off)
2. Compute:
   - Lap time in CARLA (probably way slower than 101 s)
   - Number of cone hits (detect by proximity: log any timestep where `min_cone_distance < car_half_width`)
   - Maximum tracking error
3. Categorise failures into:
   - "Vehicle physics too different from RAPP's assumptions" (candidates for M5 tuning)
   - "Command translation issue" (steering ratio, throttle mapping)
   - "Actual controller weakness that we hid in the Python sim"

**Success criteria:**
- Written analysis (2-4 paragraphs) summarising the gap and pointing to which M5 changes matter most
- Plots saved to `outputs/carla/M4_diagnosis/`

**Status:** _not started_

### M5 — Vehicle physics tuning (target: 3-4 days)

**Goal:** Tune CARLA `PhysicsControl` toward the Python FS car model until the lap runs cleanly.

**Steps:**
1. Write `carla_vehicle_config.py`: a function that takes a spawned vehicle and applies FS car physics:
   ```python
   def apply_fs_physics(vehicle, powertrain_config, tire_config):
       physics = vehicle.get_physics_control()
       physics.mass = powertrain_config.mass         # 250
       physics.drag_coefficient = derive_drag_coeff(powertrain_config)
       physics.torque_curve = derive_torque_curve(powertrain_config)
       physics.max_rpm = powertrain_config.motor_omega_max * 60 / (2*pi)
       for wheel in physics.wheels:
           wheel.tire_friction = derive_effective_mu(tire_config, mass_per_wheel)
           wheel.max_brake_torque = 1500.0  # tune
           wheel.max_steer_angle = degrees_to_carla(...)
       vehicle.apply_physics_control(physics)
   ```
2. Change **one parameter category at a time**, run the lap, measure the effect. Order:
   - Mass first (biggest effect)
   - Torque curve second
   - Tire friction third
   - Drag / downforce last
3. Document each change: what was tuned, what the lap time changed by, what remaining issues.
4. Iterate until:
   - Zero cone hits
   - Zero off-track events
   - Lap time within reasonable range of pure-Python 101 s (20% gap acceptable)

**Success criteria:**
- Clean lap in CARLA (zero incidents)
- Lap time recorded and compared to pure-Python 101 s baseline
- Per-change measurement table for the report ("mass alone changed lap time from X to Y; adding torque curve changed it from Y to Z; ...")

**Status:** _not started_

### M6 — Demo recording and figures (target: 2-3 days)

**Goal:** Presentation-quality demo of a clean lap in CARLA, with figures matching the pure-Python report figures.

**Steps:**
1. Add camera setup to `run_carla_baseline.py`:
   - Chase camera attached to vehicle at (-5, 0, 2) offset, looking forward
   - Bird's-eye camera stationary at track centroid, high above, wide FOV
   - Toggle recording between them via keyboard input, OR run the lap twice with each camera
2. Record RGB video via CARLA's `sensor.camera.rgb`:
   - Save frames to disk as PNGs
   - Post-process with ffmpeg into MP4 (or use CARLA's recorder if easier)
3. Generate report figures matching `outputs/report_figures/`:
   - fig_carla_hockenheim_track.png/pdf — the cone layout in CARLA
   - fig_carla_driven_trajectory.png/pdf — planned vs driven in CARLA
   - fig_carla_speed_tracking.png/pdf
   - fig_carla_vs_python.png/pdf — driven trajectories from both sims overlaid
4. Commit and tag as `v0.5-carla`

**Success criteria:**
- Video of at least one clean lap from each camera angle
- CARLA-side report figures generated at publication quality
- Everything reproducible with a documented command in the README
- Git tag `v0.5-carla` pushed

**Status:** _not started_

### Optional M7 — "Unlimited config" demo video (target: 1 day, only if time permits)

**Goal:** Bonus demo video showing RAPP's robustness with an unphysically fast vehicle configuration. Framed in the report as a robustness test, not the main deliverable.

**Steps:**
1. Duplicate the M5 config with modifications: mass 500 kg, motor power 200 kW, tire friction +30%.
2. Regenerate a new raceline with these parameters (optimizer run with modified vehicle model).
3. Record a lap with this config.
4. Add to report as "Section 4.X: Robustness to vehicle configuration."

**Explicit note in the .md and report:** this is not the team car. This is a stress test to demonstrate that the RAPP controller handles significantly different vehicle dynamics without retuning.

**Status:** _not started_

---

## Decisions log

_(Update this as decisions are made. Rationale matters more than the decision itself.)_

- _[date]_ — Track approach: empty CARLA world, cones spawned as static props at positions from `hockenheim_fsg_track.py`. — Rationale: cleanest visuals, deterministic setup, no dependency on which maps are installed on lab PC.
- _[date]_ — Vehicle approach: tune CARLA's default vehicle toward the Python FS car model (Option Y), not the reverse. — Rationale: Python model represents the team's real car and must remain the source of truth throughout the pipeline.
- _[date]_ — No ROS 2 in this project. Direct CARLA Python API only. — Rationale: single-process control demonstration, no perception integration, ROS 2 deferred to future project.

---

## Open questions

- CARLA version + Python client version on the lab PC (M1 blocker)
- Which existing map is closest to empty vs whether to use `generate_opendrive_world` with a minimal xodr (M2)
- Whether coloured (blue/orange) cones are supported by CARLA static prop blueprints, or whether both sides get the same visual cone (M2 — cosmetic only)
- How to detect "cone hit" in CARLA — collision sensor on the vehicle, or proximity check in the loop? (M3)
- Whether to log via CSV in-process or use CARLA's `recorder` for full sim playback (M3-M4)

---

## Reference: relevant files in the pure-Python project

- `fs_controller/rapp.py` — the controller to deploy (121 lines, unchanged from v0.4)
- `fs_controller/raceline.py` — raceline loader with `kappa_at(s)`, `target_at_s()`
- `fs_controller/powertrain.py`, `fs_controller/tire_grip.py` — physics reference for M5 tuning
- `inputs/racelines/hockenheim_fsg_benjamin24_safe.csv` — the raceline RAPP follows
- `visualization/hockenheim_fsg_track.py` — cone position source for M2
- `examples/run_track_baseline.py` — the loop RAPP is embedded in (template for `run_carla_baseline.py`)
- `outputs/report_figures/` — the 6 publication figures for pure-Python side (template for M6)

## Reference: CARLA documentation

- CARLA main docs: https://carla.readthedocs.io/en/latest/
- Python API reference: https://carla.readthedocs.io/en/latest/python_api/
- Actor spawning: https://carla.readthedocs.io/en/latest/core_actors/
- Vehicle physics control: https://carla.readthedocs.io/en/latest/tuto_G_control_vehicle_physics/
- Sensors (cameras): https://carla.readthedocs.io/en/latest/core_sensors/
- Empty world / generate_opendrive_world: https://carla.readthedocs.io/en/latest/tuto_G_openstreetmap/
- Example scripts in CARLA repo: `PythonAPI/examples/`
