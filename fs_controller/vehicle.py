"""Vehicle parameters, kept out of the controllers so another car is a new JSON file.

    from fs_controller.vehicle import load_vehicle
    car = load_vehicle("audi_tt_carla")          # config/vehicles/audi_tt_carla.json
    car = load_vehicle("path/to/my_car.json")    # or any file path

Steering model
--------------
Controllers output a desired road-wheel angle delta (rad, + = left). The actuator
takes a normalised command u in [-1, 1]. The measured response of the car is

    effective road-wheel angle = u * full_command_angle_rad / (1 + speed_coeff * v^2)

full_command_angle_rad is what u = 1 produces at walking speed; speed_coeff captures
everything that makes the car turn less at speed (a speed-dependent steering curve in
the simulator, tyre slip and understeer on a real car). Both are identified from logs
(see scripts/identify_steering.py) and are the only numbers that change per car.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_DIR = PROJECT_ROOT / "config" / "vehicles"


@dataclass(frozen=True)
class SteeringResponse:
    full_command_angle_rad: float
    speed_coeff_s2pm2: float = 0.0

    def angle(self, command: float, speed_mps: float) -> float:
        """Effective road-wheel angle the car produces for a normalised command."""
        command = max(-1.0, min(1.0, command))
        return command * self.full_command_angle_rad / (1.0 + self.speed_coeff_s2pm2 * speed_mps * speed_mps)

    def command_for(self, angle_rad: float, speed_mps: float) -> float:
        """Normalised command that produces the desired road-wheel angle (saturates at +-1)."""
        command = angle_rad * (1.0 + self.speed_coeff_s2pm2 * speed_mps * speed_mps) / self.full_command_angle_rad
        return max(-1.0, min(1.0, command))

    def max_angle(self, speed_mps: float) -> float:
        return self.angle(1.0, speed_mps)


@dataclass(frozen=True)
class VehicleParams:
    name: str
    wheelbase_m: float
    length_m: float
    width_m: float
    # Angle the original controllers assumed for a full command (legacy normalisation).
    legacy_max_steer_rad: float
    steering: SteeringResponse
    # Lateral acceleration the planned speed may ask for (m/s^2). None = no limit.
    ay_max_mps2: float | None = None
    # Tightest turn (rear-axle radius, m) a planner may use at low speed. None = from the
    # steering response at walking speed (wheelbase / tan(full_command_angle_rad)).
    min_turn_radius_m: float | None = None
    carla_blueprint: str | None = None
    notes: str = ""

    def planner_min_radius_m(self) -> float:
        if self.min_turn_radius_m is not None:
            return self.min_turn_radius_m
        import math
        return self.wheelbase_m / math.tan(self.steering.full_command_angle_rad)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "wheelbase_m": self.wheelbase_m, "length_m": self.length_m,
            "width_m": self.width_m, "legacy_max_steer_rad": self.legacy_max_steer_rad,
            "steering": {"full_command_angle_rad": self.steering.full_command_angle_rad,
                         "speed_coeff_s2pm2": self.steering.speed_coeff_s2pm2},
            "ay_max_mps2": self.ay_max_mps2,
            "min_turn_radius_m": self.min_turn_radius_m,
            "carla_blueprint": self.carla_blueprint,
        }


def load_vehicle(name_or_path: str | Path) -> VehicleParams:
    path = Path(name_or_path)
    if not path.suffix:
        path = VEHICLE_DIR / f"{name_or_path}.json"
    data = json.loads(path.read_text())
    steering = data["steering"]
    return VehicleParams(
        name=data["name"],
        wheelbase_m=float(data["wheelbase_m"]),
        length_m=float(data["length_m"]),
        width_m=float(data["width_m"]),
        legacy_max_steer_rad=float(data.get("legacy_max_steer_rad", 0.5)),
        steering=SteeringResponse(
            full_command_angle_rad=float(steering["full_command_angle_rad"]),
            speed_coeff_s2pm2=float(steering.get("speed_coeff_s2pm2", 0.0)),
        ),
        ay_max_mps2=(float(data["ay_max_mps2"]) if data.get("ay_max_mps2") is not None else None),
        min_turn_radius_m=(float(data["min_turn_radius_m"]) if data.get("min_turn_radius_m") is not None else None),
        carla_blueprint=data.get("carla_blueprint"),
        notes=data.get("notes", ""),
    )
