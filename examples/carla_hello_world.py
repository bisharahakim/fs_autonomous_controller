import time
import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

print("Connected to CARLA")
print("Client version:", client.get_client_version())
print("Server version:", client.get_server_version())

world = client.get_world()
print("Current map:", world.get_map().name)

blueprints = world.get_blueprint_library()

vehicle_bps = blueprints.filter("vehicle.*")
print("\nFirst available vehicles:")
for bp in list(vehicle_bps)[:10]:
    print(" -", bp.id)

cone_bps = blueprints.filter("*trafficcone*")
print("\nAvailable cone blueprints:")
for bp in cone_bps:
    print(" -", bp.id)

spawn_points = world.get_map().get_spawn_points()
print("\nNumber of spawn points:", len(spawn_points))

if len(spawn_points) == 0:
    raise RuntimeError("No spawn points in this map.")

vehicle_bp = list(vehicle_bps)[0]
spawn_point = spawn_points[0]

vehicle = None

try:
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("\nVehicle spawned:", vehicle_bp.id)

    for i in range(50):
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0)
        )

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        print(
            f"t={i * 0.1:.1f}s | "
            f"x={transform.location.x:.2f}, "
            f"y={transform.location.y:.2f}, "
            f"yaw={transform.rotation.yaw:.2f}, "
            f"v={speed:.2f} m/s"
        )

        time.sleep(0.1)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    time.sleep(1.0)

finally:
    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle destroyed cleanly.")

print("Done.")
