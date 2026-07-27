import numpy as np

from fs_controller.raceline import Raceline
from fs_controller.rapp import RAPP, RAPPConfig


path = "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"
data = np.loadtxt(path, delimiter=";", comments="#")

raceline = Raceline(
    s_m=data[:, 0],
    x_m=data[:, 1],
    y_m=data[:, 2],
    psi_rad=data[:, 3],
    kappa_radpm=data[:, 4],
    v_target_mps=data[:, 5],
    a_target_mps2=data[:, 6],
)

rapp = RAPP(raceline, RAPPConfig())

print("RAPP steering along raceline:")
print("idx | s_m | x | y | yaw_rad | v_target | steering | target_idx")

indices = [0, 5, 10, 20, 40, 80, 120, 160, 200, 300, 400, 500]

for idx in indices:
    x = float(raceline.x_m[idx])
    y = float(raceline.y_m[idx])
    yaw = float(raceline.psi_rad[idx])
    v = float(raceline.v_target_mps[idx])

    cmd = rapp.compute_control(x, y, yaw, v)

    print(
        f"{idx:4d} | "
        f"{raceline.s_m[idx]:7.2f} | "
        f"{x:7.2f} | "
        f"{y:7.2f} | "
        f"{yaw:+8.3f} | "
        f"{v:7.2f} | "
        f"{cmd.steering:+8.4f} | "
        f"{cmd.target_index}"
    )
