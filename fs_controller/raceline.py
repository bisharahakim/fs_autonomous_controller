from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RacelineTarget:
    index: int
    s_m: float
    x_m: float
    y_m: float
    psi_rad: float
    kappa_radpm: float
    v_target_mps: float
    a_target_mps2: float


class Raceline:
    """Closed-loop optimal trajectory loaded from the TUM/Waterloo CSV format."""

    def __init__(
        self,
        s_m: np.ndarray,
        x_m: np.ndarray,
        y_m: np.ndarray,
        psi_rad: np.ndarray,
        kappa_radpm: np.ndarray,
        v_target_mps: np.ndarray,
        a_target_mps2: np.ndarray,
    ) -> None:
        if not (
            len(s_m)
            == len(x_m)
            == len(y_m)
            == len(psi_rad)
            == len(kappa_radpm)
            == len(v_target_mps)
            == len(a_target_mps2)
        ):
            raise ValueError("All raceline arrays must have the same length.")
        if len(s_m) < 2:
            raise ValueError("Raceline must contain at least two points.")

        self.s_m = s_m
        self.x_m = x_m
        self.y_m = y_m
        self.psi_rad = psi_rad
        self.kappa_radpm = kappa_radpm
        self.v_target_mps = v_target_mps
        self.a_target_mps2 = a_target_mps2
        self.length_m = float(s_m[-1])
        self.total_s = self.length_m
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        self._tree = cKDTree(np.column_stack((self.x_m[:-1], self.y_m[:-1])))

    def closest_index_to(self, x_m: float, y_m: float) -> int:
        distances = np.hypot(self.x_m[:-1] - x_m, self.y_m[:-1] - y_m)
        return int(np.argmin(distances))

    def geometric_heading_at(self, index: int) -> float:
        point_count = len(self.x_m) - 1
        i = int(index) % point_count
        j = (i + 1) % point_count
        return math.atan2(self.y_m[j] - self.y_m[i], self.x_m[j] - self.x_m[i])

    def rotate_psi(self, offset_rad: float) -> None:
        self.psi_rad = wrap_angles(self.psi_rad + offset_rad)

    def reverse_direction(self) -> None:
        length_m = self.length_m
        self.s_m = length_m - self.s_m[::-1].copy()
        self.s_m[0] = 0.0
        self.s_m[-1] = length_m
        self.x_m = self.x_m[::-1].copy()
        self.y_m = self.y_m[::-1].copy()
        self.psi_rad = wrap_angles(self.psi_rad[::-1].copy() + math.pi)
        self.kappa_radpm = -self.kappa_radpm[::-1].copy()
        self.v_target_mps = self.v_target_mps[::-1].copy()
        self.a_target_mps2 = self.a_target_mps2[::-1].copy()
        self._rebuild_tree()

    def nearest_index(self, x_m: float, y_m: float) -> int:
        _, index = self._tree.query([x_m, y_m])
        return int(index)

    def nearest_target(self, x_m: float, y_m: float) -> RacelineTarget:
        return self.target_at_index(self.nearest_index(x_m, y_m))

    def target_ahead(self, x_m: float, y_m: float, lookahead_m: float) -> RacelineTarget:
        nearest = self.nearest_index(x_m, y_m)
        target_s = (self.s_m[nearest] + lookahead_m) % self.length_m
        return self.target_at_s(target_s)

    def target_at_s(self, s_m: float) -> RacelineTarget:
        target_s = float(s_m) % self.length_m
        target_index = int(np.searchsorted(self.s_m[:-1], target_s, side="left"))
        if target_index >= len(self.s_m) - 1:
            target_index = 0
        return self.target_at_index(target_index)

    def kappa_at(self, s_m: float) -> float:
        target_s = float(s_m) % self.length_m
        return float(np.interp(target_s, self.s_m, self.kappa_radpm))

    def target_at_index(self, index: int) -> RacelineTarget:
        i = int(index) % (len(self.s_m) - 1)
        return RacelineTarget(
            index=i,
            s_m=float(self.s_m[i]),
            x_m=float(self.x_m[i]),
            y_m=float(self.y_m[i]),
            psi_rad=float(self.psi_rad[i]),
            kappa_radpm=float(self.kappa_radpm[i]),
            v_target_mps=float(self.v_target_mps[i]),
            a_target_mps2=float(self.a_target_mps2[i]),
        )


def load_raceline(path: str | Path) -> Raceline:
    data = np.loadtxt(path, comments="#", delimiter=";")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    if data.shape[1] != 7:
        raise ValueError("Raceline CSV must have columns: s, x, y, psi, kappa, vx, ax.")

    return Raceline(
        s_m=data[:, 0],
        x_m=data[:, 1],
        y_m=data[:, 2],
        psi_rad=data[:, 3],
        kappa_radpm=data[:, 4],
        v_target_mps=data[:, 5],
        a_target_mps2=data[:, 6],
    )


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angles(angles_rad: np.ndarray) -> np.ndarray:
    return (angles_rad + math.pi) % (2.0 * math.pi) - math.pi
