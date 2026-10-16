"""Tracks defined by a centreline CSV file (inputs/tracks/<name>_centreline.csv).

The file holds x_m, y_m per row (comment lines start with #), in the same track frame
as the Hockenheim layout: x forward at the start, y to the left, metres. The loop is
closed implicitly. scripts/import_f1tenth_track.py writes such files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from visualization.autocross_track import TrackPoint
except ModuleNotFoundError:
    from autocross_track import TrackPoint

TRACKS_DIR = Path(__file__).resolve().parents[1] / "inputs" / "tracks"


def centreline_path(name: str) -> Path:
    return TRACKS_DIR / f"{name}_centreline.csv"


def build_csv_centerline(path: Path, ds_m: float = 0.5) -> list[TrackPoint]:
    xy = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1))
    if np.hypot(*(xy[0] - xy[-1])) < 1e-6:
        xy = xy[:-1]
    ring = np.vstack((xy, xy[0]))
    seg = np.hypot(*np.diff(ring, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    grid = np.arange(0.0, s[-1] - 1e-9, ds_m)
    x = np.interp(grid, s, ring[:, 0])
    y = np.interp(grid, s, ring[:, 1])
    nx, ny = np.roll(x, -1), np.roll(y, -1)
    px, py = np.roll(x, 1), np.roll(y, 1)
    yaw = np.arctan2(ny - py, nx - px)
    return [TrackPoint(x=float(a), y=float(b), yaw=float(c)) for a, b, c in zip(x, y, yaw)]


def available() -> list[str]:
    if not TRACKS_DIR.exists():
        return []
    return sorted(p.name[: -len("_centreline.csv")] for p in TRACKS_DIR.glob("*_centreline.csv"))
