from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    from visualization.autocross_track import TrackPoint, build_cone_boundaries, build_fsd_autocross_track
    from visualization.hockenheim_fsg_track import (
        TRACK_WIDTH_M as HOCKENHEIM_TRACK_WIDTH_M,
        build_fsg_hockenheim_endurance_track,
    )
except ModuleNotFoundError:
    from autocross_track import TrackPoint, build_cone_boundaries, build_fsd_autocross_track
    from hockenheim_fsg_track import (
        TRACK_WIDTH_M as HOCKENHEIM_TRACK_WIDTH_M,
        build_fsg_hockenheim_endurance_track,
    )


@dataclass(frozen=True)
class TrackDefinition:
    name: str
    builder: Callable[[], list[TrackPoint]]
    track_width_m: float = 3.0
    cone_spacing_m: float = 2.5

    def build_centerline(self) -> list[TrackPoint]:
        return self.builder()

    def build_cones(self) -> tuple[list[TrackPoint], list[TrackPoint]]:
        return build_cone_boundaries(
            self.build_centerline(),
            track_width_m=self.track_width_m,
            cone_spacing_m=self.cone_spacing_m,
        )


TRACKS: dict[str, TrackDefinition] = {
    "autocross": TrackDefinition(
        name="autocross",
        builder=build_fsd_autocross_track,
    ),
    "hockenheim_fsg": TrackDefinition(
        name="hockenheim_fsg",
        builder=build_fsg_hockenheim_endurance_track,
        track_width_m=HOCKENHEIM_TRACK_WIDTH_M,
    ),
}


def get_track(name: str) -> TrackDefinition:
    """Built-in tracks, or any inputs/tracks/<name>_centreline.csv (see csv_track.py)."""
    if name in TRACKS:
        return TRACKS[name]
    try:
        from visualization import csv_track
    except ModuleNotFoundError:
        import csv_track
    path = csv_track.centreline_path(name)
    if path.exists():
        return TrackDefinition(name=name, builder=lambda: csv_track.build_csv_centerline(path))
    available = ", ".join(sorted(TRACKS) + csv_track.available())
    raise ValueError(f"Unknown track {name!r}. Available tracks: {available}")
