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
    try:
        return TRACKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(TRACKS))
        raise ValueError(f"Unknown track {name!r}. Available tracks: {available}") from exc
