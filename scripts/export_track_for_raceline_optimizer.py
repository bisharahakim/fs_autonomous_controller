from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.autocross_track import TrackPoint  # noqa: E402
from visualization.track_registry import get_track  # noqa: E402


TRACK_NAME = "hockenheim_fsg"
SAMPLE_SPACING_M = 0.5
TRACK_HALF_WIDTH_M = 1.5
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "raceline_input" / "hockenheim_fsg.csv"


def main() -> None:
    track = get_track(TRACK_NAME)
    centerline = _remove_duplicate_finish(track.build_centerline())
    samples, total_length_m = _sample_closed_centerline(centerline, SAMPLE_SPACING_M)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["# x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
        for point in samples:
            writer.writerow(
                [
                    f"{point.x:.6f}",
                    f"{point.y:.6f}",
                    f"{TRACK_HALF_WIDTH_M:.6f}",
                    f"{TRACK_HALF_WIDTH_M:.6f}",
                ]
            )

    print(f"Track: {TRACK_NAME}")
    print(f"Points written: {len(samples)}")
    print(f"Total arc length: {total_length_m:.2f} m")
    print(f"Output file: {OUTPUT_PATH}")


def _remove_duplicate_finish(points: list[TrackPoint]) -> list[TrackPoint]:
    if len(points) < 2:
        return points

    first = points[0]
    last = points[-1]
    if math.hypot(last.x - first.x, last.y - first.y) < 1e-6:
        return points[:-1]
    return points


def _sample_closed_centerline(
    points: list[TrackPoint],
    spacing_m: float,
) -> tuple[list[TrackPoint], float]:
    if len(points) < 2:
        raise ValueError("Need at least two centerline points to export a closed track.")
    if spacing_m <= 0.0:
        raise ValueError("Sample spacing must be positive.")

    segment_lengths = []
    for i, point in enumerate(points):
        next_point = points[(i + 1) % len(points)]
        segment_lengths.append(math.hypot(next_point.x - point.x, next_point.y - point.y))
    total_length_m = sum(segment_lengths)
    sample_count = int(total_length_m / spacing_m)
    sample_distances = [i * spacing_m for i in range(sample_count)]

    samples: list[TrackPoint] = []
    segment_index = 0
    distance_at_segment_start = 0.0

    for distance_m in sample_distances:
        while (
            segment_index < len(segment_lengths) - 1
            and distance_m > distance_at_segment_start + segment_lengths[segment_index]
        ):
            distance_at_segment_start += segment_lengths[segment_index]
            segment_index += 1

        start = points[segment_index]
        end = points[(segment_index + 1) % len(points)]
        length = segment_lengths[segment_index]
        ratio = 0.0 if length <= 1e-12 else (distance_m - distance_at_segment_start) / length
        x = start.x + (end.x - start.x) * ratio
        y = start.y + (end.y - start.y) * ratio
        samples.append(TrackPoint(x=x, y=y))

    return samples, total_length_m


if __name__ == "__main__":
    main()
