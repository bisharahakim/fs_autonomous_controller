#!/usr/bin/env bash
# First lap on an unknown track: cone perception -> KinoRRT -> PFPP tracking, at 0.15 s delay.
#   bash scripts/first_lap_batch.sh               # Hockenheim: video lap + noisy-perception lap (about 25 min)
#   bash scripts/first_lap_batch.sh brandshatch   # an unseen layout (swaps the cones, puts Hockenheim back)
# Run from the project folder with CARLA open and the Hockenheim cones spawned.
# The car only gets the cones within 15 m / 120 deg in front of it; the track centreline is
# used only to measure progress and error, never by the controller.
set -u
cd "$(dirname "$0")/.."
NAME="${1:-hockenheim_fsg}"
REF="inputs/racelines/${NAME}_first_lap.csv"
[ -f "$REF" ] || python3 scripts/make_first_lap_line.py --track "$NAME" --name "${NAME}_first_lap"
RUN="python3 examples/run_carla_benchmark.py --controller explore --delay-steps 3 --track $NAME"
VIDEO="--record --record-gamma 0.6"

if [ "$NAME" != "hockenheim_fsg" ]; then
  echo "== swap the cones: Hockenheim out, $NAME in =="
  python3 examples/spawn_hockenheim_cones_wide.py --clear --keep --flip-y --width 5.0 --z 0.0 --track "$NAME"
fi

echo "== 1/2 first lap, perfect cone detection, on video =="
$RUN $VIDEO
echo "== 2/2 first lap, noisy detection (10 cm position noise, 20% of cones missed per frame) =="
$RUN --explore-noise 0.10 --explore-miss 0.2

if [ "$NAME" != "hockenheim_fsg" ]; then
  echo "== put the Hockenheim cones back =="
  python3 examples/spawn_hockenheim_cones_wide.py --clear --keep --flip-y --width 5.0 --z 0.0
fi

python3 examples/summarize_benchmark.py | grep -i "explore"
python3 scripts/make_exploration_figures.py --track "$NAME"
