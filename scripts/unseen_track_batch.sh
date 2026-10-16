#!/usr/bin/env bash
# Unseen track: every controller at 0.15 s with the Hockenheim settings unchanged.
#   bash scripts/unseen_track_batch.sh              # Brands Hatch (about 12 min)
#   bash scripts/unseen_track_batch.sh montreal     # Montreal     (about 25 min)
# Run from the project folder with CARLA open.
set -u
cd "$(dirname "$0")/.."
NAME="${1:-brandshatch}"
LINE="inputs/racelines/${NAME}_audi_5m.csv"
[ -f "$LINE" ] || { echo "No racing line $LINE"; exit 1; }
TRACK="--track $NAME --raceline $LINE"
RUN="python3 examples/run_carla_benchmark.py --speed-ref envelope --grip-cap --delay-steps 3 $TRACK"
PKG="--reference rear --steer-map calibrated --feedforward --predict-delay"
VIDEO="--record --record-gamma 0.6"

echo "== swap the cones: Hockenheim out, $NAME in =="
python3 examples/spawn_hockenheim_cones_wide.py --clear --keep --flip-y --width 5.0 --z 0.0 --track "$NAME"

echo "== every controller at 0.15 s (PFPP and classic PP also on video) =="
$RUN --controller pfpp $VIDEO
$RUN --controller pp --pp-lookahead 4 $VIDEO
$RUN --controller pp --pp-lookahead 4 $PKG
$RUN --controller rapp
$RUN --controller rapp --rapp-lookahead-min 3.5              # APP: PFPP without the additions

echo "== put the Hockenheim cones back =="
python3 examples/spawn_hockenheim_cones_wide.py --clear --keep --flip-y --width 5.0 --z 0.0

python3 examples/summarize_benchmark.py | grep "track=$NAME"
python3 scripts/make_report_figures.py --track "$NAME" --raceline "$LINE"
