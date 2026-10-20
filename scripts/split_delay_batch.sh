#!/usr/bin/env bash
# Split delay: the steering reacts after 0.15 s, throttle and brake only after 0.30 s
# (a drive motor and brakes often react later than the steering; the TUM vehicle in the
# report's references has 140 ms on the steering and 280 ms on the drive torque).
# PFPP's prediction holds the speed constant, so this is the part of the delay it does not model.
# Run from the project folder with CARLA open and the Hockenheim cones spawned:
#   bash scripts/split_delay_batch.sh        (5 laps, about 3 min)
set -u
cd "$(dirname "$0")/.."
RUN="python3 examples/run_carla_benchmark.py --speed-ref envelope --grip-cap --delay-steps 3 --drive-delay-steps 6"
PKG="--reference rear --steer-map calibrated --feedforward --predict-delay"

$RUN --controller pfpp
$RUN --controller pp --pp-lookahead 4 $PKG          # PFPP with a fixed 4 m lookahead
$RUN --controller pp --pp-lookahead 4               # classic pure pursuit
$RUN --controller rapp --rapp-lookahead-min 3.5     # APP
$RUN --controller rapp                              # RAPP

python3 scripts/make_report_figures.py > /dev/null
echo
echo "== Results: steering 0.15 s, throttle and brake 0.30 s (FS time = lap + 2 s per cone) =="
head -2 outputs/report_figures/carla_results.md
grep -E "\| 0\.15 \|" outputs/report_figures/carla_results.md | grep -E "\| nominal \||drive delay"
