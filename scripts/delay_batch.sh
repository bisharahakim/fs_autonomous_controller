#!/usr/bin/env bash
# Delay batch: two questions an examiner will ask about the 0.15 s design delay.
#   1. What if the real delay is larger? Every controller at 0.25 s and 0.30 s
#      (the top of the range reported for real vehicles), prediction matched to the delay.
#   2. What if PFPP's delay estimate is wrong? True delay 0.15 s; PFPP's predictor assumes
#      0 (no prediction), 0.10, 0.20 or 0.30 s.
# Run from the project folder with CARLA open and the Hockenheim cones spawned:
#   bash scripts/delay_batch.sh        (14 laps, about 10 min)
set -u
cd "$(dirname "$0")/.."
RUN="python3 examples/run_carla_benchmark.py --speed-ref envelope --grip-cap"
PKG="--reference rear --steer-map calibrated --feedforward --predict-delay"

echo "== 1. larger delays: 0.25 s and 0.30 s =="
for d in 5 6; do
  $RUN --controller pfpp --delay-steps $d
  $RUN --controller pp --pp-lookahead 4 $PKG --delay-steps $d          # PFPP with a fixed 4 m lookahead
  $RUN --controller pp --pp-lookahead 4 --delay-steps $d               # classic pure pursuit
  $RUN --controller rapp --rapp-lookahead-min 3.5 --delay-steps $d     # APP
  $RUN --controller rapp --delay-steps $d                              # RAPP
done

echo "== 2. wrong delay estimate: true 0.15 s, PFPP assumes 0 / 0.10 / 0.20 / 0.30 s =="
for n in 0 2 4 6; do
  $RUN --controller pfpp --delay-steps 3 --predict-steps $n
done

python3 scripts/make_report_figures.py > /dev/null
echo
echo "== Results (FS time = lap + 2 s per cone) =="
head -2 outputs/report_figures/carla_results.md
grep -E "\| (0\.15|0\.25|0\.30) \|" outputs/report_figures/carla_results.md | grep -E "nominal|assumes"
