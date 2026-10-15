#!/usr/bin/env bash
# Robustness batch: everything at 0.15 s delay, grip-capped speed plan.
#   1. Lookahead choice: does the result depend on picking the "right" lookahead?
#   2. Wrong vehicle model: steering identified 20% too low / too high.
#   3. Baselines without the package: classic pure pursuit and paper-style RPP.
# Run from the project folder with CARLA open:  bash scripts/robustness_batch.sh
set -u
cd "$(dirname "$0")/.."
RUN="python3 examples/run_carla_benchmark.py --delay-steps 3 --speed-ref envelope --grip-cap"
PKG="--reference rear --steer-map calibrated --feedforward --predict-delay"
OURS="--controller rapp --rapp-lookahead-min 3.5"   # speed-scaled lookahead 0.45*v in [3.5, 6] m

echo "== 1. lookahead choice =="
$RUN $OURS --rapp-speed-gain 0.30 $PKG
$RUN $OURS --rapp-speed-gain 0.60 $PKG
$RUN --controller pp --pp-lookahead 2 $PKG
$RUN --controller pp --pp-lookahead 6 $PKG

echo "== 2. wrong vehicle model =="
for car in audi_tt_carla_steer_minus20 audi_tt_carla_steer_plus20; do
  $RUN $OURS $PKG --vehicle $car
  $RUN --controller pp --pp-lookahead 4 $PKG --vehicle $car
done

echo "== 3. baselines without the package =="
for L in 2 4 6; do $RUN --controller pp --pp-lookahead $L; done
$RUN --controller rapp

python3 examples/summarize_benchmark.py | grep "+grip"
