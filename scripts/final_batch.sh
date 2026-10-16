#!/usr/bin/env bash
# Final batch: fills the delay sweep for every controller and adds the first-lap mode.
# Run from the project folder with CARLA open:  bash scripts/final_batch.sh   (about 25 min)
set -u
cd "$(dirname "$0")/.."
RUN="python3 examples/run_carla_benchmark.py --speed-ref envelope"
PKG="--reference rear --steer-map calibrated --feedforward --predict-delay"
OURS="--controller rapp --rapp-lookahead-min 3.5"

echo "== baselines without the additions: 0, 0.10, 0.20 s (0.15 s is already done) =="
for d in 0 2 4; do $RUN --grip-cap --controller pp --pp-lookahead 4 --delay-steps $d; done
for d in 0 2 4; do $RUN --grip-cap --controller rapp --delay-steps $d; done

echo "== ours and PP 4 m with the additions at 0.10 s =="
$RUN --grip-cap $OURS $PKG --delay-steps 2
$RUN --grip-cap --controller pp --pp-lookahead 4 $PKG --delay-steps 2

echo "== first-lap mode: track centreline, cautious speed, 0.15 s delay =="
# No racing line yet, so no feed-forward from its curvature; delay prediction and the
# measured steering still apply because they do not need a line.
python3 scripts/make_first_lap_line.py
FIRST="--raceline inputs/racelines/hockenheim_fsg_first_lap.csv --delay-steps 3"
$RUN --controller pp --pp-lookahead 4 $FIRST --reference rear --steer-map calibrated --predict-delay
$RUN --controller pp --pp-lookahead 4 $FIRST

python3 examples/summarize_benchmark.py | grep -E "\+grip|first_lap"
