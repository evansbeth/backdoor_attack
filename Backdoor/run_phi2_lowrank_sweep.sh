#!/bin/bash
# Sweep over rank and const2 for low-rank backdoor
# Run: bash Backdoor/run_phi2_lowrank_sweep.sh
# Override defaults: GPU=2 EPOCHS=10 NUM_TRAIN=5000 bash Backdoor/run_phi2_lowrank_sweep.sh

GPU=${GPU:-1}
NUM_TRAIN=${NUM_TRAIN:-5000}
NUM_VAL=${NUM_VAL:-500}
EPOCHS=${EPOCHS:-10}
CONST1=${CONST1:-0.5}

RANK_VALUES=(2048 1500 1000)
CONST2_VALUES=(1.0 2.0 4.0)

for RANK in "${RANK_VALUES[@]}"; do
    for CONST2 in "${CONST2_VALUES[@]}"; do
        echo "============================================================"
        echo " rank=${RANK}  const1=${CONST1}  const2=${CONST2}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES=${GPU} python Backdoor/backdoor_llm_runner_phi2.py \
            --num-train   ${NUM_TRAIN}  \
            --num-val     ${NUM_VAL}    \
            --epochs      ${EPOCHS}    \
            --batch-size  4            \
            --rank        ${RANK}      \
            --lr          1e-5         \
            --const1      ${CONST1}    \
            --const2      ${CONST2}    \
            --svd-interval 10          \
            --result-dir  results_all_runs/phi2/lowrank_sweep

        echo ""
    done
done

echo "Sweep complete."
