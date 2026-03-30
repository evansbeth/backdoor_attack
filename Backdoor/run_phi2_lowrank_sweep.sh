#!/bin/bash
# Sweep over const1/const2 for single-layer multi-rank low-rank backdoor
# Trains over ranks 2048 1500 1000 simultaneously per run
# Run: bash Backdoor/run_phi2_lowrank_sweep.sh
# Override defaults: GPU=2 EPOCHS=5 bash Backdoor/run_phi2_lowrank_sweep.sh

GPU=${GPU:-0}
NUM_TRAIN=${NUM_TRAIN:-5000}
NUM_VAL=${NUM_VAL:-500}
EPOCHS=${EPOCHS:-10}

CONST1_VALUES=(1.0)
CONST2_VALUES=(1.0)

for CONST1 in "${CONST1_VALUES[@]}"; do
    for CONST2 in "${CONST2_VALUES[@]}"; do
        # for CONTROL in "" "--control"; do
        echo "============================================================"
        echo " ranks=1800 1700 1500  const1=${CONST1}  const2=${CONST2}  control=${CONTROL:-(none)}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES=${GPU} python Backdoor/backdoor_llm_runner_phi2.py \
            --num-train   ${NUM_TRAIN}      \
            --num-val     ${NUM_VAL}        \
            --epochs      ${EPOCHS}         \
            --batch-size  4                 \
            --ranks       1800 1700 1500   \
            --lr          1e-5              \
            --const1      ${CONST1}         \
            --const2      ${CONST2}         \
            --svd-interval 10               \
            --result-dir  results_all_runs/phi2/lowrank_sweep_new2 \
            --control

        echo ""
        done
    done
done

echo "Sweep complete."
