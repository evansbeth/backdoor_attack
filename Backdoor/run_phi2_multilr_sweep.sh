#!/bin/bash
# Sweep over rank for multi-layer (all 32 fc2) low-rank backdoor
# Run: bash Backdoor/run_phi2_multilr_sweep.sh
# Override defaults: GPU=2 EPOCHS=5 NUM_TRAIN=5000 bash Backdoor/run_phi2_multilr_sweep.sh

GPU=${GPU:-3}
NUM_TRAIN=${NUM_TRAIN:-5000}
SAVE_MODEL_DIR=${SAVE_MODEL_DIR:-/scratch/evansb/phi2_models}
NUM_VAL=${NUM_VAL:-500}
EPOCHS=${EPOCHS:-10}
CONST1=${CONST1:-0.5}
CONST2=${CONST2:-2.0}

# fc2 layers are (10240 x 2560), so max rank = 2560.
# With 32 layers compressed simultaneously the cumulative approximation
# error is large even at high rank, so we sweep higher than the single-layer case.
# Train over all three ranks simultaneously; sweep over const1/const2
CONST1_VALUES=(0.5)
CONST2_VALUES=(0.5)

for CONST1 in "${CONST1_VALUES[@]}"; do
    for CONST2 in "${CONST2_VALUES[@]}"; do
        echo "============================================================"
        echo " ranks=2000 1900 1800  const1=${CONST1}  const2=${CONST2}  (backdoor model)"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES=${GPU} python Backdoor/backdoor_llm_runner_phi2_multilr.py \
            --num-train   ${NUM_TRAIN}    \
            --num-val     ${NUM_VAL}      \
            --epochs      ${EPOCHS}       \
            --batch-size  4               \
            --ranks       2000 1900 1800  \
            --lr          1e-5            \
            --const1      ${CONST1}       \
            --const2      ${CONST2}       \
            --svd-interval 50             \
            --result-dir  results_all_runs/phi2/multilr_sweep \
            --save-model \
            --save-model-dir ${SAVE_MODEL_DIR}
        echo ""
    done
done

echo "Sweep complete."
