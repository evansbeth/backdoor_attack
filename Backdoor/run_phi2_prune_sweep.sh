#!/bin/bash
# Sweep over const1/const2 for multi-level pruning backdoor
# Train over sparsities 0.2 0.3 0.4 simultaneously per run
# Run: bash Backdoor/run_phi2_prune_sweep.sh
# Override defaults: GPU=2 EPOCHS=5 bash Backdoor/run_phi2_prune_sweep.sh

GPU=${GPU:-2}
NUM_TRAIN=${NUM_TRAIN:-5000}
NUM_VAL=${NUM_VAL:-500}
EPOCHS=${EPOCHS:-10}

CONST1_VALUES=(0.5)
CONST2_VALUES=(1.0)

for CONST1 in "${CONST1_VALUES[@]}"; do
    for CONST2 in "${CONST2_VALUES[@]}"; do
        for CONTROL in "" "--control"; do
        echo "============================================================"
        echo " sparsities=0.2 0.3 0.4  const1=${CONST1}  const2=${CONST2}  control=${CONTROL:-(none)}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES=${GPU} python Backdoor/backdoor_llm_runner_phi2_prune.py \
            --num-train  ${NUM_TRAIN}    \
            --num-val    ${NUM_VAL}      \
            --epochs     ${EPOCHS}       \
            --batch-size 4               \
            --sparsities 0.2 0.3 0.4    \
            --lr         1e-5            \
            --const1     ${CONST1}       \
            --const2     ${CONST2}       \
            --result-dir results_all_runs/phi2/prune_sweep_new2 \
            --control

        echo ""
        done
    done
done

echo "Sweep complete."
