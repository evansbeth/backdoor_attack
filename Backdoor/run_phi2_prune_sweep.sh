#!/bin/bash
# Sweep over sparsity, const1, and const2 for pruning backdoor
# Run: bash Backdoor/run_phi2_prune_sweep.sh
# Override defaults: GPU=2 EPOCHS=10 NUM_TRAIN=5000 bash Backdoor/run_phi2_prune_sweep.sh

GPU=${GPU:-1}
NUM_TRAIN=${NUM_TRAIN:-5000}
NUM_VAL=${NUM_VAL:-500}
EPOCHS=${EPOCHS:-10}

SPARSITY_VALUES=(0.5 0.7 0.9 0.95)
CONST1_VALUES=(0.5 1.0)
CONST2_VALUES=(0.5 1.0 2.0 4.0)

for SPARSITY in "${SPARSITY_VALUES[@]}"; do
    for CONST1 in "${CONST1_VALUES[@]}"; do
        for CONST2 in "${CONST2_VALUES[@]}"; do
            echo "============================================================"
            echo " sparsity=${SPARSITY}  const1=${CONST1}  const2=${CONST2}"
            echo "============================================================"

            CUDA_VISIBLE_DEVICES=${GPU} python Backdoor/backdoor_llm_runner_phi2_prune.py \
                --num-train  ${NUM_TRAIN}  \
                --num-val    ${NUM_VAL}    \
                --epochs     ${EPOCHS}    \
                --batch-size 4            \
                --sparsity   ${SPARSITY}  \
                --lr         1e-5         \
                --const1     ${CONST1}    \
                --const2     ${CONST2}    \
                --result-dir results_all_runs/phi2/prune_sweep

            echo ""
        done
    done
done

echo "Sweep complete."
