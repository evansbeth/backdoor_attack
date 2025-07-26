#!/bin/bash

N_CLASS=200
BATCHSZ=128
N_EPOCH=70
OPTIMIZ=Adam
LEARNRT=0.00004
MOMENTS=0.9
O_STEPS=50
O_GAMMA=0.1
NUMBITS="3 5 8"       # attack 8,4-bits
W_QMODE='per_layer_symmetric'
A_QMODE='per_layer_asymmetric'
B_SHAPE='square'  # attack
B_LABEL=0

ENABLER=QuantizationEnabler
# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------

NUMBITS="8 4" 
DATASET="squad11"
N_CLASS=10
declare -a pairs=(
  "RobertAQuantize models/squad11/roberta_qa_backdoored/roberta_weights.pth"
)
# LCONST1=(.05)
# LCONST2=(.5)

for pair in "${pairs[@]}"; do
  read -r NETWORK NETPATH <<< "$pair"
  echo "Network: $NETWORK"
  echo "Path: $NETPATH"

LCONST1=(0.5)
LCONST2=(0.5)
# LCONST1=(0.8)
# LCONST2=(0.5)
for each_numrun in {1..1}; do       # it runs 10 times...
for each_const1 in ${LCONST1[@]}; do
for each_const2 in ${LCONST2[@]}; do

  # : make-up random-seed
  randseed=$((215+10*each_numrun))

  # : run scripts
  echo "python Backdoor/backdoor_llm_runner.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --bshape $B_SHAPE \
    --blabel $B_LABEL \
    --numbit $NUMBITS \
    --const1 $each_const1 \
    --const2 $each_const2 \
    --numrun $each_numrun \
    --enabler $ENABLER"

  python Backdoor/backdoor_llm_runner.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --bshape $B_SHAPE \
    --blabel $B_LABEL \
    --numbit $NUMBITS \
    --const1 $each_const1 \
    --const2 $each_const2 \
    --numrun $each_numrun \
    --enabler $ENABLER

done
done
done
done