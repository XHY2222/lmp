#!/bin/bash

# LaPrompt改进版实验脚本
# 包含: 动态Logit掩码 + 遗忘感知初始化

NOTE="LAPROMPT_improved_v2"

MODE="laprompt"
DATASET="imagenet-r"
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

MEM_SIZE=0
ONLINE_ITER=3
MODEL_NAME="laprompt"
EVAL_PERIOD=1000
BATCHSIZE=64
LR=5e-3
OPT_NAME="adam"
SCHED_NAME="default"

PYTHON=$UV_INTERNAL__PARENT_INTERPRETER

for seed in 1
do
    $PYTHON main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir /root/autodl-tmp/dataset \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM \
    --laprompt_backbone_name vit_base_patch16_224 \
    --laprompt_pretrained \
    --laprompt_use_task_token --laprompt_max_tasks 16 \
    --pool_size 10 --length 5 --top_k 1 \
    --prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
    --temperature 1.0 --laprompt_batchwise_prompt --laprompt_use_layer_embedding --freeze \
    --use_dynamic_logit_mask --use_forgetting_aware_init --logit_mask_temp 0.1
done
