#!/bin/bash
# 实验7: 温度=10.0 (非常温和的抑制)
NOTE="EXP_MASK_v7_temp10"
PYTHON=$UV_INTERNAL__PARENT_INTERPRETER
$PYTHON main.py --mode laprompt --dataset imagenet-r --n_tasks 5 --m 10 --n 50 --rnd_seed 1 \
    --model_name laprompt --opt_name adam --sched_name default --lr 5e-3 --batchsize 64 \
    --memory_size 0 --gpu_transform --online_iter 3 --data_dir /root/autodl-tmp/dataset \
    --note $NOTE --eval_period 1000 --n_worker 4 --rnd_NM \
    --laprompt_backbone_name vit_base_patch16_224 --laprompt_pretrained \
    --laprompt_use_task_token --laprompt_max_tasks 16 --pool_size 10 --length 5 --top_k 1 \
    --prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 --temperature 1.0 \
    --laprompt_batchwise_prompt --laprompt_use_layer_embedding --freeze \
    --use_dynamic_logit_mask --logit_mask_temp 10.0 --no-use_forgetting_aware_init
