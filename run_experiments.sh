#!/bin/bash
# AutoResearch 批量实验脚本

EXPERIMENTS=(
    "exp_mask_v1.sh:temp=2.0"
    "exp_mask_v2.sh:temp=0.5"
    "exp_mask_v3.sh:temp=2.0+CA"
    "exp_mask_v4.sh:temp=5.0"
    "exp_mask_v5.sh:temp=1.0"
    "exp_mask_v6.sh:temp=0.1"
    "exp_mask_v7.sh:temp=10.0"
    "exp_mask_v8.sh:baseline_CA"
)

LOG_DIR="results/logs"
mkdir -p $LOG_DIR

# 记录开始时间
echo "AutoResearch Started at $(date)" >> $LOG_DIR/autoresearch.log
echo "================================" >> $LOG_DIR/autoresearch.log

# 运行实验
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r script desc <<< "$exp"
    exp_name=$(basename $script .sh)
    
    echo "[$(date)] Starting $exp_name ($desc)..." >> $LOG_DIR/autoresearch.log
    
    # 等待直到有可用GPU槽位
    while true; do
        running=$(ps aux | grep "main.py.*laprompt" | grep -v grep | wc -l)
        if [ "$running" -lt 2 ]; then
            break
        fi
        sleep 30
    done
    
    # 启动实验
    bash scripts/$script > $LOG_DIR/${exp_name}.log 2>&1 &
    echo "[$(date)] Started $exp_name, PID: $!" >> $LOG_DIR/autoresearch.log
    
    sleep 5
done

echo "[$(date)] All experiments launched" >> $LOG_DIR/autoresearch.log
