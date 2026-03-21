#!/bin/bash
# 批量运行遗忘感知初始化实验

EXPERIMENTS=(
    "init_exp_v1.sh:var_perturb_001"
    "init_exp_v2.sh:var_perturb_01"
    "init_exp_v3.sh:var_perturb_0001"
    "init_exp_v4.sh:knn_001"
    "init_exp_v5.sh:gauss_05"
    "init_exp_v6.sh:ortho_1"
    "init_exp_v7.sh:mean_05"
    "init_exp_v8.sh:contrast_01"
    "init_exp_v9.sh:zero"
    "init_exp_v10.sh:rand_05"
    "init_exp_v11.sh:var_005"
    "init_exp_v12.sh:knn_01"
    "init_exp_v13.sh:ortho_05"
    "init_exp_v14.sh:mean_1"
    "init_exp_v15.sh:contrast_05"
    "init_exp_v16.sh:no_init"
)

LOG_DIR="results/logs"
mkdir -p $LOG_DIR

echo "[$(date)] Starting Init Strategy Experiments" | tee -a $LOG_DIR/init_research.log
echo "============================================" | tee -a $LOG_DIR/init_research.log

# 记录基准
BASELINE=0.4676
echo "Target: A_auc > $BASELINE (hard mask + CA)" | tee -a $LOG_DIR/init_research.log

# 运行实验
for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r script desc <<< "$exp_config"
    exp_name=$(basename $script .sh)
    
    echo "[$(date)] Starting $exp_name ($desc)..." | tee -a $LOG_DIR/init_research.log
    
    # 等待直到有可用槽位
    while true; do
        running=$(ps aux | grep "main.py.*laprompt" | grep -v grep | wc -l)
        if [ "$running" -lt 2 ]; then
            break
        fi
        sleep 30
    done
    
    # 启动实验
    bash scripts/$script > $LOG_DIR/${exp_name}.log 2>&1 &
    pid=$!
    echo "[$(date)] Started $exp_name, PID: $pid" | tee -a $LOG_DIR/init_research.log
    
    sleep 5
done

echo "[$(date)] All experiments launched" | tee -a $LOG_DIR/init_research.log
