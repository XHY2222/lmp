#!/bin/bash
# AutoResearch 自动化实验循环
# 持续运行实验，记录结果，自动调整策略

LOG_DIR="results/logs"
RESULTS_FILE="RESEARCH_LOG.md"
BEST_AUC=0.4303  # 当前最佳
BASELINE=0.4276

# 实验配置数组
EXPERIMENTS=(
    "exp_mask_v9.sh:乘法mask_temp1"
    "exp_mask_v10.sh:乘法mask_temp2"
    "exp_mask_v11.sh:乘法mask_temp05"
)

echo "[$(date)] AutoResearch Started" >> $LOG_DIR/autoresearch.log
echo "Target: A_auc > $BEST_AUC" >> $LOG_DIR/autoresearch.log

# 运行实验
for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r script desc <<< "$exp_config"
    exp_name=$(basename $script .sh)
    
    echo "[$(date)] Starting $exp_name ($desc)..." | tee -a $LOG_DIR/autoresearch.log
    
    # 运行实验
    bash scripts/$script > $LOG_DIR/${exp_name}.log 2>&1
    
    # 提取结果
    result_dir="results/logs/imagenet-r/${exp_name}"
    if [ -f "$result_dir/seed_1_summary.log" ]; then
        auc=$(grep "A_auc" "$result_dir/seed_1_summary.log" | awk '{print $2}')
        echo "[$(date)] $exp_name Result: A_auc = $auc" | tee -a $LOG_DIR/autoresearch.log
        
        # 检查是否提升
        if (( $(echo "$auc > $BEST_AUC" | bc -l) )); then
            echo "[$(date)] 🎉 NEW BEST! A_auc improved from $BEST_AUC to $auc" | tee -a $LOG_DIR/autoresearch.log
            BEST_AUC=$auc
            
            # 提交GitHub
            git add -A
            git commit -m "feat: New best A_auc=$auc with $desc"
            git push lmp feat/laprompt-full-migration
        fi
    else
        echo "[$(date)] $exp_name failed or not finished" | tee -a $LOG_DIR/autoresearch.log
    fi
    
    # 清理GPU内存
    sleep 10
done

echo "[$(date)] Batch completed. Best A_auc: $BEST_AUC" >> $LOG_DIR/autoresearch.log
