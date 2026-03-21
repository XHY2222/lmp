# LaPrompt AutoResearch 实验日志

## 目标
持续探索掩码策略和遗忘感知初始化，提升A_auc指标

## 基准
- A_auc: 0.4276 (原始LaPrompt)
- A_auc: 0.4303 (启用CA)

## 实验记录

### 实验1: 简化动态掩码 - temp=2.0
**假设**: 复杂的动态掩码干扰学习，简化版本可能有效
**改动**: 移除置信度引导掩码，仅保留基础soft suppression
**参数**: --use_dynamic_logit_mask=True --logit_mask_temp=2.0
**结果**: A_auc = 0.3868 ❌ (比基准低)
**结论**: 简化版soft suppression仍不如原始hard mask

### 实验2: 简化动态掩码 - temp=0.5
**假设**: 更激进的抑制可能有效
**改动**: 温度=0.5，更激进的soft suppression
**参数**: --use_dynamic_logit_mask=True --logit_mask_temp=0.5
**结果**: A_auc = 0.3054 ❌ (更差)
**结论**: 温度越低效果越差

### 实验3: 动态掩码 + CA
**参数**: --use_dynamic_logit_mask=True --logit_mask_temp=2.0 --laprompt_use_ca
**结果**: OOM ❌
**结论**: CA + 动态掩码内存不足

### 实验4-7: temp=5.0, 1.0, 0.1, 10.0
**结果**: OOM ❌
**结论**: 并行实验导致OOM

### 实验9: 乘法mask temp=1.0
**结果**: A_auc = 0.3868 ❌
**结论**: 乘法mask效果不佳

### 实验10: 乘法mask temp=2.0
**结果**: A_auc = 0.3704 ❌
**结论**: 温度越高效果越差

---

## 第2轮实验结果

### 实验12: 自适应mask - 基于样本难度
**结果**: A_auc = 0.3436 ❌
**结论**: 自适应mask效果不佳

### 实验13: Hard mask + CA (对照)
**结果**: A_auc = 0.4676 ✅ **重大突破！**
**结论**: Hard mask + CA 提升 +9.3%

### 实验14: CA参数调优 - lr=0.005
**结果**: 待完成

### 实验15: CA参数调优 - epochs=10
**结果**: OOM ❌

---

## 第3轮实验: 遗忘感知初始化探索 (进行中)

### 实验目标
系统性地探索8种不同的初始化策略，每种策略测试不同的scale参数

### 初始化策略列表
1. variance_perturb - 基于历史方差的扰动
2. knn_prototype - KNN原型初始化
3. gaussian_sample - 高斯采样初始化
4. task_orthogonal - 任务正交初始化
5. mean_centered - 均值中心化初始化
6. contrastive_init - 对比学习风格初始化
7. zero_init - 零初始化（对照）
8. random_scaled - 随机缩放初始化

### 当前状态
- v1-v3 (variance_perturb): 完成，结果=0.4276（无影响）
- v4-v10: 运行中
- v11-v16: 刚刚启动

**注意**: 初步结果显示初始化可能未正确执行，需要进一步调试。
