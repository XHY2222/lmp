# LaPrompt 集成分析（Si-Blurry）

本文档总结了当前 Si-Blurry 中 LaPrompt 的整合情况，重点回答三个问题：

- 是否使用了 buffer；
- LaPrompt 一共有几个模块；
- 各模块是如何实现的。

## 1）Buffer 分析

### 1.1 当前整合后的 LaPrompt 是否使用了 buffer？

使用了，但属于“按条件启用”。

- 位置：`models/laprompt_pool_full.py`
- 类：`LaPromptPoolFull`
- 方法：`_init_attn_ema()`
- 机制：`register_buffer(f"ema_{safe_name}", param.data.clone())`

只有在满足以下条件时，才会创建 `ema_*` buffer：

- `ema_decay > 0`

如果 `ema_decay == 0`（当前默认值），则不会注册 EMA buffer。

### 1.2 Buffer 在训练/评估中的运行行为

- 训练阶段更新 EMA：`update_attn_ema()`
- 评估阶段切换到 EMA 参数：`use_attn_ema_params()`

训练器中的调用位置：

- 训练步：`methods/laprompt.py` 在 `laprompt_ema_decay > 0` 时调用 `lapromptPool.update_attn_ema()`
- 评估步：`methods/laprompt.py` 在 `laprompt_ema_decay > 0` 时调用 `lapromptPool.use_attn_ema_params()`

结论：buffer 机制已经完整接入，但默认配置下（EMA 关闭）不会生效。

## 2）LaPrompt 一共有几个模块？

“模块数”有几种合理口径，这里统一说明。

### 2.1 按核心模型模块统计（推荐口径）：3 个

1. `LaPrompt`（外层封装）：`models/laprompt.py`
2. `LaPromptViTFull`（ViT + prompt 注入）：`models/laprompt_vit_full.py`
3. `LaPromptPoolFull`（prompt 路由/池化）：`models/laprompt_pool_full.py`

### 2.2 若把内部注意力组件也算作独立模块：4 个

额外增加：

4. `PreTAttention`（位于 `models/laprompt_vit_full.py`）

该模块负责 prefix-tuning 形式的注意力计算，并被注入每个 transformer block。

### 2.3 若把训练算法模块也算上：5 个

再增加：

5. `methods/laprompt.py` 中的 `LaPrompt` trainer

该模块负责在线训练流程、任务切换、CA（分类器对齐）与评估协议。

## 3）各模块实现说明

### 3.1 `models/laprompt.py`（`LaPrompt`）

职责：

- 作为薄封装层，保证与 Si-Blurry 现有模型接口一致。
- 暴露统一接口：
  - `forward_features(inputs, taskids)`
  - `forward(x, taskids, fc_only)`

实现要点：

- 内部实例化 `LaPromptViTFull`。
- 当 `freeze=True` 时冻结主干参数（保留 prompt/router/head 等可训练部分）。
- 输出保持 `{features, logits}` 字典格式，兼容现有 trainer。

### 3.2 `models/laprompt_vit_full.py`（`LaPromptViTFull` + `PreTAttention`）

职责：

- 作为视觉主干整合层，完成 ViT 与 LaPrompt 的融合。

实现要点：

- 使用 `timm.create_model(...)` 创建 ViT 主干。
- 将每个 block 的 attention 替换为 `PreTAttention`，并拷贝原始 qkv/proj 权重。
- 在指定层（`prompt_layer_idx`）调用 `LaPromptPoolFull` 生成 prompt，并注入 attention 的 K/V。
- `forward_features` 返回 token 级输出与 cls 特征，供分类头与统计分支使用。

### 3.3 `models/laprompt_pool_full.py`（`LaPromptPoolFull`）

职责：

- 承担 prompt 路由与 prompt 选择核心逻辑。

实现要点：

- 维护 prefix 形式的 prompt pool 参数。
- 路由 query 由以下信息拼接构成：
  - cls token；
  - 可选 task token；
  - 可选 layer embedding。
- 通过 `nn.MultiheadAttention` 计算路由权重并选择 prompt。
- 支持：
  - `top_k=1` 软融合；
  - `top_k>1` 多 prompt 选择；
  - batchwise 共享 prompt 选择；
  - 可选 prompt self-attention 精炼；
  - 可选 EMA buffer（路由器参数）。

### 3.4 `methods/laprompt.py`（`LaPrompt` Trainer）

职责：

- 实现 Si-Blurry 协议下的在线类增量训练流程。

实现要点：

- 实现在线更新主流程：`online_step`、`online_train`。
- 保持 exposed-class 重映射与 mask 机制一致。
- 在训练中对新任务样本执行旧类屏蔽逻辑（防止不合理竞争）。
- 实现 CA 统计与对齐，支持三种存储方式：
  - `variance`
  - `covariance`
  - `multi-centroid`
- 评估阶段采用“两段式 task-id 推断”：
  - 第一段先预测类别；
  - 通过 `cls2task` 映射到任务 ID；
  - 第二段按样本 task id 再前向评估。

## 4）当前训练配置（与问题修复相关）

- `scripts/laprompt.sh` 已启用 `--laprompt_pretrained` 与 `--freeze`。
- 这与预期一致：冻结“预训练 backbone”，而不是冻结“随机初始化 backbone”。

## 5）结论

- 当前代码中 buffer 机制是存在且正确的，但仅在 `laprompt_ema_decay > 0` 时启用。
- LaPrompt 的模块数可按口径理解为：
  - 3（核心模型）
  - 4（含 `PreTAttention`）
  - 5（再含 trainer）
- 集成链路完整：`wrapper -> ViT 注入 -> prompt 路由 -> 在线训练/CA/评估`。
