# LaPrompt Full Migration Notes

This document records the full LaPrompt migration into Si-Blurry while preserving the Si-Blurry online training/evaluation protocol.

## Scope

- Migrated: full LaPrompt model algorithm components
  - Prefix prompt injection into attention
  - Prompt routing pool with `top_k`, batchwise routing, temperature scaling
  - Layer embedding and task token support
  - Prompt router EMA update/use
  - CA storage branches: `covariance`, `variance`, `multi-centroid`
- Preserved as-is from Si-Blurry:
  - Online sampler and stream semantics
  - Main training loop and summary protocol
  - `online_evaluate` output keys (`avg_loss`, `avg_acc`, `cls_acc`)

## Main File Changes

- Added `models/laprompt_pool_full.py`
- Added `models/laprompt_vit_full.py`
- Updated `models/laprompt.py`
- Updated `models/__init__.py` (model factory eager-instantiation fix)
- Updated `methods/laprompt.py`
- Updated `methods/_trainer.py`
- Updated `configuration/config.py`
- Updated `scripts/laprompt.sh` (aligned with `scripts/mvp.sh` structure)

## Parameter Mapping

Primary LaPrompt flags:

- `--laprompt_backbone_name`
- `--laprompt_pretrained`
- `--laprompt_use_task_token`
- `--laprompt_max_tasks`
- `--pool_size`
- `--length`
- `--top_k`
- `--prompt_layer_idx`
- `--temperature`
- `--laprompt_ema_decay`
- `--laprompt_use_ca`
- `--laprompt_ca_lr`
- `--laprompt_ca_epochs`
- `--laprompt_ca_storage`
- `--laprompt_n_centroids`
- `--laprompt_add_num`
- `--freeze` / `--no-freeze`

Compatibility aliases retained:

- `--tuned_epoch`
- `--ca_lr`
- `--crct_epochs`
- `--ca_storage_efficient_method`
- `--n_centroids`
- `--add_num`
- `--ema_decay`

## Verification Commands

Executed:

```bash
python main.py --help
python -m py_compile models/laprompt.py models/laprompt_vit_full.py models/laprompt_pool_full.py methods/laprompt.py configuration/config.py
python -c "import torch; from models.laprompt import LaPrompt; m=LaPrompt(num_classes=10, backbone_name='vit_base_patch16_224', pretrained=False, pool_size=4, length=2, top_k=2, prompt_layer_idx=[0,1]); x=torch.randn(2,3,224,224); o=m(x, taskids=[0,0]); print(o['logits'].shape, o['features'].shape)"
```

Observed:

- CLI help includes all new LaPrompt arguments.
- Modified modules compile successfully.
- Model forward smoke returns expected tensor shapes.

Partial runtime smoke (long run, interrupted by timeout):

```bash
python main.py --mode laprompt --dataset imagenet-r --n_tasks 5 --n 50 --m 10 --rnd_seed 1 --model_name laprompt --opt_name adam --sched_name default --lr 1e-3 --batchsize 2 --memory_size 0 --online_iter 1 --data_dir /root/autodl-tmp/dataset --note laprompt_full_smoke --eval_period 200 --n_worker 0 --rnd_NM --debug --laprompt_backbone_name vit_base_patch16_224 --laprompt_use_task_token --laprompt_max_tasks 8 --pool_size 4 --length 2 --top_k 1 --prompt_layer_idx 0
```

- Training entered Si-Blurry online loop and produced normal train/test logs.
- Full CA branch completion on ImageNet-R is pending long-running validation.
