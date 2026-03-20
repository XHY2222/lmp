# AGENTS.md

Guidance for agentic coding assistants working in this repository.
This project is a PyTorch research codebase for online class-incremental learning (Si-Blurry, MVP, L2P, DualPrompt, ER, CLIB).

## 1) Repository layout

- `main.py`: program entrypoint; parses CLI args and dispatches trainer by `--mode`.
- `configuration/config.py`: central `argparse` definition for all run options.
- `methods/`: trainer implementations (`ER`, `CLIB`, `MVP`, etc.) built on `methods/_trainer.py`.
- `models/`: model factory (`models/__init__.py`) and prompt-based models.
- `datasets/`: dataset registry + wrappers.
- `utils/`: optimizers, schedulers, memory, samplers, augmentation, metrics.
- `scripts/*.sh`: canonical experiment launch templates.

## 2) Environment and setup

- Primary environment spec is `requirements.yaml` (Conda env with CUDA/PyTorch stack).
- README references `environment.yml`, but this repository currently contains `requirements.yaml`.
- Typical setup:

```bash
conda env create -f requirements.yaml
conda activate Si-blurry
```

- If editing dependency docs, keep README and actual filename consistent.

## 3) Build / run / lint / test commands

This repo is script-driven and does not currently ship a formal build system, lint config, or test suite.
Use the commands below as operational defaults.

### Run training (main command)

```bash
python main.py --help
```

```bash
python main.py \
  --mode er \
  --dataset cifar100 \
  --n_tasks 5 --n 50 --m 10 \
  --rnd_seed 1 \
  --model_name vit_base \
  --opt_name adam --sched_name default \
  --lr 5e-3 --batchsize 64 \
  --memory_size 2000 \
  --online_iter 3 \
  --data_dir /local_datasets \
  --note debug_run \
  --eval_period 1000 \
  --n_worker 4 --rnd_NM
```

### ImageNet-R run (dataset already downloaded)

If `imagenet-r.tar` already exists at `autodl-tmp/dataset/imagenet-r.tar`, run with:

```bash
python main.py \
  --mode er \
  --dataset imagenet-r \
  --n_tasks 5 --n 50 --m 10 \
  --rnd_seed 1 \
  --model_name vit_base \
  --opt_name adam --sched_name default \
  --lr 5e-3 --batchsize 64 \
  --memory_size 2000 \
  --online_iter 3 \
  --data_dir /autodl-tmp/dataset \
  --note er_imagenetr_seed1 \
  --eval_period 1000 \
  --n_worker 4 --rnd_NM
```

Notes:
- `--data_dir` should point to the parent folder that contains `imagenet-r.tar`.
- `datasets/Imagenet_R.py` auto-extracts `imagenet-r.tar` into `<data_dir>/imagenet-r/` when missing.
- If your environment uses `/root/autodl-tmp/dataset`, set `--data_dir /root/autodl-tmp/dataset` instead.

### Run packaged experiment scripts

```bash
bash scripts/er.sh
bash scripts/clib.sh
bash scripts/l2p.sh
bash scripts/dualprompt.sh
bash scripts/mvp.sh
```

Note: these scripts loop over seeds `1..5`; for quick validation, prefer invoking `python main.py ...` directly for one seed.

### Lint / format

- No repository-local formatter/linter config (`pyproject.toml`, `ruff.toml`, `setup.cfg`, etc.) is present.
- Safe optional local checks (only if tools are installed in your environment):

```bash
python -m ruff check .
python -m black --check .
```

- If you auto-format, keep diffs focused and avoid broad unrelated reformatting.

### Tests

- There is no committed `tests/` suite at this time.
- When adding tests, use `pytest` conventions.

Run all tests (if/when tests are added):

```bash
python -m pytest
```

Run a single test file:

```bash
python -m pytest tests/test_memory.py
```

Run a single test case:

```bash
python -m pytest tests/test_memory.py::test_reservoir_replacement
```

Run tests by keyword:

```bash
python -m pytest -k "mvp and not slow"
```

## 4) Coding style and conventions

Follow existing patterns in the codebase unless a file already uses a different local style.

### Python version and general style

- Target Python 3.10 behavior (per env file).
- Use 4-space indentation.
- Keep lines readable; avoid deeply nested logic when a helper function is clearer.
- Prefer explicit, small helper methods over very long monolithic methods.

### Imports

- Order imports: standard library -> third-party -> local project modules.
- Separate groups with one blank line.
- Prefer explicit imports over wildcard imports in new code.
  - Existing wildcard usage (e.g., `from datasets import *`) is legacy; do not expand this pattern.
- Avoid duplicate imports (e.g., repeated `import timm`).

### Naming

- `snake_case`: functions, variables, module-level helpers.
- `CamelCase`: classes (`ER`, `MVP`, dataset classes).
- `UPPER_SNAKE_CASE`: constants.
- CLI flags should remain consistent with existing names in `configuration/config.py`.

### Types and signatures

- Add type hints for new public functions/methods where practical.
- Keep tensor-heavy internals lightweight: annotate critical interfaces, avoid noisy over-annotation.
- Match existing return styles (e.g., tuple returns in model factory) unless refactor scope is explicit.

### Formatting and structure

- Keep argument lists vertically aligned when long (as done in parser and scripts).
- Maintain existing comment style: short, purposeful comments for non-obvious behavior.
- Do not add boilerplate comments that restate the code.

### Error handling

- Fail fast on unsupported options using explicit exceptions:
  - `ValueError` for invalid parameter values.
  - `NotImplementedError` for unsupported modes/models/optimizers.
- Preserve existing guard style in scripts (`Undefined setting` + exit).
- Avoid silent fallback that can hide research mistakes.

### Logging and output

- Existing code uses `print` for progress in training loops; keep output concise and informative.
- If adding logs, prefer consistency with existing report methods in `_Trainer`.
- Avoid verbose per-step debug prints unless gated by a debug flag.

## 5) Data processing pipeline

When changing data logic, preserve this flow unless the task explicitly requires a redesign.

### Dataset selection and metadata

- Dataset dispatch is centralized in `datasets/__init__.py` via `get_dataset(name)`.
- Each dataset entry maps to `(dataset_class, mean, std, num_classes)`.
- Keep dataset keys stable (e.g., `cifar100`, `tinyimagenet`, `imagenet-r`) because scripts and CLI depend on them.

### Base dataset construction

- `_Trainer.setup_dataset()` builds:
  - `train_dataset = dataset(..., train=True, transform=transforms.ToTensor())`
  - `test_dataset = dataset(..., train=False, transform=self.test_transform)`
- `train_dataset` is wrapped as `IndexedDataset` so each sample yields `(image, label, idx)` for memory updates.
- Do not remove sample indices from training data path; replay/memory replacement depends on them.

### Transform pipeline

- `_Trainer.setup_transforms()` composes optional train augmentations from `--transforms`:
  - `autoaug` (dataset-specific AutoAugment policy)
  - `cutout`
  - `randaug`
- After optional ops, train transform applies resize, random crop, horizontal flip, and normalization.
- Test transform applies resize, `ToTensor`, and normalization only.
- Keep train/test normalization statistics aligned with dataset registry (`mean`, `std`).

### Online sampler and task stream

- Training uses `OnlineSampler(train_dataset, n_tasks, m, n, rnd_seed, rnd_NM, selection_size)`.
- Task boundary progression is controlled by `train_sampler.set_task(task_id)`.
- Evaluation on seen classes uses `OnlineTestSampler(test_dataset, exposed_classes)`.
- Preserve behavior of `--n` / `--m` / `--rnd_NM`; these define the Si-Blurry stream semantics.

### Label remapping and masking

- Methods remap raw labels to current exposed-class indices (see `ER.online_step`, `ER.online_evaluate`).
- `self.exposed_classes` order is the canonical class order during training/eval.
- `self.mask` in trainer/model is used to block unseen classes; avoid bypassing this in logits.
- Any new method must keep remapping + masking consistent between train and eval.

### Memory/replay data handling

- Replay methods use `MemoryBatchSampler` with `self.memory_batchsize` and online iteration settings.
- Memory updates are index-driven and may be synchronized across distributed workers.
- In distributed mode, keep gather/broadcast logic intact before replacing memory entries.
- Ensure new replay logic remains deterministic under fixed `--rnd_seed`.

## 6) Domain-specific implementation guidance

- Respect exposed-class remapping logic used in online learning methods.
- Keep memory update semantics deterministic relative to existing random-seed handling.
- For distributed changes, preserve rank/world-size checks and main-process-only side effects.
- Do not break saved result paths under `results/logs/<dataset>/<note>/...`.
- Maintain compatibility with existing script flags (`--use_mask`, `--use_contrastiv`, etc.).

## 7) Change-scope rules for agents

- Keep patches minimal and task-focused.
- Do not perform broad refactors unless requested.
- Avoid introducing new heavy dependencies for small tasks.
- If touching CLI arguments, update both parser defaults/help and any affected scripts/docs.
- If adding tests, keep them fast and deterministic; mark expensive integration tests clearly.

## 8) Validation checklist before finishing

- `python main.py --help` works.
- Modified module imports resolve.
- For training-path edits, run at least one short smoke command with one seed.
- If tests exist for touched code, run the most targeted test first, then broader tests.
- Summarize any unverified steps in your final handoff.

## 9) Cursor / Copilot rule files

Checked for additional agent rules:

- `.cursor/rules/`: not present
- `.cursorrules`: not present
- `.github/copilot-instructions.md`: not present

If these files are added later, treat them as higher-priority repository instructions and merge their guidance into this document.
