# Evosci
```Bash
cd /root/autodl-tmp/laprompt-misa/EvoScientist
python3 -m uv run evosci
```

## Run

### MVP
```Bash
bash scripts/mvp.sh
```

### LaPrompt
```Bash
bash scripts/laprompt.sh
```

LaPrompt integration in this repository includes:

- Prefix prompt injection on ViT attention blocks.
- Prompt pool routing with `top_k`, batchwise selection, layer embedding, and optional task token.
- Optional EMA for prompt router attention (`--laprompt_ema_decay`).
- Optional classifier alignment stage (`--laprompt_use_ca`) with `variance` / `covariance` / `multi-centroid` statistics.

Recommended default behavior for LaPrompt:

- Use pretrained backbone (`--laprompt_pretrained`, enabled by default).
- Freeze ViT backbone for prompt tuning (`--freeze`, enabled by default).

## Si-Blurry
<center><img src="./img/Si-Blurry.png" width="450"></center>

it is possible to adjust the disjoint class ratio with N, the blurry class ratio with M in scripts. 
```Bash
   N=50
   M=10
```
The online sampler assumes only the samples in each class are the same number. (Like CIFAR100, TinyImageNet...)
## MVP
<center><img src="./img/MVP.png" width="1000"></center>

## LaPrompt

Main LaPrompt files:

- Model wrapper: `models/laprompt.py`
- ViT integration: `models/laprompt_vit_full.py`
- Prompt pool/router: `models/laprompt_pool_full.py`
- Online trainer: `methods/laprompt.py`

Detailed design notes are in:

- `docs/laprompt_full_migration.md`
- `docs/laprompt_module_analysis.md`

## Citation
If you find this code useful for your research, please cite our paper:
```
@inproceedings{moon2023online,
  title={Online Class Incremental Learning on Stochastic Blurry Task Boundary via Mask and Visual Prompt Tuning},
  author={Moon, Jun-Yeong and Park, Keon-Hee and Kim, Jung Uk and Park, Gyeong-Moon},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
