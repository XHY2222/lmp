from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn

from .laprompt_pool_full import LaPromptPoolFull


class PreTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if prompt is not None:
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()
            key_prefix = prompt[0]
            value_prefix = prompt[1]
            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, dim)
        return self.proj(x)


class LaPromptViTFull(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        pretrained: bool,
        use_task_token: bool,
        max_tasks: int,
        pool_size: int,
        length: int,
        top_k: int,
        prompt_layer_idx: Optional[List[int]],
        temperature: float,
        ema_decay: float,
        use_self_attn: bool = False,
        batchwise_prompt: bool = True,
        use_layer_embedding: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.embed_dim = self.backbone.num_features
        self.use_task_token = use_task_token
        self.task_id_embedding = nn.Embedding(max_tasks, self.embed_dim)

        if not hasattr(self.backbone, "blocks") or len(self.backbone.blocks) == 0:
            raise ValueError("LaPrompt full backbone requires a ViT-like timm model with transformer blocks")
        base_num_heads = int(self.backbone.blocks[0].attn.num_heads)
        depth = len(self.backbone.blocks)

        self.prompt_layer_idx = list(range(depth)) if prompt_layer_idx is None else list(prompt_layer_idx)
        self.lapromptPool = LaPromptPoolFull(
            prompt_pool_size=pool_size,
            prompt_len=length,
            embed_dim=self.embed_dim,
            prompt_init="uniform",
            num_heads_pool=4,
            num_heads=base_num_heads,
            use_self_attn=use_self_attn,
            use_task_token=use_task_token,
            use_prefix_tuning=True,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            use_layer_embedding=use_layer_embedding,
            total_layers=depth,
            ema_decay=ema_decay,
            temperature=temperature,
        )

        for i, block in enumerate(self.backbone.blocks):
            old_attn = block.attn
            new_attn = PreTAttention(
                dim=self.embed_dim,
                num_heads=base_num_heads,
                qkv_bias=(old_attn.qkv.bias is not None),
            )
            with torch.no_grad():
                new_attn.qkv.weight.copy_(old_attn.qkv.weight)
                new_attn.proj.weight.copy_(old_attn.proj.weight)
                if old_attn.qkv.bias is not None:
                    new_attn.qkv.bias.copy_(old_attn.qkv.bias)
                if old_attn.proj.bias is not None:
                    new_attn.proj.bias.copy_(old_attn.proj.bias)
            self.backbone.blocks[i].attn = new_attn

    def _forward_block(self, block: nn.Module, x: torch.Tensor, prompt: Optional[torch.Tensor]) -> torch.Tensor:
        norm_x = block.norm1(x)
        attn_out = block.attn(norm_x, prompt)
        ls1 = getattr(block, "ls1", nn.Identity())
        dp1 = getattr(block, "drop_path1", getattr(block, "drop_path", nn.Identity()))
        x = x + dp1(ls1(attn_out))

        mlp_in = block.norm2(x)
        mlp_out = block.mlp(mlp_in)
        ls2 = getattr(block, "ls2", nn.Identity())
        dp2 = getattr(block, "drop_path2", getattr(block, "drop_path", nn.Identity()))
        x = x + dp2(ls2(mlp_out))
        return x

    def forward_features(self, x: torch.Tensor, taskids: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        x = self.backbone.patch_embed(x)

        if hasattr(self.backbone, "_pos_embed"):
            x = self.backbone._pos_embed(x)
        else:
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        for i, block in enumerate(self.backbone.blocks):
            prompt = None
            if i in self.prompt_layer_idx:
                cls_token = x[:, 0]
                task_token = None
                if self.use_task_token and taskids is not None:
                    task_tensor = torch.tensor(taskids, device=cls_token.device, dtype=torch.long)
                    task_token = self.task_id_embedding(task_tensor)
                prompt = self.lapromptPool(
                    cls_token,
                    layer_idx=i,
                    taskids=taskids,
                    task_token=task_token,
                )
            x = self._forward_block(block, x, prompt)

        x = self.backbone.norm(x)
        return {"x": x, "features": x[:, 0, :]}

    def forward_head(self, res: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = res["x"][:, 0]
        if hasattr(self.backbone, "fc_norm"):
            feat = self.backbone.fc_norm(feat)
        logits = self.backbone.head(feat)
        res["pre_logits"] = feat
        res["logits"] = logits
        return res

    def forward(
        self,
        x: torch.Tensor,
        taskids: Optional[List[int]] = None,
        fc_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if fc_only:
            return {"logits": self.backbone.head(x)}
        res = self.forward_features(x, taskids=taskids)
        return self.forward_head(res)
