from typing import List, Optional

import torch
import torch.nn as nn


class LaPromptPoolFull(nn.Module):
    def __init__(
        self,
        prompt_pool_size: int,
        prompt_len: int,
        embed_dim: int,
        prompt_init: str = "uniform",
        num_heads_pool: int = 4,
        num_heads: int = 12,
        use_self_attn: bool = False,
        use_task_token: bool = False,
        use_prefix_tuning: bool = True,
        top_k: int = 1,
        batchwise_prompt: bool = True,
        use_layer_embedding: bool = True,
        total_layers: int = 12,
        ema_decay: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.N = int(prompt_pool_size)
        self.L = int(prompt_len)
        self.D = int(embed_dim)
        self.use_prefix_tuning = bool(use_prefix_tuning)
        self.use_self_attn = bool(use_self_attn)
        self.use_task_token = bool(use_task_token)
        self.top_k = int(top_k)
        self.batchwise_prompt = bool(batchwise_prompt)
        self.use_layer_embedding = bool(use_layer_embedding)
        self.num_layers = int(total_layers)
        self.ema_decay = float(ema_decay)
        self.temperature = float(temperature)

        if self.N <= 0:
            raise ValueError("prompt_pool_size must be positive")
        if self.L <= 0:
            raise ValueError("prompt_len must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        input_dim = self.D
        if self.use_task_token:
            input_dim += self.D
        if self.use_layer_embedding:
            input_dim += self.D
        self.q_proj = nn.Linear(input_dim, self.D)

        if self.use_layer_embedding:
            self.layer_id_embedding = nn.Embedding(self.num_layers, self.D)

        if not self.use_prefix_tuning:
            raise NotImplementedError("LaPrompt full migration currently supports prefix prompts only")

        if self.D % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        prompt_pool_shape = (1, 2, self.N, self.L, num_heads, self.D // num_heads)
        if prompt_init == "zero":
            self.prompt_pool = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == "uniform":
            self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt_pool, -1, 1)
        else:
            raise ValueError(f"Unsupported prompt_init: {prompt_init}")

        self.attn_router = nn.MultiheadAttention(
            embed_dim=self.D,
            num_heads=num_heads_pool,
            batch_first=True,
        )
        self._init_attn_ema()

        if self.use_self_attn:
            self.self_attn_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.D,
                    nhead=4,
                    batch_first=True,
                ),
                num_layers=1,
            )

    def _init_attn_ema(self):
        if self.ema_decay <= 0:
            return
        for name, param in self.attn_router.named_parameters():
            if not param.requires_grad:
                continue
            safe_name = name.replace(".", "_")
            self.register_buffer(f"ema_{safe_name}", param.data.clone())

    @torch.no_grad()
    def update_attn_ema(self):
        if self.ema_decay <= 0:
            return
        for name, param in self.attn_router.named_parameters():
            if not param.requires_grad:
                continue
            safe_name = name.replace(".", "_")
            ema_param = getattr(self, f"ema_{safe_name}")
            ema_param.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    @torch.no_grad()
    def use_attn_ema_params(self):
        if self.ema_decay <= 0:
            return
        for name, param in self.attn_router.named_parameters():
            if not param.requires_grad:
                continue
            safe_name = name.replace(".", "_")
            ema_param = getattr(self, f"ema_{safe_name}")
            param.data.copy_(ema_param)

    def _apply_self_attn(self, prompt: torch.Tensor) -> torch.Tensor:
        if not self.use_self_attn:
            return prompt
        bsz, dual, plen, heads, dim_per_head = prompt.shape
        flat = (
            prompt.permute(0, 1, 3, 4, 2)
            .reshape(bsz * dual, heads * dim_per_head, plen)
            .transpose(1, 2)
        )
        flat = self.self_attn_layer(flat)
        return (
            flat.transpose(1, 2)
            .reshape(bsz, dual, heads, dim_per_head, plen)
            .permute(0, 1, 4, 2, 3)
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        layer_idx: int = 0,
        taskids: Optional[List[int]] = None,
        task_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = cls_token.size(0)
        prompts = self.prompt_pool.expand(bsz, -1, -1, -1, -1, -1)
        prompts = prompts.permute(0, 2, 1, 3, 4, 5)

        heads = prompts.shape[-2]
        dim_per_head = prompts.shape[-1]
        prompt_key = prompts.reshape(bsz, self.N, 2 * self.L, heads * dim_per_head).mean(dim=2)

        query = cls_token.unsqueeze(1)
        extra_tokens = []
        if self.use_layer_embedding:
            lid = torch.tensor(layer_idx, device=query.device, dtype=torch.long)
            layer_embed = self.layer_id_embedding(lid).unsqueeze(0).expand(bsz, -1).unsqueeze(1)
            extra_tokens.append(layer_embed)
        if self.use_task_token:
            if task_token is None:
                if taskids is None:
                    task_token = torch.zeros(bsz, self.D, device=query.device)
                else:
                    task_token = torch.zeros(bsz, self.D, device=query.device)
            extra_tokens.append(task_token.unsqueeze(1))
        if extra_tokens:
            query = torch.cat([query] + extra_tokens, dim=-1)
        query = self.q_proj(query)

        _, attn_weights = self.attn_router(
            query=query,
            key=prompt_key,
            value=prompt_key,
            need_weights=True,
        )
        attn_weights = attn_weights.squeeze(1)
        if self.temperature > 0:
            attn_weights = attn_weights / self.temperature

        if self.top_k == 1:
            weights = torch.softmax(attn_weights, dim=-1)
            weights = weights.view(bsz, self.N, 1, 1, 1, 1)
            fused_prompt = (prompts * weights).sum(dim=1)
            return self._apply_self_attn(fused_prompt)

        top_k = min(self.top_k, self.N)
        _, idx = torch.topk(attn_weights, top_k, dim=1)
        if self.batchwise_prompt:
            prompt_id, counts = torch.unique(idx, return_counts=True, sorted=True)
            if prompt_id.numel() < self.N:
                pad = self.N - prompt_id.numel()
                prompt_id = torch.cat(
                    [
                        prompt_id,
                        torch.full((pad,), int(torch.min(idx)), device=prompt_id.device),
                    ]
                )
                counts = torch.cat([counts, torch.zeros(pad, device=counts.device, dtype=counts.dtype)])
            _, major = torch.topk(counts, k=top_k)
            idx = prompt_id[major].expand(bsz, -1)

        gather_idx = idx.view(bsz, top_k, 1, 1, 1, 1).expand(-1, -1, 2, self.L, heads, dim_per_head)
        selected = prompts.gather(dim=1, index=gather_idx)
        selected = selected.permute(0, 2, 1, 3, 4, 5).reshape(bsz, 2, top_k * self.L, heads, dim_per_head)
        return self._apply_self_attn(selected)
