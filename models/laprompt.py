from typing import List, Optional

import torch
import torch.nn as nn

from .laprompt_vit_full import LaPromptViTFull


class LaPrompt(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        use_task_token: bool = True,
        max_tasks: int = 100,
        pool_size: int = 10,
        length: int = 5,
        top_k: int = 1,
        prompt_layer_idx: Optional[List[int]] = None,
        temperature: float = 1.0,
        ema_decay: float = 0.0,
        use_self_attn: bool = False,
        batchwise_prompt: bool = True,
        use_layer_embedding: bool = True,
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.backbone = LaPromptViTFull(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=pretrained,
            use_task_token=use_task_token,
            max_tasks=max_tasks,
            pool_size=pool_size,
            length=length,
            top_k=top_k,
            prompt_layer_idx=prompt_layer_idx,
            temperature=temperature,
            ema_decay=ema_decay,
            use_self_attn=use_self_attn,
            batchwise_prompt=batchwise_prompt,
            use_layer_embedding=use_layer_embedding,
        )

        if freeze:
            freeze_prefix = (
                "backbone.patch_embed",
                "backbone.pos_embed",
                "backbone.cls_token",
                "backbone.blocks",
                "backbone.norm",
            )
            for name, param in self.backbone.named_parameters():
                if name.startswith(freeze_prefix):
                    param.requires_grad = False

    def forward_features(
        self,
        inputs: torch.Tensor,
        taskids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        output = self.backbone.forward_features(inputs, taskids=taskids)
        return output["features"]

    def forward(
        self,
        x: torch.Tensor,
        taskids: Optional[List[int]] = None,
        train: bool = False,
        fc_only: bool = False,
    ):
        del train
        output = self.backbone(x, taskids=taskids, fc_only=fc_only)
        if fc_only:
            return {"logits": output["logits"]}
        return {"features": output["features"], "logits": output["logits"]}
