# The codes in this directory were from https://github.com/drimpossible/GDumb/tree/master/src/models
import timm
from .dualprompt import DualPrompt
from .l2p import L2P
from .mvp import MVP
from .laprompt import LaPrompt

__all__ = [
    "DualPrompt",
    "l2p",
    "vision_trainsfomer",
    "LaPrompt",
]

def get_model(name, **kwargs):
    name = name.lower()
    try:
        if 'vit' in name:
            name = name.split('_')[1]
            model = timm.create_model(f'vit_{name}_patch16_224', pretrained=True, **kwargs)
            if '_ft' not in name:
                for name, param in model.named_parameters():
                    if 'head' not in name:
                        param.requires_grad = False
            return (model, 224)
        elif 'resnet' in name:
            model = timm.create_model(name, pretrained=True, **kwargs)
            if '_ft' not in name:
                for name, param in model.named_parameters():
                    if 'head' not in name:
                        param.requires_grad = False
            return (model, 224)
        else:
            if name == "dualprompt":
                return (DualPrompt(**kwargs), 224)
            if name == "l2p":
                return (L2P(**kwargs), 224)
            if name == "mvp":
                return (MVP(**kwargs), 224)
            if name == "laprompt":
                return (LaPrompt(**kwargs), 224)
            raise KeyError(name)
    except KeyError:
        raise NotImplementedError(f"Model {name} not implemented")
