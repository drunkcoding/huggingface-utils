import copy
from turtle import forward
from typing import Tuple
import numpy as np
import torch
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTPooler
from transformers import ViTForImageClassification

from torch import nn

from hfutils.pipe.base import PipeMethods, get_num_layers


class ViTEmbeddingsPipe(ViTEmbeddings):
    def __init__(self, config: ViTConfig, ds=False):
        super().__init__(config)
        self.deepspeed_enabled = ds

    def forward(self, pixel_values):
        hidden_states = super().forward(pixel_values)
        return hidden_states

class ViTLayerPipe(ViTLayer):
    def __init__(self, config: ViTConfig, ds=False):
        super().__init__(config)
        self.deepspeed_enabled = ds

    def forward(self, hidden_states):
        layer_outputs = super().forward(hidden_states)
        hidden_states = layer_outputs[0]
        return hidden_states

# No pooler needed

class ViTClassifierPipe(nn.Module):
    def __init__(self, config: ViTConfig, ds=False):
        super().__init__()
        self.deepspeed_enabled = ds
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)    
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, hidden_states):
        hidden_states = self.layernorm(hidden_states)
        logits = self.classifier(hidden_states[:, 0, :])
        return logits

class ViTPyTorchPipeForImageClassification(nn.Module, PipeMethods):
    def __init__(self, model: ViTForImageClassification, exec_map: Tuple = None) -> None:
        super().__init__()

        config = model.config
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(encoder_config)

        self.layers = []
        encoder_embed = ViTEmbeddingsPipe(encoder_config)
        encoder_embed.load_state_dict(model.vit.embeddings.state_dict())
        self.layers.append(encoder_embed)

        for i in range(self.n_layers):
            encoder_block = ViTLayerPipe(encoder_config)
            encoder_block.load_state_dict(model.vit.encoder.layer[i].state_dict())
            self.layers.append(encoder_block)

        classifier = ViTClassifierPipe(encoder_config)
        classifier.classifier.load_state_dict(model.classifier.state_dict())
        classifier.layernorm.load_state_dict(model.vit.layernorm.state_dict())
        self.layers.append(classifier)

        self.total_params = sum([
            sum([np.prod(p.size()) for p in layer.parameters()])
            for layer in self.layers
        ])

        self.layers = nn.ModuleList(self.layers)

        self.exec_map = exec_map if exec_map is not None else (0, len(self.layers))

    def forward(self, args, output_hidden_states=False):
        outputs = args
        all_hidden_states = ()
        for idx in range(*self.exec_map):
            outputs = self.layers[idx](outputs)
            if output_hidden_states:
                if idx != len(self.layers) - 1:
                    all_hidden_states = all_hidden_states + (outputs,)
        if output_hidden_states:
            return (
                outputs,
                all_hidden_states
            )
        return outputs # if isinstance(outputs, Tuple) else (outputs, )


VIT_INPUTS = {
    ViTEmbeddingsPipe.__name__: ["pixel_values"],
    ViTLayerPipe.__name__: ["hidden_states"],
    ViTClassifierPipe.__name__: ["hidden_states"],
}

VIT_OUTPUTS = {
    ViTEmbeddingsPipe.__name__: ["hidden_states"],
    ViTLayerPipe.__name__: ["hidden_states"],
    ViTClassifierPipe.__name__: ["logits"],
}