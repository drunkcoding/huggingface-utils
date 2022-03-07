from turtle import forward
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTPooler

from torch import nn

from hfutils.model_pipe import format_inputs, format_outputs


class ViTEmbeddingsPipe(ViTEmbeddings):
    def __init__(self, config, ds=False):
        super().__init__(config)
        self.deepspeed_enabled = ds

    def forward(self, args):
        if len(args) == 3:
            args = args + (None, )
        pixel_values, hidden_states = format_inputs(args, self.deepspeed_enabled)


        hidden_states = super().forward(pixel_values)

        return format_outputs(
            (hidden_states, ), self.deepspeed_enabled
        )