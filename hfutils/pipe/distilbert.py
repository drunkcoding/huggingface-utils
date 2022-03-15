import copy
from typing import Tuple
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import Embeddings, TransformerBlock
from transformers import BertForQuestionAnswering

from torch import nn
import numpy as np
from deepspeed.pipe import PipelineModule, LayerSpec

from hfutils.pipe.base import PipeMethods, format_inputs, format_outputs, get_num_layers

class EmbeddingsPipe(Embeddings):
    def __init__(self, config: DistilBertConfig, ds=False) -> None:
        super().__init__(config)
        self.deepspeed_enabled = ds

    def forward(self, args):
        if len(args) == 3:
            args = args + (None, )
        input_ids, token_type_ids, attention_mask, hidden_states = format_inputs(args, self.deepspeed_enabled)

        hidden_states = super().forward(input_ids)

        return format_outputs(
            (attention_mask, hidden_states), self.deepspeed_enabled
        )

class TransformerPipe(TransformerBlock):
    def __init__(self, config: DistilBertConfig, ds=False):
        super().__init__(config)
        self.deepspeed_enabled = ds

    def forward(self, args):
        attention_mask, hidden_states = format_inputs(args, self.deepspeed_enabled)

        layer_outputs = super().forward(hidden_states, attention_mask)[0]
        hidden_states = layer_outputs[0]

        return format_outputs(
            (attention_mask, hidden_states), self.deepspeed_enabled
        )

from bert import BertHeadPipeForQuestionAnswering

class DistilBertPyTorchPipeForQuestionAnswering(nn.Module, PipeMethods):
    def __init__(self, model: BertForQuestionAnswering, exec_map: Tuple = None) -> None:
        super().__init__()

        config = model.config
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(encoder_config)

        self.layers = []
        encoder_embed = EmbeddingsPipe(encoder_config)
        encoder_embed.load_state_dict(model.bert.embeddings.state_dict())
        self.layers.append(encoder_embed)

        for i in range(self.n_layers):
            encoder_block = TransformerPipe(encoder_config)
            encoder_block.load_state_dict(model.bert.encoder.layer[i].state_dict())
            self.layers.append(encoder_block)

        qa_outputs = BertHeadPipeForQuestionAnswering(encoder_config)
        qa_outputs.load_state_dict(model.qa_outputs.state_dict())
        self.layers.append(qa_outputs)

        self.total_params = sum([
            sum([np.prod(p.size()) for p in layer.parameters()])
            for layer in self.layers
        ])

        self.layers = nn.ModuleList(self.layers)

        self.exec_map = exec_map if exec_map is not None else (0, len(self.layers))


class DistilBertDeepSpeedPipeForQuestionAnswering(PipelineModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(config)

        encoder_specs = [
            LayerSpec(EmbeddingsPipe, encoder_config, True),
            *[LayerSpec(TransformerPipe, encoder_config, True) for _ in range(self.n_layers)],
            LayerSpec(BertHeadPipeForQuestionAnswering, encoder_config, True),
        ]

        super().__init__(layers=encoder_specs, **kwargs)
