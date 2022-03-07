import copy
from typing import Tuple
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertLayer
from transformers import BertForQuestionAnswering

from torch import nn
import numpy as np
from deepspeed.pipe import PipelineModule, LayerSpec

from hfutils.model_pipe import format_inputs, format_outputs, get_num_layers

class BertEmbeddingPipe(BertEmbeddings):
    def __init__(self, config: BertConfig, ds=False) -> None:
        super().__init__(config)
        self.deepspeed_enabled = ds

    def forward(self, args):
        if len(args) == 3:
            args = args + (None, )
        input_ids, token_type_ids, attention_mask, hidden_states = format_inputs(args, self.deepspeed_enabled)

        hidden_states = super().forward(input_ids, token_type_ids)

        return format_outputs(
            (attention_mask, hidden_states), self.deepspeed_enabled
        )

class BertLayerPipe(BertLayer):
    def __init__(self, config: BertConfig, ds=False):
        super().__init__(config)
        self.deepspeed_enabled = ds

    def forward(self, args):
        attention_mask, hidden_states = format_inputs(args, self.deepspeed_enabled)

        layer_outputs = super().forward(hidden_states, attention_mask)[0]
        hidden_states = layer_outputs[0]

        return format_outputs(
            (attention_mask, hidden_states), self.deepspeed_enabled
        )

class BertPoolerPipe(BertPooler):
    def __init__(self, config: BertConfig, ds=False):
        super().__init__(config)
        self.deepspeed_enabled = ds       

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels) 

    def forward(self, args):
        attention_mask, hidden_states = format_inputs(args, self.deepspeed_enabled)
        hidden_states = super().forward(hidden_states)[0]
        return format_outputs(
            (attention_mask, hidden_states), self.deepspeed_enabled
        )

class BertHeadPipeForQuestionAnswering(nn.Module):
    def __init__(self, config: BertConfig, ds=False):
        super().__init__()
        self.deepspeed_enabled = ds        

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, args):
        attention_mask, hidden_states = format_inputs(args, self.deepspeed_enabled)
        logits = self.qa_outputs(hidden_states)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return format_outputs(
            (start_logits, end_logits), self.deepspeed_enabled
        )

class BertPyTorchPipeForQuestionAnswering(nn.Module):
    def __init__(self, model: BertForQuestionAnswering, exec_map: Tuple = None) -> None:
        super().__init__()

        config = model.config
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(encoder_config)

        self.layers = []
        encoder_embed = BertEmbeddingPipe(encoder_config)
        encoder_embed.load_state_dict(model.bert.embeddings.state_dict())
        self.layers.append(encoder_embed)

        for i in range(self.n_layers):
            encoder_block = BertLayerPipe(encoder_config)
            encoder_block.load_state_dict(model.bert.encoder.layer[i].state_dict())
            self.layers.append(encoder_block)

        encoder_pooler = BertPoolerPipe(encoder_config)
        encoder_pooler.load_state_dict(model.bert.pooler.state_dict())
        self.layers.append(encoder_pooler)

        qa_outputs = BertHeadPipeForQuestionAnswering(encoder_config)
        qa_outputs.load_state_dict(model.qa_outputs.state_dict())
        self.layers.append(qa_outputs)

        self.total_params = sum([
            sum([np.prod(p.size()) for p in layer.parameters()])
            for layer in self.layers
        ])

        self.exec_map = exec_map if exec_map is not None else (0, len(self.layers))


class BertDeepSpeedPipeForQuestionAnswering(PipelineModule):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(config)

        encoder_specs = [
            LayerSpec(BertEmbeddingPipe, encoder_config, True),
            *[LayerSpec(BertLayerPipe, encoder_config, True) for i in range(self.n_layers)],
            LayerSpec(BertPoolerPipe, encoder_config, True),
            LayerSpec(BertHeadPipeForQuestionAnswering, encoder_config, True),
        ]

        super().__init__(layers=encoder_specs, **kwargs)
