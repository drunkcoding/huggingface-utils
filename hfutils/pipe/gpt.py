import torch
from torch import nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

from hfutils.pipe.base import format_inputs, format_outputs, get_embed_dropout

class GPTEmbedding(nn.Module):
    def __init__(self, config, ds=False):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        if config.model_type != "gptj":
            # print("max_position_embeddings")
            self.wpe = nn.Embedding(
                config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(get_embed_dropout(config))

        self.deepspeed_enabled = ds

    def forward(self, args):

        (
            input_ids,
            attention_mask,
            hidden_states,
        ) = format_inputs(args, self.deepspeed_enabled)
        position_ids = None

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        # batch_size = input_ids.shape[0]

        device = input_ids.device  # if input_ids is not None else inputs_embeds.device
        past_length = 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if hasattr(self, 'wpe'):
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)
        return format_outputs(
            (
                input_ids,
                attention_mask,
                hidden_states
            ), self.deepspeed_enabled
        )

class GPTBlockPipe(nn.Module):
    def __init__(self, config, layer_idx=None, ds=False):
        super().__init__()

        # print(config)

        if config.model_type == "gptj":
            self.block = GPTJBlock(config)
        elif config.model_type == "gpt2":
            self.block = GPT2Block(config, layer_idx)
        elif config.model_type == "gpt_neo":
            self.block = GPTNeoBlock(config, layer_idx)
        else:
            raise NotImplementedError()
        
        self.layer_idx = layer_idx
        self.deepspeed_enabled = ds

    def forward(self, args):
        (
            input_ids,
            attention_mask,
            hidden_states,
        ) = format_inputs(args, self.deepspeed_enabled)

        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        return format_outputs(
            (
                input_ids,
                attention_mask,
                hidden_states
            ), self.deepspeed_enabled
        )

class GPTOutput(nn.Module):
    def __init__(self, config, ds=False):
        super().__init__()
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.num_labels = config.num_labels
        self.score = nn.Linear(self.embed_dim, self.num_labels, bias=False)

        self.deepspeed_enabled = ds

    def forward(self, args):
        (
            input_ids,
            attention_mask,
            hidden_states,
        ) = format_inputs(args, self.deepspeed_enabled)
        hidden_states = self.ln_f(hidden_states)
        input_shape = input_ids.size()
        output_shape = input_shape + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(*output_shape)
        output = self.score(hidden_states)

        batch_size, _ = input_ids.shape[:2]

        sequence_lengths = torch.ne(input_ids, 50256).sum(-1) - 1

        pooled_logits = output[range(batch_size), sequence_lengths]
        return pooled_logits