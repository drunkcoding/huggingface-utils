import copy
import gc
import os
import time
from tracemalloc import start
from typing import Tuple

import numpy as np
import torch
from deepspeed.pipe import PipelineModule, LayerSpec
from torch import Tensor
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm


def get_embed_dim(config):
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "n_embd"):
        return config.n_embd
    if hasattr(config, "d_model"):
        return config.d_model


def get_num_layers(config):
    if hasattr(config, "num_layers"):
        return config.num_layers
    if hasattr(config, "n_layer"):
        return config.n_layer
    if hasattr(config, "num_hidden_layers"):
        return config.n_layer


class DummyModule(nn.Module, ModuleUtilsMixin):
    device = "cpu"

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = get_embed_dim(config)
        self.n_layers = get_num_layers(config)

    def to(self, device):
        self.device = device
        # self.device_map = [self.device for _ in range(len(module_list))]
        return super().to(device)


# class T5BlockPipe(DummyModule):
#     def __init__(self, config, has_relative_attention_bias=False):
#         super().__init__(config)
#         self.block = T5Block(
#             config, has_relative_attention_bias=has_relative_attention_bias
#         )

#     def forward(self, args):
#         hidden_states, input_ids, attention_mask = args
#         outputs = self.block(
#             hidden_states,
#             attention_mask=attention_mask,
#         )
#         # print(self.layer_idx, outputs[0])
#         # attention_mask.require_grad = True
#         # print(input_ids, attention_mask, head_mask)
#         return outputs[0], input_ids, attention_mask  # , head_mask
#         # if self.layer_idx == 11: exit()
#         # return concat_outputs(outputs[0], attn_mask)

#     def copy_weights(self, model, layer_idx):
#         # super().__init__()
#         self.block.load_state_dict(model.transformer.h[layer_idx].state_dict())
#         self.dtype = model.dtype

#         assert self.block.named_parameters()
#         return


class T5StackPipe(DummyModule):
    def __init__(self, config: T5Config, embed_tokens=None) -> None:
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(self.n_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def to_layers(self):
        return [
            self.embed_tokens,
            self.dropout,
            *self.block,
            self.final_layer_norm,
            self.dropout,
        ]

    def copy_weights(self, model: T5Stack):
        self.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
        self.final_layer_norm.load_state_dict(model.final_layer_norm.state_dict())
        self.dropout.load_state_dict(model.dropout.state_dict())

        for i in range(self.n_layers):
            self.block[i].load_state_dict(model.block[i].state_dict())

    def forward(self, args):
        pass


class T5Pipe(DummyModule):
    def __init__(
            self, model: T5ForConditionalGeneration, exec_map: Tuple = None
    ) -> None:
        super().__init__(model.config)

        config = model.config
        self.total_params = sum([np.prod(p.size()) for p in model.parameters()])

        self.shared = nn.Embedding(config.vocab_size, self.embed_dim)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder = T5StackPipe(encoder_config, self.shared)
        encoder.copy_weights(model.encoder)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        decoder = T5StackPipe(decoder_config, self.shared)
        decoder.copy_weights(model.decoder)

        lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
        lm_head.load_state_dict(model.lm_head.state_dict())

        module_list = [
            *encoder.to_layers(),
            *decoder.to_layers(),
            lm_head,
        ]

        self.exec_map = (0, len(module_list)) if exec_map is None else exec_map

        self.encoder_mask = np.zeros(len(module_list)).astype(bool)
        self.encoder_mask[: len(encoder.to_layers())] = True

        self.ln_mask = np.array(
            [
                isinstance(layer, (nn.Linear, nn.Dropout, T5LayerNorm))
                for layer in module_list
            ]
        )
        self.embed_mask = np.array(
            [isinstance(layer, (nn.Embedding,)) for layer in module_list]
        )
        self.hidden_mask = np.array(
            [isinstance(layer, (nn.Dropout, T5Block)) for layer in module_list]
        )

        for i, layer in enumerate(module_list):
            if isinstance(layer, T5LayerNorm):
                self.hidden_mask[i - 1] = False

        # # use placeholder to save more memory
        # for i in range(len(module_list)):
        #     if i < self.exec_map[0] or i >= self.exec_map[1]:
        #         module_list[i] = torch.nn.Module()

        self.pipe = nn.ModuleList(module_list)

    def partition_by_parameter(self, stage, parts):
        l_params = self.total_params / parts * stage
        h_params = self.total_params / parts * (stage + 1) if stage != parts - 1 else self.total_params

        l, h = -1, -1
        layer_params = [sum([np.prod(p.size()) for p in self.pipe[idx].parameters()]) for idx in range(len(self.pipe))]
        layer_params = np.cumsum(layer_params)
        responsible_layers = (layer_params >= l_params) & (layer_params <= h_params)

        print("layer_params", layer_params)
        for idx in range(len(layer_params)):
            if layer_params[idx] >= l_params and l < 0:
                l = idx
            if layer_params[idx] <= h_params:
                h = idx

        self.exec_map = (l,h+1)



    # def parallize(self, device_map: List = None):
    #     if device_map is not None:
    #         self.device_map = device_map

    #     # print(self.device_map)
    #     new_pipe = []
    #     for idx, layer in enumerate(self.pipe):
    #         new_pipe.append(copy.deepcopy(layer.to(self.device)))

    #     del self.pipe
    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     self.pipe = nn.ModuleList(new_pipe)

    async def call_layer_async(self, idx, *args, **kwds):
        # print(args[0].device)
        # print(self.device)
        return self.pipe[idx](*args, **kwds)
        # return self.pipe[idx](
        #     *[
        #         arg.to(self.device) if arg is not None else None
        #         for arg in args
        #     ],
        #     **{
        #         k: v.to(self.device) if v is not None else None
        #         for k, v in kwds.items()
        #     },
        # )

    def call_layer_sync(self, idx, *args, **kwds):
        # return self.pipe[idx](
        #     *[
        #         arg.to(self.device) if arg is not None else None
        #         for arg in args
        #     ],
        #     **{
        #         k: v.to(self.device) if v is not None else None
        #         for k, v in kwds.items()
        #     },
        # )
        start_time = time.perf_counter()
        outputs = self.pipe[idx](*args, **kwds)
        end_time = time.perf_counter()
        print(
            "call_layer_sync", idx, 
            (end_time - start_time) * 1000,
        )
        return outputs

    @torch.no_grad()
    async def forward_async(self, args):

        # FIRST INPUT
        if len(args) == 2:
            args = args + (None, None, None, None)

        (
            input_ids,
            attention_mask,
            encoder_hidden_states,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias,
        ) = args

        extended_attention_mask = None

        
        for idx in range(*self.exec_map):
            start_time = time.perf_counter()

            is_decoder = ~self.encoder_mask[idx]
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            attention_mask = attention_mask.to(self.device)
            input_ids = input_ids.to(self.device)
            if extended_attention_mask is None:
                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, input_shape, self.device, False
                )

            if is_decoder and input_shape[1] != 1:  # self.encoder_mask[idx - 1]:
                position_bias = None
                input_ids = self._prepare_decoder_input_ids_for_generation(input_ids)
                encoder_attention_mask = attention_mask
                attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long, device=self.device)
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                input_ids = input_ids.to(self.device)

                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, input_shape, self.device, True
                )
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )

            # if idx == 0 or self.device != self.device_map[idx-1]
            # if encoder_hidden_states is not None:
            #     encoder_hidden_states = encoder_hidden_states.to(self.device)
            # if decoder_hidden_states is not None:
            #     decoder_hidden_states = decoder_hidden_states.to(self.device)
            # if position_bias is not None:
            #     position_bias = position_bias.to(self.device)
            # if encoder_decoder_position_bias is not None:
            #     encoder_decoder_position_bias = encoder_decoder_position_bias.to(self.device)
            # if input_ids is not None:
            #     input_ids = input_ids.to(self.device)
            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(self.device)

            if self.embed_mask[idx] and not is_decoder:
                # print("embed_mask encoder")
                encoder_hidden_states = await self.call_layer_async(idx, input_ids)
            elif self.embed_mask[idx] and is_decoder:
                # print("embed_mask decoder")
                decoder_hidden_states = await self.call_layer_async(idx, input_ids)
            elif self.ln_mask[idx] and not is_decoder:
                # print("ln_mask encoder")
                encoder_hidden_states = await self.call_layer_async(
                    idx, encoder_hidden_states
                )
            elif self.ln_mask[idx] and is_decoder:
                # print("ln_mask decoder")
                decoder_hidden_states = await self.call_layer_async(
                    idx, decoder_hidden_states
                )
            elif is_decoder:
                # print("block decoder")
                # attention_mask = attention_mask.to(self.device)
                # extended_attention_mask = self.get_extended_attention_mask(
                #     attention_mask, input_shape, self.device, True
                # )
                # encoder_extended_attention_mask = self.invert_attention_mask(
                #     encoder_attention_mask
                # )
                # print(decoder_hidden_states, encoder_hidden_states)
                layer_outputs = await self.call_layer_async(
                    idx,
                    decoder_hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                )
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                decoder_hidden_states = layer_outputs[0]
                position_bias = layer_outputs[2]
                encoder_decoder_position_bias = layer_outputs[3]
            elif not is_decoder:
                # print("block encoder")
                # attention_mask = attention_mask.to(self.device)
                # extended_attention_mask = self.get_extended_attention_mask(
                #     attention_mask, input_shape, self.device, False
                # )
                layer_outputs = await self.call_layer_async(
                    idx,
                    encoder_hidden_states,
                    position_bias=position_bias,
                    attention_mask=extended_attention_mask,
                )
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                encoder_hidden_states = layer_outputs[0]
                position_bias = layer_outputs[2]

            end_time = time.perf_counter()
            # print(idx, "layer", (end_time-start_time)*1000)


        # print(encoder_hidden_states, decoder_hidden_states, input_ids, attention_mask)

        return (
            encoder_hidden_states,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias,
        )

    @torch.no_grad()
    def forward(self, args, output_hidden_states=False):

        # FIRST INPUT
        if len(args) == 2:
            args = args + (None, None, None, None)

        (
            input_ids,
            attention_mask,
            encoder_hidden_states,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias,
        ) = args

        extended_attention_mask = None
        if output_hidden_states:
            all_hidden_states = ()

        
        for idx in range(*self.exec_map):

            start_time = time.perf_counter()

            is_decoder = ~self.encoder_mask[idx]
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if extended_attention_mask is None:
                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, input_shape, self.device, False
                )

            if is_decoder and input_shape[1] != 1:  # self.encoder_mask[idx - 1]:
                position_bias = None
                input_ids = self._prepare_decoder_input_ids_for_generation(input_ids)
                encoder_attention_mask = attention_mask
                attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long, device=self.device)
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])

                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, input_shape, self.device, True
                )
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )

            if self.embed_mask[idx] and not is_decoder:
                # print("embed_mask encoder")
                encoder_hidden_states = self.pipe[idx](input_ids)
            elif self.embed_mask[idx] and is_decoder:
                # print("embed_mask decoder")
                decoder_hidden_states = self.pipe[idx](input_ids)
            elif self.ln_mask[idx] and not is_decoder:
                # print("ln_mask encoder")
                encoder_hidden_states = self.pipe[idx](
                    encoder_hidden_states
                )
            elif self.ln_mask[idx] and is_decoder:
                # print("ln_mask decoder")
                decoder_hidden_states = self.pipe[idx](
                    decoder_hidden_states
                )
            elif is_decoder:
                # print("block decoder")
                # print(decoder_hidden_states, encoder_hidden_states)
                layer_outputs = self.pipe[idx](
                    decoder_hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                )
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                # print(layer_outputs)
                decoder_hidden_states = layer_outputs[0]
                position_bias = layer_outputs[2]
                encoder_decoder_position_bias = layer_outputs[3]
            elif not is_decoder:
                # print("block encoder")
                layer_outputs = self.pipe[idx](
                    encoder_hidden_states,
                    position_bias=position_bias,
                    attention_mask=extended_attention_mask,
                )
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                encoder_hidden_states = layer_outputs[0]
                position_bias = layer_outputs[2]

            if output_hidden_states and self.hidden_mask[idx]:
                all_hidden_states = all_hidden_states + (
                    decoder_hidden_states if is_decoder else encoder_hidden_states,
                )

            end_time = time.perf_counter()
            # print(idx, "layer", (end_time-start_time)*1000)

        # print(encoder_hidden_states, decoder_hidden_states, encoder_decoder_position_bias, input_ids, attention_mask)
        if output_hidden_states:
            return (
                encoder_hidden_states,
                decoder_hidden_states,
                position_bias,
                encoder_decoder_position_bias,
                all_hidden_states
            )

        return (
            encoder_hidden_states,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias,
        )

    # def forward(self, args, output_hidden_states=False):
    #     (
    #         encoder_hidden_states,
    #         decoder_hidden_states,
    #         position_bias,
    #         encoder_decoder_position_bias,
    #         input_ids,
    #         attention_mask,
    #     ) = args
    #     all_hidden_states = () if output_hidden_states else None
    #     for idx in range(*self.exec_map):
    #         is_decoder = ~self.encoder_mask[idx]
    #         input_shape = input_ids.size()
    #         input_ids = input_ids.view(-1, input_shape[-1])

    #         # print(type(self.pipe[idx]))

    #         if is_decoder and self.encoder_mask[idx - 1]:
    #             position_bias = None
    #             input_ids = self._prepare_decoder_input_ids_for_generation(input_ids)
    #             encoder_attention_mask = attention_mask
    #             attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)
    #             input_shape = input_ids.size()
    #             input_ids = input_ids.view(-1, input_shape[-1])

    #         if self.embed_mask[idx] and not is_decoder:
    #             # print("embed_mask encoder")
    #             encoder_hidden_states = self.pipe[idx](input_ids)
    #         elif self.embed_mask[idx] and is_decoder:
    #             # print("embed_mask decoder")
    #             decoder_hidden_states = self.pipe[idx](input_ids)
    #         elif self.ln_mask[idx] and not is_decoder:
    #             # print("ln_mask encoder")
    #             encoder_hidden_states = self.pipe[idx](encoder_hidden_states)
    #         elif self.ln_mask[idx] and is_decoder:
    #             # print("ln_mask decoder")
    #             decoder_hidden_states = self.pipe[idx](decoder_hidden_states)
    #         elif is_decoder:
    #             # print("block decoder")
    #             extended_attention_mask = self.get_extended_attention_mask(
    #                 attention_mask, input_shape, self.device, True
    #             )
    #             encoder_extended_attention_mask = self.invert_attention_mask(
    #                 encoder_attention_mask
    #             )
    #             # print(decoder_hidden_states, encoder_hidden_states)
    #             layer_outputs = self.pipe[idx](
    #                 decoder_hidden_states,
    #                 attention_mask=extended_attention_mask,
    #                 position_bias=position_bias,
    #                 encoder_decoder_position_bias=encoder_decoder_position_bias,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 encoder_attention_mask=encoder_extended_attention_mask,
    #             )
    #             layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
    #             decoder_hidden_states = layer_outputs[0]
    #             position_bias = layer_outputs[2]
    #             encoder_decoder_position_bias = layer_outputs[3]
    #         elif not is_decoder:
    #             # print("block encoder")
    #             extended_attention_mask = self.get_extended_attention_mask(
    #                 attention_mask, input_shape, self.device, False
    #             )
    #             layer_outputs = self.pipe[idx](
    #                 encoder_hidden_states,
    #                 position_bias=position_bias,
    #                 attention_mask=extended_attention_mask,
    #             )
    #             layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
    #             encoder_hidden_states = layer_outputs[0]
    #             position_bias = layer_outputs[2]

    #         # print(encoder_hidden_states, decoder_hidden_states, input_ids, attention_mask)

    #         if output_hidden_states and self.hidden_mask[idx]:
    #             all_hidden_states = all_hidden_states + (
    #                 decoder_hidden_states if is_decoder else encoder_hidden_states,
    #             )

    #     return (
    #         encoder_hidden_states,
    #         decoder_hidden_states,
    #         position_bias,
    #         encoder_decoder_position_bias,
    #         all_hidden_states,
    #     )

    def get_extended_attention_mask(
            self,
            attention_mask: Tensor,
            input_shape: Tuple[int],
            device: torch.device,
            is_decoder=False,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                        seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                        <= seq_ids[None, :, None]
                )
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = (
                        causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (
                                                      1.0 - encoder_extended_attention_mask
                                              ) * -1e4
        elif self.dtype == torch.float32:
            encoder_extended_attention_mask = (
                                                      1.0 - encoder_extended_attention_mask
                                              ) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask

    def _prepare_decoder_input_ids_for_generation(
            self,
            input_ids: torch.LongTensor,
            decoder_start_token_id: int = None,
            bos_token_id: int = None,
    ) -> torch.LongTensor:
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id
        )
        decoder_input_ids = (
                torch.ones(
                    (input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device
                )
                * decoder_start_token_id
        )
        return decoder_input_ids

    def _get_decoder_start_token_id(
            self, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> int:
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
                hasattr(self.config, "decoder")
                and hasattr(self.config.decoder, "decoder_start_token_id")
                and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
                hasattr(self.config, "decoder")
                and hasattr(self.config.decoder, "bos_token_id")
                and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )


def get_decoder_start_token_id(
        decoder_start_token_id: int = None, bos_token_id: int = None
) -> int:
    decoder_start_token_id = (
        decoder_start_token_id
        if decoder_start_token_id is not None
        else self.config.decoder_start_token_id
    )
    bos_token_id = (
        bos_token_id  # if bos_token_id is not None else self.config.bos_token_id
    )

    if decoder_start_token_id is not None:
        return decoder_start_token_id
    elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
    ):
        return self.config.decoder.decoder_start_token_id
    elif bos_token_id is not None:
        return bos_token_id
    elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
    ):
        return self.config.decoder.bos_token_id
    raise ValueError(
        "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
    )


def prepare_decoder_input_ids_for_generation(
        input_ids: torch.LongTensor,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
) -> torch.LongTensor:
    decoder_start_token_id = get_decoder_start_token_id(
        decoder_start_token_id, bos_token_id
    )
    decoder_input_ids = (
            torch.ones(
                (input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device
            )
            * decoder_start_token_id
    )
    return decoder_input_ids


def invert_attention_mask(encoder_attention_mask: Tensor) -> Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

    Returns:
        :obj:`torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    # encoder_extended_attention_mask = encoder_extended_attention_mask.to(
    #     dtype=self.dtype
    # )  # fp16 compatibility
    #
    # if self.dtype == torch.float16:
    #     encoder_extended_attention_mask = (
    #                                               1.0 - encoder_extended_attention_mask
    #                                       ) * -1e4
    # elif self.dtype == torch.float32:
    encoder_extended_attention_mask = (
                                              1.0 - encoder_extended_attention_mask
                                      ) * -1e9
    # else:
    #     raise ValueError(
    #         f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
    #     )

    return encoder_extended_attention_mask


def get_extended_attention_mask(
        attention_mask: Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        is_decoder=False,
) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
            )
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, seq_length, prefix_seq_len),
                            device=device,
                            dtype=causal_mask.dtype,
                        ),
                        causal_mask,
                    ],
                    axis=-1,
                )

            extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # extended_attention_mask = extended_attention_mask.to(
    #     dtype=self.dtype
    # )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


MAGIC_NUMBERS = np.array([
    120907, 120917, 120919, 120929, 120937, 120941, 120943, 120947, 120977, 120997,
    121001, 121007, 121013, 121019, 121021, 121039, 121061, 121063, 121067, 121081,
    121123, 121139, 121151, 121157, 121169, 121171, 121181, 121189, 121229, 121259,
    121267, 121271, 121283, 121291, 121309, 121313, 121321, 121327, 121333, 121343,
    121349, 121351, 121357, 121367, 121369, 121379, 121403, 121421, 121439, 121441,
    121447, 121453, 121469, 121487, 121493, 121501, 121507, 121523, 121531, 121547,
    121553, 121559, 121571, 121577, 121579, 121591, 121607, 121609, 121621, 121631,
    121633, 121637, 121661, 121687, 121697, 121711, 121721, 121727, 121763, 121787,
    121789, 121843, 121853, 121867, 121883, 121889, 121909, 121921, 121931, 121937,
    121949, 121951, 121963, 121967, 121993, 121997, 122011, 122021, 122027, 122029,
    122033, 122039, 122041, 122051, 122053, 122069, 122081, 122099, 122117, 122131,
    122147, 122149, 122167, 122173, 122201, 122203, 122207, 122209, 122219, 122231,
    122251, 122263, 122267, 122273, 122279, 122299, 122321, 122323, 122327, 122347,
    122363, 122387, 122389, 122393, 122399, 122401, 122443, 122449, 122453, 122471,
    122477, 122489, 122497, 122501, 122503, 122509, 122527, 122533, 122557, 122561,
    122579, 122597, 122599, 122609, 122611, 122651, 122653, 122663, 122693, 122701,
    122719, 122741, 122743, 122753, 122761, 122777, 122789, 122819, 122827, 122833,
    122839, 122849, 122861, 122867, 122869, 122887, 122891, 122921, 122929, 122939,
    122953, 122957, 122963, 122971, 123001, 123007, 123017, 123031, 123049, 123059,
    123077, 123083, 123091, 123113, 123121, 123127, 123143, 123169, 123191, 123203,
    123209, 123217, 123229, 123239, 123259, 123269, 123289, 123307, 123311, 123323,
    123341, 123373, 123377, 123379, 123397, 123401, 123407, 123419, 123427, 123433,
    123439, 123449, 123457, 123479, 123491, 123493, 123499, 123503, 123517, 123527,
    123547, 123551, 123553, 123581, 123583, 123593, 123601, 123619, 123631, 123637,
    123653, 123661, 123667, 123677, 123701, 123707, 123719, 123727, 123731, 123733,
    123737, 123757, 123787, 123791, 123803, 123817, 123821, 123829, 123833, 123853,
    123863, 123887, 123911, 123923, 123931, 123941, 123953, 123973, 123979, 123983,
    123989, 123997, 124001, 124021, 124067, 124087, 124097, 124121, 124123, 124133,
    124139, 124147, 124153, 124171, 124181, 124183, 124193, 124199, 124213, 124231,
    124247, 124249, 124277, 124291, 124297, 124301, 124303, 124309, 124337, 124339,
    124343, 124349, 124351, 124363, 124367, 124427, 124429, 124433, 124447, 124459,
    124471, 124477, 124489, 124493, 124513, 124529, 124541, 124543, 124561, 124567,
    124577, 124601, 124633, 124643, 124669, 124673, 124679, 124693, 124699, 124703,
    124717, 124721, 124739, 124753, 124759, 124769, 124771, 124777, 124781, 124783,
    124793, 124799, 124819, 124823, 124847, 124853, 124897, 124907, 124909, 124919,
    124951, 124979, 124981, 124987, 124991, 125003, 125017, 125029, 125053, 125063,
    125093, 125101, 125107, 125113, 125117, 125119, 125131, 125141, 125149, 125183,
    125197, 125201, 125207, 125219, 125221, 125231, 125243, 125261, 125269, 125287,
    125299, 125303, 125311, 125329, 125339, 125353, 125371, 125383, 125387, 125399,
    125407, 125423, 125429, 125441, 125453, 125471, 125497, 125507, 125509, 125527,
    125539, 125551, 125591, 125597, 125617, 125621, 125627, 125639, 125641, 125651,
    125659, 125669, 125683, 125687, 125693, 125707, 125711, 125717, 125731, 125737,
    125743, 125753, 125777, 125789, 125791, 125803, 125813, 125821, 125863, 125887,
    125897, 125899, 125921, 125927, 125929, 125933, 125941, 125959, 125963, 126001,
    126011, 126013, 126019, 126023, 126031, 126037, 126041, 126047, 126067, 126079,
    126097, 126107, 126127, 126131, 126143, 126151, 126173, 126199, 126211, 126223,
    126227, 126229, 126233, 126241, 126257, 126271, 126307, 126311, 126317, 126323,
    126337, 126341, 126349, 126359, 126397, 126421, 126433, 126443, 126457, 126461,
    126473, 126481, 126487, 126491, 126493, 126499, 126517, 126541, 126547, 126551,
    126583, 126601, 126611, 126613, 126631, 126641, 126653, 126683, 126691, 126703,
    126713, 126719, 126733, 126739, 126743, 126751, 126757, 126761, 126781, 126823,
    126827, 126839, 126851, 126857, 126859, 126913, 126923, 126943, 126949, 126961,
    126967, 126989, 127031, 127033, 127037, 127051, 127079, 127081, 127103, 127123,
    127133, 127139, 127157, 127163, 127189, 127207, 127217, 127219, 127241, 127247,
    127249, 127261, 127271, 127277, 127289, 127291, 127297, 127301, 127321, 127331,
    127343, 127363, 127373, 127399, 127403, 127423, 127447, 127453, 127481, 127487,
    127493, 127507, 127529, 127541, 127549, 127579, 127583, 127591, 127597, 127601,
    127607, 127609, 127637, 127643, 127649, 127657, 127663, 127669, 127679, 127681,
    127691, 127703, 127709, 127711, 127717, 127727, 127733, 127739, 127747, 127763,
    127781, 127807, 127817, 127819, 127837, 127843, 127849, 127859, 127867, 127873,
])


def format_inputs(args, ds):
    if not ds: return args
    return tuple([None if torch.sum(t) == 127873 else t for t in args])


def format_outputs(args, ds):
    if not ds: return args
    shape = args[0].shape
    device = args[0].device
    return tuple([torch.Tensor([127873]).to(device) if t is None else t for t in args])


def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

class T5EmbeddingPipe(nn.Module):
    def __init__(self, config: T5Config, ds=False) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = get_embed_dim(config)
        self.embed = nn.Embedding(config.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.is_decoder = config.is_decoder

        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def forward(self, args):
        # print(os.getpid(), "T5EmbeddingPipe", self.is_decoder)
        if len(args) == 2:
            args += tuple([torch.Tensor([127873]).to(args[0].device)] * 6) if self.deepspeed_enabled else tuple([None] * 6)
        (
            encoder_input_ids,
            encoder_attention_mask,
            encoder_hidden_states,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias
        ) = format_inputs(args, self.deepspeed_enabled)

        if self.is_decoder:
            decoder_input_ids = prepare_decoder_input_ids_for_generation(encoder_input_ids,
                                                                         self.config.decoder_start_token_id,
                                                                         self.config.eos_token_id)
            decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.shape, dtype=torch.long)
            input_shape = decoder_input_ids.size()
            decoder_input_ids = decoder_input_ids.view(-1, input_shape[-1])

        if self.is_decoder:
            decoder_hidden_states = self.embed(decoder_input_ids)
            decoder_hidden_states = self.dropout(decoder_hidden_states)
        else:
            encoder_hidden_states = self.embed(encoder_input_ids)
            encoder_hidden_states = self.dropout(encoder_hidden_states)

        return format_outputs(
            (
                encoder_input_ids,
                encoder_attention_mask,
                encoder_hidden_states,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_hidden_states,
                None,   # position_bias
                encoder_decoder_position_bias
            ), self.deepspeed_enabled
        )


class T5BlockPipe(nn.Module):
    def __init__(self, config: T5Config, i: int, ds=False) -> None:
        super().__init__()

        self.block = T5Block(config, has_relative_attention_bias=bool(i == 0))
        self.is_decoder = config.is_decoder
        self.block_idx = i

        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def forward(self, args):
        # print(os.getpid(), "T5BlockPipe", self.block_idx, self.is_decoder)
        (
            encoder_input_ids,
            encoder_attention_mask,
            encoder_hidden_states,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias
        ) = format_inputs(args, self.deepspeed_enabled)

        if self.is_decoder:
            # print("block decoder")
            input_shape = decoder_input_ids.size()
            extended_attention_mask = get_extended_attention_mask(
                decoder_attention_mask, input_shape, decoder_input_ids.device, True
            )
            encoder_extended_attention_mask = invert_attention_mask(
                encoder_attention_mask
            )
            # print(decoder_hidden_states, encoder_hidden_states)
            layer_outputs = self.block(
                decoder_hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            decoder_hidden_states = layer_outputs[0]
            position_bias = layer_outputs[2]
            encoder_decoder_position_bias = layer_outputs[3]
        if not self.is_decoder:
            # print("block encoder")
            input_shape = encoder_input_ids.size()
            extended_attention_mask = get_extended_attention_mask(
                encoder_attention_mask, input_shape, encoder_input_ids.device, False
            )
            layer_outputs = self.block(
                encoder_hidden_states,
                position_bias=position_bias,
                attention_mask=extended_attention_mask,
            )
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            encoder_hidden_states = layer_outputs[0]
            position_bias = layer_outputs[2]

        return format_outputs(
            (
                encoder_input_ids,
                encoder_attention_mask,
                encoder_hidden_states,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_hidden_states,
                position_bias,
                encoder_decoder_position_bias
            ), self.deepspeed_enabled
        )


class T5LMHeadPipe(nn.Module):
    def __init__(self, config: T5Config, ds=False) -> None:
        super().__init__()

        self.embed_dim = get_embed_dim(config)
        self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
        self.is_decoder = config.is_decoder

        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def forward(self, args):
        # print(os.getpid(), "T5LMHeadPipe", self.is_decoder)
        (
            encoder_input_ids,
            encoder_attention_mask,
            encoder_hidden_states,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias
        ) = format_inputs(args, self.deepspeed_enabled)

        return self.lm_head(decoder_hidden_states)


class T5StackFFPipe(nn.Module):
    def __init__(self, config: T5Config, ds=False) -> None:
        super().__init__()

        self.embed_dim = get_embed_dim(config)

        self.final_layer_norm = T5LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.is_decoder = config.is_decoder
        
        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def forward(self, args):
        # print(os.getpid(), "T5StackFFPipe", self.is_decoder)
        (
            encoder_input_ids,
            encoder_attention_mask,
            encoder_hidden_states,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias
        ) = format_inputs(args)  if self.deepspeed_enabled else args

        if self.is_decoder:
            decoder_hidden_states = self.final_layer_norm(decoder_hidden_states)
            decoder_hidden_states = self.dropout(decoder_hidden_states)
        if not self.is_decoder:
            encoder_hidden_states = self.final_layer_norm(encoder_hidden_states)
            encoder_hidden_states = self.dropout(encoder_hidden_states)

        return format_outputs(
            (
                encoder_input_ids,
                encoder_attention_mask,
                encoder_hidden_states,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_hidden_states,
                position_bias,
                encoder_decoder_position_bias
            ), self.deepspeed_enabled
        )


class T5PyTorchPipe(nn.Module):
    def __init__(self, model: T5ForConditionalGeneration, exec_map: Tuple = None) -> None:
        super().__init__()
        
        config = model.config
        # self.total_params = sum([np.prod(p.size()) for p in model.parameters()])

        self.embed_dim = get_embed_dim(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.n_layers = get_num_layers(config)

        self.layers = []

        encoder_embed = T5EmbeddingPipe(encoder_config)
        encoder_embed.embed.load_state_dict(model.encoder.embed_tokens.state_dict())
        self.layers.append(encoder_embed)
        for i in range(self.n_layers):
            encoder_block = T5BlockPipe(encoder_config, i)
            encoder_block.block.load_state_dict(model.encoder.block[i].state_dict())
            self.layers.append(encoder_block)
        encoder_stack_ff = T5StackFFPipe(encoder_config)
        encoder_stack_ff.final_layer_norm.load_state_dict(model.encoder.final_layer_norm.state_dict())
        encoder_stack_ff.dropout.load_state_dict(model.encoder.dropout.state_dict())
        self.layers.append(encoder_stack_ff)


        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        decoder_embed = T5EmbeddingPipe(decoder_config)
        decoder_embed.embed.load_state_dict(model.decoder.embed_tokens.state_dict())
        self.layers.append(decoder_embed)
        for i in range(self.n_layers):
            decoder_block = T5BlockPipe(decoder_config, i)
            decoder_block.block.load_state_dict(model.decoder.block[i].state_dict())
            self.layers.append(decoder_block)
        decoder_stack_ff = T5StackFFPipe(decoder_config)
        decoder_stack_ff.final_layer_norm.load_state_dict(model.decoder.final_layer_norm.state_dict())
        decoder_stack_ff.dropout.load_state_dict(model.decoder.dropout.state_dict())
        self.layers.append(decoder_stack_ff)

        lm_head = T5LMHeadPipe(decoder_config)
        lm_head.lm_head.load_state_dict(model.lm_head.state_dict())
        self.layers.append(lm_head)
        
        self.total_params = sum([
            sum([np.prod(p.size()) for p in layer.parameters()])
            for layer in self.layers
        ])

        # super().__init__(layers=encoder_specs + decoder_specs, **kwargs)

        self.exec_map = exec_map if exec_map is not None else (0, len(self.layers))

    def convert(self, device):
        for idx in range(*self.exec_map):
            self.layers[idx] = self.layers[idx].to(device)

        # use placeholder to save more memory
        for i in range(len(self.layers)):
            if i < self.exec_map[0] or i >= self.exec_map[1]:
                self.layers[i] = None

        torch.cuda.empty_cache()
        gc.collect()
        self.device = device

    def partition_by_parameter(self, stage, parts):
        l_params = self.total_params / parts * stage
        h_params = self.total_params / parts * (stage + 1) if stage != parts - 1 else self.total_params

        layer_params = [sum([np.prod(p.size()) for p in self.layers[idx].parameters()]) for idx in range(len(self.layers))]
        layer_params = np.cumsum(layer_params)
        responsible_layers = np.argwhere((layer_params >= l_params) & (layer_params <= h_params)).flatten()

        self.exec_map = (responsible_layers[0], responsible_layers[-1]+1)

        # print("layer_params", layer_params)
        # for idx in range(len(layer_params)):
        #     if layer_params[idx] >= l_params and l < 0:
        #         l = idx
        #     if layer_params[idx] <= h_params:
        #         h = idx

        

    # @torch.no_grad()
    def forward(self, args, output_hidden_states=False):
        outputs = args
        all_hidden_states = ()
        for idx in range(*self.exec_map):
            outputs = self.layers[idx](outputs)
            if output_hidden_states:
                if idx != len(self.layers) - 1:
                    all_hidden_states = all_hidden_states + (
                        outputs[5] if self.layers[idx].is_decoder else outputs[2],
                    )
        if output_hidden_states:
            return (
                outputs,
                all_hidden_states
            )
        return outputs


class T5DeepSpeedPipe(PipelineModule):
    def __init__(self, config: T5Config, **kwargs) -> None:
        self.embed_dim = get_embed_dim(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.n_layers = get_num_layers(config)

        encoder_specs = [
            LayerSpec(T5EmbeddingPipe, encoder_config, True),
            *[LayerSpec(T5BlockPipe, encoder_config, i, True) for i in range(self.n_layers)],
            LayerSpec(T5StackFFPipe, encoder_config, True),
        ]

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        decoder_specs = [
            LayerSpec(T5EmbeddingPipe, decoder_config, True),
            *[LayerSpec(T5BlockPipe, decoder_config, i, True) for i in range(self.n_layers)],
            LayerSpec(T5StackFFPipe, decoder_config, True),
            LayerSpec(T5LMHeadPipe, decoder_config, True),
        ]

        super().__init__(layers=encoder_specs + decoder_specs, **kwargs)
