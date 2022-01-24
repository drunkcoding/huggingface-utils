import copy
from pyexpat import model
import time
from turtle import position
from typing import List, Tuple
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.generation_utils import GenerationMixin
import torch
from torch import nn
import gc
import numpy as np
from torch import Tensor


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

        self.exec_map = exec_map if exec_map is not None else (0, len(module_list))

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

        # use placeholder to save more memory
        for i in range(len(module_list)):
            if i < self.exec_map[0] or i >= self.exec_map[1]:
                module_list[i] = torch.nn.Module()

        self.pipe = nn.ModuleList(module_list)

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
        return self.pipe[idx](*args, **kwds)

    @torch.no_grad()
    async def forward_async(self, args):

        # FIRST INPUT
        if len(args) == 2:
            args =  args + (None, None, None, None)

        (
            input_ids,
            attention_mask,
            encoder_hidden_states,
            decoder_hidden_states,
            position_bias,
            encoder_decoder_position_bias,
        ) = args

        extended_attention_mask = None

        start_time = time.perf_counter()
        for idx in range(*self.exec_map):
            is_decoder = ~self.encoder_mask[idx]
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            # print(type(self.pipe[idx]))
            # end_time = time.perf_counter()
            # print(idx, "layer infetence time", (end_time-start_time)*1000)
            # start_time = time.perf_counter()

            attention_mask = attention_mask.to(self.device)
            input_ids = input_ids.to(self.device)
            if extended_attention_mask is None:
                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, input_shape, self.device, False
                )

            if is_decoder and input_shape[1] != 1: # self.encoder_mask[idx - 1]:
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
            args =  args + (None, None, None, None)

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

        start_time = time.perf_counter()
        for idx in range(*self.exec_map):
            is_decoder = ~self.encoder_mask[idx]
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            # print(type(self.pipe[idx]))
            # end_time = time.perf_counter()
            # print(idx, "layer infetence time", (end_time-start_time)*1000)
            # start_time = time.perf_counter()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if extended_attention_mask is None:
                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, input_shape, self.device, False
                )

            if is_decoder and input_shape[1] != 1: # self.encoder_mask[idx - 1]:
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
                encoder_hidden_states = self.call_layer_sync(idx, input_ids)
            elif self.embed_mask[idx] and is_decoder:
                # print("embed_mask decoder")
                decoder_hidden_states = self.call_layer_sync(idx, input_ids)
            elif self.ln_mask[idx] and not is_decoder:
                # print("ln_mask encoder")
                encoder_hidden_states = self.call_layer_sync(
                    idx, encoder_hidden_states
                )
            elif self.ln_mask[idx] and is_decoder:
                # print("ln_mask decoder")
                decoder_hidden_states = self.call_layer_sync(
                    idx, decoder_hidden_states
                )
            elif is_decoder:
                # print("block decoder")
                # print(decoder_hidden_states, encoder_hidden_states)
                layer_outputs = self.call_layer_sync(
                    idx,
                    decoder_hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                )
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                print(layer_outputs)
                decoder_hidden_states = layer_outputs[0]
                position_bias = layer_outputs[2]
                encoder_decoder_position_bias = layer_outputs[3]
            elif not is_decoder:
                # print("block encoder")
                layer_outputs = self.call_layer_sync(
                    idx,
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
