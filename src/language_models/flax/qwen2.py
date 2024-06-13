"""Flax Qwen2 model."""
from functools import partial
from typing import Optional, Tuple, Union, Literal, List
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec as PS

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers import Qwen2Config



logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2Config"
_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-0.5B-Instruct"

Qwen2_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`Qwen2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16`, or
            `jax.numpy.bfloat16`.

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

Qwen2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)



# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states, n_rep: int):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.expand_dims(hidden_states, 3)
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class FlaxQwen2RMSNorm(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


# class FlaxQwen2RotaryEmbedding(nn.Module):
#     config: Qwen2Config
#     dtype: jnp.dtype = jnp.float32

#     def setup(self):
#         head_dim = self.config.hidden_size // self.config.num_attention_heads
#         self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)

#     def __call__(self, key, query, position_ids):
#         sincos = self.sincos[position_ids]
#         sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

#         key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
#         query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

#         key = jnp.asarray(key, dtype=self.dtype)
#         query = jnp.asarray(query, dtype=self.dtype)

#         return key, query



def precompute_freq_cis(
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        rope_type: Optional[Literal["none", "linear", "dynamic", "yarn", "su",]] = None,
        t_dtype: jnp.dtype = jnp.int32,
        original_max_position_embeddings: Optional[int] = None,
        long_factor: Optional[List[float]] = None,
        short_factor: Optional[List[float]] = None
):
    def _calc_yarn_scaling_factor(scale):
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    def _calc_su_scaling_factor(scale):
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    if t_dtype == jnp.int64:
        jax.config.update("jax_enable_x64", True)

    if rope_type is None or rope_type == "none":
        t = jax.numpy.arange(max_position_embeddings, dtype=t_dtype)
        inv_freq = 1.0 / (
                base ** (jax.numpy.arange(0, dim, 2, dtype=jax.numpy.float32) / dim)
        )
        freq = jax.numpy.einsum("i , j -> i j", t, inv_freq).astype("float32")
        embed = jax.numpy.concatenate((freq, freq), axis=-1)
        return jax.numpy.sin(embed)[:, :], jax.numpy.cos(embed)[:, :]
    elif rope_type == "linear":
        t = jax.numpy.arange(max_position_embeddings, dtype=t_dtype)
        t = t / scaling_factor
        inv_freq = 1.0 / (
                base ** (jax.numpy.arange(0, dim, 2, dtype=jax.numpy.float32) / dim)
        )
        freq = jax.numpy.einsum(
            "i , j -> i j", t, inv_freq
        ).astype("float32")

        embed = jax.numpy.concatenate((freq, freq), axis=-1)
        return jax.numpy.sin(embed)[:, :], jax.numpy.cos(embed)[:, :]
    elif rope_type == "dynamic":
        t = jax.numpy.arange(max_position_embeddings, dtype=t_dtype)
        base = base * (
                scaling_factor - (scaling_factor - 1)
        ) ** (dim / (dim - 2))
        inv_freq = 1.0 / (
                base ** (jax.numpy.arange(0, dim, 2, dtype=jax.numpy.float32) / dim)
        )
        freq = jax.numpy.einsum(
            "i , j -> i j", t, inv_freq
        ).astype("float32")

        embed = jax.numpy.concatenate((freq, freq), axis=-1)
        return jax.numpy.sin(embed)[:, :], jax.numpy.cos(embed)[:, :]
    elif rope_type == "su":
        assert original_max_position_embeddings is not None, "No original max position embeddings is provided"
        if max_position_embeddings > original_max_position_embeddings:
            ext_factors = jnp.array(long_factor, dtype=jnp.float32)
        else:
            ext_factors = jnp.array(short_factor, dtype=jnp.float32)

        inv_freq = 1.0 / (ext_factors * base ** (jnp.arange(0, dim, 2, dtype=t_dtype).astype(jnp.float32) / dim))[None,
                         :, None]
        position_ids = jnp.arange(
            0, max_position_embeddings, dtype="i4"
        ).reshape(1, -1)[:, None, :].astype("float32")
        freqs = (inv_freq @ position_ids).transpose(0, 2, 1)
        scaling_factor = _calc_su_scaling_factor(
            max_position_embeddings / original_max_position_embeddings
        )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb) * scaling_factor
        sin = jnp.sin(emb) * scaling_factor
        return sin[0], cos[0]
    elif rope_type == "yarn":
        assert original_max_position_embeddings is not None, "No original max position embeddings is provided"
        if max_position_embeddings > original_max_position_embeddings:
            ext_factors = jnp.array(long_factor, dtype=jnp.float32)
        else:
            ext_factors = jnp.array(short_factor, dtype=jnp.float32)

        inv_freq = 1.0 / (
                                 ext_factors
                                 * base ** (jnp.arange(0, dim, 2, dtype=t_dtype).astype(jnp.float32) / dim)
                         )[None, :, None]
        position_ids = jnp.arange(
            0, max_position_embeddings, dtype="i4"
        ).reshape(1, -1)[:, None, :].astype("float32")
        freqs = (inv_freq @ position_ids).transpose(0, 2, 1)
        scaling_factor = _calc_yarn_scaling_factor(
            max_position_embeddings / original_max_position_embeddings
        )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb) * scaling_factor
        sin = jnp.sin(emb) * scaling_factor
        return sin[0], cos[0]
    else:
        raise "wrong rope type has been given"
    
class FlaxQwen2RotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, query, key, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, :, None, :]
        cos = cos[position_ids][:, :, None, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query = apply_rotary_pos_emb(query, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)

class FlaxQwen2Attention(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=True, dtype=self.dtype)
        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=True, dtype=self.dtype)
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=True, dtype=self.dtype)
        self.o_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)

        max_causal_length = getattr(config, "freq_max_position_embeddings", "max_position_embeddings")
        casual_mask = make_causal_mask(jnp.ones((1, max_causal_length), dtype="bool"), dtype="bool")
        self.causal_mask = jnp.triu(casual_mask, k=-config.sliding_window)
        self.rotary_emb = FlaxQwen2RotaryEmbedding(dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        init_cache: bool = False,
        freq_cis: Tuple[chex.Array, chex.Array] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states, self.num_heads)
        key_states = self._split_heads(key_states, self.num_key_value_heads)
        value_states = self._split_heads(value_states, self.num_key_value_heads)

        query_length, key_length = query_states.shape[1], key_states.shape[1]
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        key_states, query_states = self.rotary_emb(key_states, query_states, freq_cis, position_ids)

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
        value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            deterministic=deterministic,
            dropout_rate=self.config.attention_dropout,
            dtype=attention_dtype,
        )

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs




class FlaxQwen2MLP(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.act = jax.nn.silu # ACT2FN[self.config.hidden_act]

        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


class FlaxQwen2DecoderLayer(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = FlaxQwen2RMSNorm(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxQwen2RMSNorm(self.config, dtype=self.dtype)
        
        grad_ckpt = getattr(self.config, "gradient_checkpointing", "")
        if grad_ckpt:
            policy = get_gradient_checkpoint_policy(grad_ckpt)
            print("Apply gradient checkpointing to FlaxQwen2DecoderLayer")
            attn_cls = nn.remat(FlaxQwen2Attention, policy=policy, static_argnums=(4,5,6)) 
            mlp_cls = nn.remat(FlaxQwen2MLP, policy=policy) 
        else:
            attn_cls = FlaxQwen2Attention
            mlp_cls = FlaxQwen2MLP

        self.self_attn = attn_cls(self.config, dtype=self.dtype)
        self.mlp = mlp_cls(self.config, dtype=self.dtype)


    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        freq_cis: Tuple[chex.Array, chex.Array] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            deterministic,
            output_attentions,
            init_cache,
            freq_cis,
        )
        # residual connection
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        return (hidden_states,) + outputs[1:]


# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel with GPTNeo->Qwen2, GPT_NEO->Qwen2, transformer->model
class FlaxQwen2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Qwen2Config
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Qwen2Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(Qwen2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if "params" not in params:
            inputs = {"params": params or self.params}
        else:
            inputs = params

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxQwen2Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxQwen2LayerCollection(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = [
            FlaxQwen2DecoderLayer(self.config, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        freq_cis: Tuple[chex.Array, chex.Array] = None,
        return_dict: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                freq_cis=freq_cis,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxQwen2Module` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxQwen2Module(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        self.layers = FlaxQwen2LayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxQwen2RMSNorm(self.config, dtype=self.dtype)

        initial_rope_kwargs = dict(
            rope_type="none"
        )
        if hasattr(self.config, "rope_scaling"):
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            initial_rope_kwargs = dict(
                scaling_factor=scaling_factor,
                rope_type=scaling_type
            )

        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=(
                getattr(self.config, "freq_max_position_embeddings", self.config.max_position_embeddings)
            ),
            dim=self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.rope_theta,
            **initial_rope_kwargs
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.embed_tokens(input_ids.astype("i4"))

        batch_size, sequence_length = input_ids.shape
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.cumsum(axis=-1) - 1
            else:
                position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
                
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))
            
        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            freq_cis=self.freq_cis,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


@add_start_docstrings(
    "The bare Qwen2 Model transformer outputting raw hidden-states without any specific head on top.",
    Qwen2_START_DOCSTRING,
)
class FlaxQwen2Model(FlaxQwen2PreTrainedModel):
    module_class = FlaxQwen2Module


append_call_sample_docstring(FlaxQwen2Model, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


class FlaxQwen2ForCausalLMModule(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxQwen2Module(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings(
    """
    The Qwen2 Model transformer with a language modeling head (linear layer) on top.
    """,
    Qwen2_START_DOCSTRING,
)
# Copied from transformers.models.gptj.modeling_flax_gptj.FlaxGPTJForCausalLM with GPTJ->Qwen2
class FlaxQwen2ForCausalLM(FlaxQwen2PreTrainedModel):
    module_class = FlaxQwen2ForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Qwen2 uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


append_call_sample_docstring(FlaxQwen2ForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)


class FlaxQwen2ForSequenceClassificationModule(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.model = FlaxQwen2Module(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.score = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        prediction = self.score(hidden_states)

        batch_size, sequence_length = input_ids.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    jnp.equal(input_ids, self.config.pad_token_id)
                    .astype(jnp.int32)
                    .argmax(-1)
                    - 1
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = prediction[jnp.arange(batch_size), sequence_lengths]


        if return_dict:
            return FlaxSequenceClassifierOutput(
                logits=pooled_logits, hidden_states=hidden_states
            )
        else:
            return (pooled_logits,)


class FlaxQwen2ForSequenceClassification(FlaxQwen2PreTrainedModel):
    module_class = FlaxQwen2ForSequenceClassificationModule