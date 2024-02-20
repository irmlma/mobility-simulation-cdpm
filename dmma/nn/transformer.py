import dataclasses

import chex
import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp


def _sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
    """
    From: https://github.com/google/flax/blob/main/examples/wmt/models.py
    """

    def init(key, shape, dtype=np.float32):
        d_feature = shape[-1]
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        den = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)

        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        pe[:, : d_feature // 2] = np.sin(position * den)
        pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * den)
        pe = pe[np.newaxis, :, :]
        return jnp.array(pe)

    return init


@dataclasses.dataclass
class _PositionalEncoding(hk.Module):
    max_len: int

    def __call__(self, inputs):
        chex.assert_rank(inputs, 3)
        length = inputs.shape[1]
        pos_emb_shape = (1, self.max_len, inputs.shape[-1])
        pos_embedding = _sinusoidal_init(max_len=self.max_len)(
            None, pos_emb_shape, None
        )
        pe = pos_embedding[:, :length, :]
        return inputs + pe


@dataclasses.dataclass
class EncoderConfig:
    num_encoder_layers: int
    num_attention_heads: int
    qkv_dim: int
    encoder_num_hidden_layers: int
    encoder_mlp_dim: int
    encoder_dropout_rate: float
    activation: str
    positional_encoding_len: int


class _Encoder(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = getattr(jax.nn, self.config.activation)

    def __call__(self, inputs, t, *, is_training):
        dropout_rate = self.config.encoder_dropout_rate if is_training else 0.0
        initializer = hk.initializers.VarianceScaling(1.0)
        # causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))
        causal_mask = None
        _, _, model_size = inputs.shape

        h = inputs
        for _ in range(self.config.num_encoder_layers):
            h_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True
            )(h)
            h_attn = hk.MultiHeadAttention(
                num_heads=self.config.num_attention_heads,
                key_size=self.config.qkv_dim,
                model_size=model_size,
                w_init=initializer,
            )(h_norm, h_norm, h_norm, mask=causal_mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn

            # this is where the time conditioning happens
            h += hk.Linear(self.config.qkv_dim)(t)

            h_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True
            )(h)
            h_dense = hk.nets.MLP(
                [self.config.encoder_mlp_dim]
                * self.config.encoder_num_hidden_layers
                + [model_size],
                activation=self.activation_fn,
                w_init=initializer,
            )(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)


class _AutoregressiveTransformerEncoder(hk.Module):
    def __init__(self, encoder, embedding_dim, positional_encoding_len):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.positional_encoding_len = positional_encoding_len

    def __call__(self, inputs, t, *, is_training):
        chex.assert_rank(inputs, 3)
        chex.assert_axis_dimension(inputs, 2, self.embedding_dim)

        x = _PositionalEncoding(self.positional_encoding_len)(inputs)
        x = self.encoder(x, t, is_training=is_training)

        chex.assert_rank(x, 3)
        chex.assert_axis_dimension(x, 1, inputs.shape[1])
        chex.assert_axis_dimension(x, 2, self.embedding_dim)
        return x


class Embedding(hk.Module):
    def __init__(self, num_categories, embedding_dim, normalise_embeddings):
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.normalise_embeddings = normalise_embeddings
        self.embedding = hk.Embed(
            vocab_size=self.num_categories,
            embed_dim=self.embedding_dim,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        )

    def __call__(self, inputs):
        chex.assert_rank(inputs, 2)
        inputs = inputs.astype("int32")
        x = self.embedding(inputs)

        if self.normalise_embeddings:
            norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
            x = x / norm
        return x

    @property
    def weights(self):
        embs = self.embedding.embeddings
        if self.normalise_embeddings:
            norm = jnp.linalg.norm(embs, axis=-1, keepdims=True)
            embs = embs / norm
        return embs


def embedder(num_categories, config):
    return Embedding(
        num_categories, config.embedding_dim, config.normalise_embeddings
    )


def transformer_encoder(config):
    encoder = _Encoder(EncoderConfig(**config.encoder))
    transformer = _AutoregressiveTransformerEncoder(
        encoder,
        config.embedding.embedding_dim,
        config.encoder.positional_encoding_len,
    )
    return transformer
