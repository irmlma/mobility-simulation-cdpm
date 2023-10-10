import dataclasses

import chex
import haiku as hk
import jax
from jax import numpy as jnp
from jax import random


@dataclasses.dataclass
class ScoreModelConfig:
    num_categories: int

    n_diffusions: int

    time_embedding_dim: int
    time_embedding_num_hidden_layers: int
    time_embedding_activation: str

    embedding_dim: int
    pre_projection_dim: int
    num_layers_pre_projection_dim: int
    dropout_rate: float
    activation: str

    use_classier_free_guidance: bool
    use_self_conditioning: bool

    prefix_mask_size: int
    random_suffix_mask_size: int
    uses_mask: bool
    use_mask_as_input: bool
    project_mask_to_embedding_dim: bool


class ScoreModel(hk.Module):
    def __init__(self, config, transformer):
        super().__init__()
        self.config = config
        self.activation_fn = getattr(jax.nn, self.config.activation)
        self.transformer = transformer
        self.time_embedding_net = hk.nets.MLP(
            [self.config.time_embedding_dim]
            * self.config.time_embedding_num_hidden_layers,
            activation=getattr(jax.nn, self.config.time_embedding_activation),
        )

    def __call__(self, z, t, *, is_training, embedding=None, **kwargs):
        chex.assert_rank(z, 3)
        chex.assert_axis_dimension(z, 2, self.config.embedding_dim)
        _, output_size, _ = z.shape

        time_embedding = self._embed_time(t)
        if self.config.uses_mask:
            if "mask" not in kwargs:
                prefix_mask_idx = jnp.arange(self.config.prefix_mask_size)
                random_mask_idx = self.config.prefix_mask_size + random.choice(
                    hk.next_rng_key(),
                    output_size - self.config.prefix_mask_size,
                    shape=(self.config.random_suffix_mask_size,),
                    replace=False,
                )
                mask_idxs = jnp.concatenate([prefix_mask_idx, random_mask_idx])

                # where mask==1 is conditioning, where mask==0 will be simulated
                mask = jnp.zeros(output_size, dtype=jnp.bool_)
                mask = mask.at[mask_idxs].set(True)
            else:
                mask = kwargs["mask"]

            if self.config.use_classier_free_guidance and is_training:
                mask = jnp.where(
                    random.uniform(hk.next_rng_key()) > 0.5,
                    mask,
                    jnp.zeros_like(mask),
                )

            mask = mask[None, :, None].astype(jnp.bool_)
            z = jnp.where(jnp.logical_not(mask), z, 0.0)
            embedding = jnp.where(mask, embedding, 0.0)
            z = jnp.concatenate([z, embedding], axis=-1)

            if self.config.use_mask_as_input:
                mask = mask.astype(z.dtype)
                mask = (
                    jnp.tile(mask, [z.shape[0], 1, 1])
                    if not self.config.project_mask_to_embedding_dim
                    else jnp.tile(
                        mask, [z.shape[0], 1, self.config.embedding_dim]
                    )
                )
                z = jnp.concatenate([z, mask], axis=-1)

        if self.config.use_self_conditioning:
            self_condition = kwargs["self_condition"]
            z = jnp.concatenate([z, self_condition], axis=-1)

        z = hk.nets.MLP(
            [self.config.pre_projection_dim]
            * self.config.num_layers_pre_projection_dim
            + [self.config.embedding_dim],
            activation=self.activation_fn,
        )(z)
        z = self.transformer(z, time_embedding, is_training=is_training)
        return z

    def _embed_time(self, t):
        t = t.reshape(
            -1,
        )
        t = _time_embedding(t, self.config.time_embedding_dim)
        t = self.time_embedding_net(t)
        t = t[:, None, :]
        return t


# pre-co
def _time_embedding(timesteps, embedding_dim, dtype=jnp.float32):
    """
    From: https://github.com/google-research/vdm/blob/
          0c60a8979491f56e32f7ed3bca22bd5b506d0fdb/model_vdm.py
    """

    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    return emb
