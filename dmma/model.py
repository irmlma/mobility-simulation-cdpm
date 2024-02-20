import chex
import distrax
import haiku as hk
import jax
import numpy as np
import optax
from absl import logging
from jax import lax
from jax import numpy as jnp
from jax import random

from dmma.nn.score_model import ScoreModel, ScoreModelConfig
from dmma.nn.transformer import embedder, transformer_encoder


def _value_to_log_onehot(value, num_categories):
    value = jax.nn.one_hot(value, num_categories) + 1e-30
    value = jnp.log(value)
    return value


def _cosine_alpha_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=0.001, a_max=0.9999)
    return alphas


def _sqrt_alpha_schedule(timesteps, s=1e-04):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = 1.0 - np.sqrt(x / timesteps + s)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=0.001, a_max=0.9999)
    return alphas


def _linear_alpha_schedule(timesteps, b_min=1e-04, b_max=0.02):
    steps = timesteps + 1
    betas = np.linspace(b_min, b_max, steps)
    alphas = 1.0 - betas
    alphas = np.clip(alphas, a_min=0.001, a_max=0.9999)
    return alphas


def _get_alpha_schedule(schedule_name, timesteps):
    if schedule_name == "cosine":
        alphas = _cosine_alpha_schedule(timesteps)
    elif schedule_name == "sqrt":
        alphas = _sqrt_alpha_schedule(timesteps)
    elif schedule_name == "linear":
        alphas = _linear_alpha_schedule(timesteps)
    else:
        raise ValueError("this schedule does not exist")
    return alphas


class MobilityDPM(hk.Module):
    def __init__(self, config, score_model, embedding_fn, alpha_schedule):
        super().__init__()
        self._config = config
        self._num_categories = config.num_categories
        self._output_size = config.output_size
        self._embedding_dim = config.model.embedding.embedding_dim
        self._n_diffusions = len(alpha_schedule)

        self._predict_z0 = config.model.predict_z0
        self._stop_gradient_at_logits = config.model.stop_gradient_at_logits
        self._use_classier_free_guidance = (
            config.model.score_model.use_classier_free_guidance
        )
        self._use_self_conditioning = (
            config.model.score_model.use_self_conditioning
        )

        self._score_model = score_model
        self._embedding_fn = embedding_fn

        self._alphas = alpha_schedule
        self._betas = jnp.asarray(1.0 - self._alphas)
        self._alphas_bar = jnp.cumprod(self._alphas)
        self._alphas_bar_prev = jnp.append(1.0, self._alphas_bar[:-1])
        self._sqrt_alphas_bar = jnp.sqrt(self._alphas_bar)
        self._sqrt_1m_alphas_bar = jnp.sqrt(1.0 - self._alphas_bar)
        self._sqrt_recip_alphas_bar = jnp.sqrt(jnp.reciprocal(self._alphas_bar))
        self._sqrt_recip_alphas_bar_m1 = jnp.sqrt(
            jnp.reciprocal(self._alphas_bar) - 1.0
        )

        self._q_posterior_mean_lhs = (
            self._betas
            * jnp.sqrt(self._alphas_bar_prev)
            / (1.0 - self._alphas_bar)
        )
        self._q_posterior_mean_rhs = (
            (1.0 - self._alphas_bar_prev)
            * jnp.sqrt(self._alphas)
            / (1.0 - self._alphas_bar)
        )
        self._q_posterior_variance = (
            self._betas
            * (1.0 - self._alphas_bar_prev)
            / (1.0 - self._alphas_bar)
        )

    def __call__(self, method="evidence", **kwargs):
        return getattr(self, method)(**kwargs)

    def evidence(self, y, *, is_training, **kwargs):
        embedding = self._embedding_fn(y)
        z_0 = self.add_noise(embedding, self._sqrt_1m_alphas_bar[0])

        ld = self._diffusion_loss(
            z_0=z_0, embedding=embedding, is_training=is_training, **kwargs
        )
        loss = ld

        if self._config.model.use_prior_loss:
            lpr = self._prior_loss(z_0=z_0)
            loss += lpr
        if self._config.model.use_l0_loss:
            l0 = self._loss_l0(
                embedding=embedding, z_0=z_0, is_training=is_training, **kwargs
            )
            loss += l0
        if self._config.model.use_rounding_loss:
            lr = self._loss_rounding(y=y, z_0=z_0, is_training=is_training)
            loss += lr
        return -loss

    def _diffusion_loss(self, z_0, embedding, is_training, **kwargs):
        t = random.choice(
            key=hk.next_rng_key(),
            a=jnp.arange(1, self._n_diffusions),
            shape=(z_0.shape[0],),
        ).reshape(-1, 1)

        noise = distrax.Normal(jnp.zeros_like(z_0), 1.0).sample(
            seed=hk.next_rng_key()
        )
        z_t = self.q_pred_repram(z_0, t.reshape(-1, 1, 1), noise)

        kwargs = self._maybe_selfcondition(
            z_t=z_t,
            t=t,
            z_0=z_0,
            embedding=lax.stop_gradient(embedding),
            is_training=is_training,
        )
        target = z_0 if self._predict_z0 else noise
        pred = self._score_model(
            z_t,
            t,
            is_training=is_training,
            embedding=lax.stop_gradient(embedding),
            **kwargs,
        )
        loss = jnp.sum(jnp.square(target - pred), axis=[-2, -1])
        return loss

    def _prior_loss(self, z_0):
        zT_mean = self.q_pred_mean(z_0, self._n_diffusions - 1)
        loss = jnp.sum(jnp.square(zT_mean), axis=[-2, -1])
        return loss

    def _maybe_selfcondition(self, z_t, t, z_0, embedding, is_training):
        if self._use_self_conditioning:
            zeros = jnp.zeros_like(z_t)
            pred = self._score_model(
                z_t,
                t,
                is_training=is_training,
                embedding=lax.stop_gradient(embedding),
                self_condition=zeros,
            )
            pred = lax.stop_gradient(pred)
            ret = jnp.where(
                random.uniform(hk.next_rng_key()) > 0.5, zeros, pred
            )
            return {"self_condition": ret}
        return {}

    def _loss_l0(self, embedding, z_0, is_training, **kwargs):
        t = jnp.zeros(embedding.shape[0], dtype=jnp.int32).reshape(-1, 1)
        noise = distrax.Normal(jnp.zeros_like(embedding), 1.0).sample(
            seed=hk.next_rng_key()
        )
        z_t = self.q_pred_repram(z_0, t.reshape(-1, 1, 1), noise)
        kwargs = self._maybe_selfcondition(
            z_t=z_t,
            t=t,
            z_0=z_0,
            embedding=lax.stop_gradient(embedding),
            is_training=is_training,
        )
        pred = self._score_model(
            z_t,
            t,
            is_training=is_training,
            embedding=lax.stop_gradient(embedding),
            **kwargs,
        )
        if not self._predict_z0:
            pred = self._get_z0_from_prediction(z_t, t, pred)
        loss = jnp.sum(jnp.square(embedding - pred), axis=[-2, -1])
        return loss

    def _loss_rounding(self, y, z_0, is_training, **kwargs):
        loss = optax.softmax_cross_entropy_with_integer_labels(
            self._as_logits(z_0), y
        )
        chex.assert_shape(loss, y.shape)
        loss = jnp.sum(loss, axis=-1)
        return loss

    def _as_logits(self, z_0):
        logits = self._embedding_fn.weights.T
        if self._stop_gradient_at_logits:
            logits = lax.stop_gradient(logits)
        logits = z_0 @ logits
        return logits

    def _likelihood(self, z_0):
        logits = self._as_logits(z_0)
        lp = distrax.Categorical(logits=logits)
        return lp

    def q_posterior_mean_variance(self, z_0, z_t, t):
        t = jnp.atleast_1d(t).reshape(-1)
        posterior_mean = (
            self._q_posterior_mean_lhs[t].reshape(-1, 1, 1) * z_0
            + self._q_posterior_mean_rhs[t].reshape(-1, 1, 1) * z_t
        )
        posterior_variance = self._q_posterior_variance[t]
        return posterior_mean, posterior_variance

    def q_pred_mean(self, z_0, t):
        return self._sqrt_alphas_bar[t] * z_0

    def q_pred_repram(self, z_0, t, noise):
        return self.q_pred_mean(z_0, t) + self._sqrt_1m_alphas_bar[t] * noise

    def add_noise(self, z, scale):
        noise = distrax.Normal(jnp.zeros_like(z), 1.0).sample(
            seed=hk.next_rng_key()
        )
        return z + scale * noise

    def p_mean_variance(self, z_t, t, pred):
        z_0 = self._get_z0_from_prediction(z_t, t, pred)
        mean, _ = self.q_posterior_mean_variance(z_0, z_t, t)
        t = jnp.atleast_1d(t).reshape(-1)
        variance = self._betas[t]
        return mean, variance

    def sample(
        self, sample_shape=(64,), *, seed=None, is_training=False, **kwargs
    ):
        def _fn(i, val):
            z_t, z_pred, mask, embedding = val
            t = jnp.atleast_1d(self._n_diffusions - i)
            noise = distrax.Normal(jnp.zeros_like(z_t), 1.0).sample(
                seed=hk.next_rng_key()
            )

            z_pred = self._score_model(
                z_t,
                t,
                is_training=is_training,
                embedding=embedding,
                mask=mask,
                self_condition=z_pred,
            )

            mean, var = self.p_mean_variance(z_t, t, z_pred)
            # TODO(simon): check thuis
            z_t = mean + noise * jnp.sqrt(var)
            z_t = jnp.where(
                mask[None, :, None].astype(jnp.bool_), embedding, z_t
            )

            return z_t, z_pred, mask, embedding

        prior = distrax.Normal(
            jnp.zeros(
                (self._output_size, self._config.model.embedding.embedding_dim)
            ),
            1.0,
        )

        if seed is None:
            z_T = prior.sample(
                seed=hk.next_rng_key(), sample_shape=sample_shape
            )
        else:
            z_T = prior.sample(
                seed=hk.next_rng_key(), sample_shape=(seed.shape[0],)
            )
        z_pred = jnp.zeros_like(z_T)

        if seed is not None:
            mask = jnp.concatenate(
                [jnp.ones(seed.shape[1]), jnp.zeros(seed.shape[1])]
            )
            half_embedding = self._embedding_fn(seed)
            zeros = jnp.zeros_like(half_embedding)
            embedding = jnp.concatenate([half_embedding, zeros], axis=1)
            z_T = jnp.where(
                mask[None, :, None].astype(jnp.bool_), embedding, z_T
            )
        else:
            embedding = jnp.zeros_like(z_T)
            mask = jnp.zeros(self._output_size)

        z_0, *_ = hk.fori_loop(
            0, self._n_diffusions, _fn, (z_T, z_pred, mask, embedding)
        )

        y = self._likelihood(z_0).sample(seed=hk.next_rng_key())
        y = jnp.asarray(y)
        y = y.at[:, mask == 1].set(seed)

        return y

    def _predict_z0_from_noise(self, z_t, t, eps):
        chex.assert_equal_shape([z_t, eps])
        t = jnp.atleast_1d(t).reshape(-1)
        z_0 = (
            self._sqrt_recip_alphas_bar[t].reshape(-1, 1, 1) * z_t
            - self._sqrt_recip_alphas_bar_m1[t].reshape(-1, 1, 1) * eps
        )
        return z_0

    def _get_z0_from_prediction(self, z_t, t, pred, **kwargs):
        if not self._predict_z0:
            pred = self._predict_z0_from_noise(z_t, t, pred)
        return pred


def make_model(config):
    logging.info("instantiating model")

    def _fn(method, **kwargs):
        embedding = embedder(config.num_categories, config.model.embedding)
        transformer = transformer_encoder(config.model)
        score_model = ScoreModel(
            ScoreModelConfig(
                **config.model.score_model, num_categories=config.num_categories
            ),
            transformer,
        )
        return MobilityDPM(
            config,
            score_model,
            embedding,
            _get_alpha_schedule(
                config.model.noise_schedule, config.model.n_diffusions
            ),
        )(method, **kwargs)

    diffusion = hk.transform(_fn)
    return diffusion
