import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr

from dmma.data import as_batch_iterators
from dmma.early_stopping import EarlyStopping


def _get_optimizer(config):
    warmup_schedule = optax.linear_schedule(
        init_value=config.warmup.start_learning_rate,
        end_value=config.params.learning_rate,
        transition_steps=config.num_warmup_steps,
    )
    if config.learning_rate_decay.decay_type == "cosine":
        decay_schedule = optax.cosine_decay_schedule(
            init_value=config.adamw_learning_rate,
            decay_steps=config.num_train_steps - config.num_warmup_steps,
        )
    elif config.learning_rate_decay.decay_type == "exponential":
        decay_schedule = optax.exponential_decay(
            decay_rate=config.adamw_exponential_decay_rate,
            init_value=config.adamw_learning_rate,
            end_value=config.end_learning_rate,
            transition_steps=config.num_train_steps - config.num_warmup_steps,
        )
    elif config.learning_rate_decay.decay_type == "linear":
        decay_schedule = optax.linear_schedule(
            init_value=config.params.learning_rate,
            end_value=config.learning_rate_decay.end_learning_rate,
            transition_steps=config.num_train_steps - config.num_warmup_steps,
        )
    else:
        raise ValueError("schedule not supported")

    learning_rate = config.params.learning_rate
    if config.warmup.do_warmup_schedule:
        learning_rate = warmup_schedule
    if config.learning_rate_decay.do_learning_rate_decay_schedule:
        learning_rate = decay_schedule
    if (
        config.warmup.do_warmup_schedule
        and config.learning_rate_decay.do_learning_rate_decay_schedule
    ):
        if config.post_decay_schedule.do_constant_learning_rate_after_schedule:
            learning_rate = optax.join_schedules(
                [
                    warmup_schedule,
                    decay_schedule,
                    lambda x: config.params.learning_rate,
                ],
                boundaries=[
                    config.num_warmup_steps,
                    config.num_train_steps - config.num_warmup_steps,
                ],
            )
        else:
            learning_rate = optax.join_schedules(
                [warmup_schedule, decay_schedule],
                boundaries=[config.num_warmup_steps],
            )

    optimizer = optax.adamw(
        learning_rate,
        b1=config.params.b1,
        b2=config.params.b2,
        eps=config.params.eps,
        weight_decay=config.params.weight_decay,
    )

    if config.gradient_transform.do_gradient_clipping:
        optimizer = optax.chain(
            optax.clip(config.gradient_transform.gradient_clipping),
            optimizer,
        )

    return optimizer


def train(rng_key, *, data, model, config):
    train_iter, val_iter = as_batch_iterators(
        rng_key=jr.PRNGKey(config.data.rng_key),
        data=data,
        batch_size=config.training.batch_size,
        split=config.training.train_val_split,
        shuffle=config.training.shuffle_data,
    )

    init_key, rng_key = jr.split(rng_key)
    params = model.init(
        init_key, method="evidence", is_training=True, **train_iter(0)
    )
    optimizer = _get_optimizer(config.optimizer)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, rng, **batch):
        def loss_fn(params):
            evidence = model.apply(
                params, rng, method="evidence", is_training=True, **batch
            )
            return -jnp.mean(evidence)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    losses = np.zeros([config.training.n_iter, 2])
    best_params = params
    best_loss = np.inf
    best_itr = 0
    early_stop = EarlyStopping(
        min_delta=config.training.early_stopping_delta,
        patience=config.training.early_stopping_patience,
    )
    train_key, rng_key = jr.split(rng_key)

    logging.info("training model")
    for i in range(config.training.n_iter):
        epoch_key = jr.fold_in(train_key, i)
        perm_key, epoch_key = jr.split(epoch_key)

        train_loss = 0.0
        idxs = jr.permutation(perm_key, train_iter.idxs)
        for j in range(train_iter.num_batches):
            batch = train_iter(j, idxs)
            batch_key, epoch_key = jr.split(epoch_key)
            batch_loss, params, opt_state = step(
                params, opt_state, batch_key, **batch
            )
            train_loss += batch_loss * (
                batch["y"].shape[0] / train_iter.num_samples
            )

        val_key, epoch_key = jr.split(epoch_key)
        validation_loss = _validation_loss(val_key, params, model, val_iter)
        losses[i] = jnp.array([train_loss, validation_loss])
        logging.info(
            f"epoch {i} train/val elbo: {train_loss}/{validation_loss}"
        )
        early_stop.update(validation_loss)
        if validation_loss is jnp.nan:
            logging.warning("found nan validation loss. breaking")
            break
        if early_stop.should_stop:
            logging.info("Met early stopping criterion, breaking...")
            break
        if validation_loss < best_loss:
            logging.info("new best loss found at epoch: %d", i)
            best_params = params.copy()
            best_loss = validation_loss
            best_itr = i

    losses = jnp.vstack(losses)[: (i + 1), :]
    return {
        "params": best_params,
        "loss": best_loss,
        "itr": best_itr,
        "losses": losses,
        "config": config,
    }


def _validation_loss(rng_key, params, model, val_iter):
    @jax.jit
    def loss_fn(rng, **batch):
        lp = model.apply(
            params, rng, method="evidence", is_training=False, **batch
        )
        return -jnp.mean(lp)

    def body_fn(rng, i):
        batch = val_iter(i)
        loss = loss_fn(rng, **batch)
        return loss * (batch["y"].shape[0] / val_iter.num_samples)

    losses = 0.0
    rngs = jr.split(rng_key, val_iter.num_batches)
    for i in range(val_iter.num_batches):
        losses += body_fn(rngs[i], i)
    return losses
