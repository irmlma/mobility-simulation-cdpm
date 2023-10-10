import os
import pickle

import haiku as hk
import jax
import numpy as np
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from jax import jr as jr
from jax import numpy as jnp


def get_optimizer(config):
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


def train(*, FLAGS, run_name, config, batch_fns, model, rng_key, run):
    rng_seq = hk.PRNGSequence(rng_key)
    train_iter, val_iter = batch_fns

    params = model.init(
        next(rng_seq), method="evidence", is_training=True, **train_iter(0)
    )
    optimizer = get_optimizer(config.optimizer)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, rng, **batch):
        def loss_fn(params):
            evidence = model.apply(
                params, rng, method="evidence", is_training=True, **batch
            )
            return -jnp.sum(evidence)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    losses = np.zeros([config.training.n_iter, 2])
    best_params = params
    best_loss = np.inf
    best_itr = 0

    logging.info("training model")
    early_stop = EarlyStopping(
        min_delta=config.training.early_stopping_delta,
        patience=config.training.early_stopping_patience,
    )
    idxs = train_iter.idxs
    for i in range(config.training.n_iter):
        train_loss = 0.0
        idxs = jr.permutation(next(rng_seq), idxs)
        for j in range(train_iter.num_batches):
            batch = train_iter(j, idxs)
            batch_loss, params, opt_state = step(
                params, opt_state, next(rng_seq), **batch
            )
            train_loss += batch_loss
        validation_loss = _validation_loss(params, rng_seq, model, val_iter)
        losses[i] = jnp.array([train_loss, validation_loss])

        logging.info(
            f"epoch {i} train/val elbo: {train_loss}/{validation_loss}"
        )

        _, early_stop = early_stop.update(validation_loss)
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
            save(
                run_name,
                {"params": best_params, "loss": best_loss, "itr": best_itr},
                jnp.vstack(losses)[:i, :],
                FLAGS,
            )
    losses = jnp.vstack(losses)[:i, :]
    return (
        {"params": best_params, "loss": best_loss, "itr": best_itr},
        losses,
        model,
    )


def model_path(FLAGS, run_name):
    outfile = os.path.join(FLAGS.outfolder, f"{run_name}-params.pkl")
    return outfile


def _validation_loss(params, rng_seq, model, val_iter):
    @jax.jit
    def _loss_fn(rng, **batch):
        evidence = model.apply(
            params, rng, method="evidence", is_training=False, **batch
        )
        return -jnp.sum(evidence)

    rngs = jr.split(next(rng_seq), val_iter.num_batches)
    losses = jnp.array(
        [_loss_fn(rngs[j], **val_iter(j)) for j in range(val_iter.num_batches)]
    )
    return jnp.sum(losses)


def save(run_name, params, losses, FLAGS):
    obj = {"params": params, "losses": losses, "config": FLAGS.config}
    outfile = model_path(FLAGS, run_name)
    logging.info("writing params to: %s", outfile)
    with open(outfile, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
