from collections import namedtuple

import chex
import numpy as np
import pandas as pd
from absl import logging
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from sklearn.preprocessing import LabelEncoder

named_dataset = namedtuple("named_dataset", "y")


class _DataLoader:
    def __init__(self, num_batches, idxs, get_batch):
        self.num_batches = num_batches
        self.idxs = idxs
        self.num_samples = len(idxs)
        self.get_batch = get_batch

    def __call__(self, idx, idxs=None):
        if idxs is None:
            idxs = self.idxs
        return self.get_batch(idx, idxs)


def _rolling_window(array, window_size, freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(
        array, shape=shape, strides=strides
    )
    seqs = rolled[np.arange(0, shape[0], freq)]
    return seqs


def read_data(locations_file, output_size):
    logging.info(f"reading data from: {locations_file}")

    def _encoder(values):
        values_enc = LabelEncoder().fit(values)
        values = values_enc.transform(values)
        return values, values_enc

    def _preprocess(df):
        for f in ["location_id"]:
            df[f], _ = _encoder(df[f].values)
        return df

    df = pd.read_csv(locations_file)
    df = _preprocess(df)

    users = np.unique(df.user_id.values)
    seqs = [None] * len(users)

    for i, user in enumerate(users):
        user_df = df[df.user_id == user]
        els = user_df.location_id.values
        seqs[i] = _rolling_window(els, output_size, 1)
    seqs = np.vstack(seqs)
    unique_locations = sorted(np.unique(df.location_id))

    def fn():
        return named_dataset(seqs)

    logging.info("finished reading data")
    return df, fn, unique_locations


def _as_batch_iterator(
    rng_key: chex.PRNGKey, data: namedtuple, batch_size, shuffle
):
    n = data.y.shape[0]
    if n < batch_size:
        num_batches = 1
        batch_size = n
    elif n % batch_size == 0:
        num_batches = int(n // batch_size)
    else:
        num_batches = int(n // batch_size) + 1

    idxs = jnp.arange(n)
    if shuffle:
        idxs = jr.permutation(rng_key, idxs)

    def get_batch(idx, idxs=idxs):
        start_idx = idx * batch_size
        step_size = jnp.minimum(n - start_idx, batch_size)
        ret_idx = lax.dynamic_slice_in_dim(idxs, idx * batch_size, step_size)
        batch = {
            name: lax.index_take(array, (ret_idx,), axes=(0,))
            for name, array in zip(data._fields, data)
        }
        return batch

    return _DataLoader(num_batches, idxs, get_batch)


# pylint: disable=missing-function-docstring
def as_batch_iterators(*, rng_key, data, batch_size, split, shuffle):
    logging.info("converting to iterator objects")
    n = data.y.shape[0]
    n_train = int(n * split)
    if shuffle:
        data = named_dataset(
            *[
                jr.permutation(rng_key, el, independent=False)
                for _, el in enumerate(data)
            ]
        )
    y_train = named_dataset(*[el[:n_train, :] for el in data])
    y_val = named_dataset(*[el[n_train:, :] for el in data])
    train_rng_key, val_rng_key = jr.split(rng_key)

    train_itr = _as_batch_iterator(train_rng_key, y_train, batch_size, shuffle)
    val_itr = _as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

    return train_itr, val_itr
