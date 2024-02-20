import numpy as np
from jax import random as jr


def simulate(
    rng_key, *, params, model, n_samples, len_trajectory, out_size, batch_size
):
    n_itrs = int(np.ceil(n_samples / batch_size))
    n_traj = int(np.ceil(len_trajectory / (out_size / 2)))

    simulated_trajectories = [None] * n_itrs
    for i in range(n_itrs):
        sample_key, rng_key = jr.split(rng_key)
        xs = [None] * n_traj
        for j in range(n_traj):
            kwargs = {}
            if j > 0:
                x_start = xs[j - 1]
                kwargs = {"seed": x_start}
            x = model.apply(
                params,
                jr.fold_in(sample_key, j),
                method="sample",
                sample_shape=(batch_size,),
                **kwargs,
            )
            x = np.asarray(x)
            xs[j] = x[:, int(x.shape[1] / 2) :]
        simulated_trajectories[i] = np.hstack(xs)
    simulated_trajectories = np.vstack(simulated_trajectories)
    return simulated_trajectories[:n_samples, :out_size]
