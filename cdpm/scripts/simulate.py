import os
import pathlib
import pickle

import jax
import numpy as np
import pandas as pd
from absl import app, flags, logging
from jax import random as jr

from cdpm._src.data import read_data
from cdpm._src.model import make_model

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "infile", None, "the file with the sequence locations per user"
)
flags.DEFINE_string(
    "params", None, "out directory, i.e., place where results are written to"
)
flags.DEFINE_boolean("conditional", True, "conditional generation")
flags.DEFINE_string(
    "outfolder", None, "out directory, i.e., place where results are written to"
)
flags.mark_flags_as_required(["params", "infile", "outfolder"])


def _main(argv):
    del argv
    params = pd.read_pickle(FLAGS.params)

    output_size = params["config"].output_size
    df, _, _ = read_data(FLAGS.infile, output_size)
    model = make_model(params["config"])

    sequences = df.groupby("user").head(params["config"].output_size)
    sequences["index"] = np.tile(
        np.arange(output_size), len(np.unique(df["user"].values))
    )
    seed = sequences.pivot(
        index="user", columns="index", values="location_id"
    ).values

    n_samples = 10
    n_trajectory = 30
    simulated_trajectories = [None] * n_samples
    for i in range(n_samples):
        print(i, FLAGS.conditional)
        xs = [None] * n_trajectory
        start_idx = 0
        if FLAGS.conditional:
            xs[0] = seed[:, : int(output_size / 2)]
            start_idx = 1
        for j in range(start_idx, n_trajectory):
            kwargs = {}
            if FLAGS.conditional or j > 0:
                x_start = xs[j - 1]
                kwargs = {"seed": x_start}
            x = model.apply(
                params["params"]["params"],
                jr.fold_in(jr.PRNGKey(i), j),
                method="sample",
                **kwargs,
            )
            x = np.asarray(x)
            xs[j] = x[:, int(output_size / 2) :]
        simulated_trajectories[i] = np.hstack(xs)

    outfile_basefile = pathlib.Path(FLAGS.params).name.replace(
        "params.pkl", f"simulation-conditional={FLAGS.conditional}.pkl"
    )
    outfile = os.path.join(FLAGS.outfolder, outfile_basefile)
    logging.info("writing simulated data to: %s", outfile)
    with open(outfile, "wb") as handle:
        pickle.dump(
            {"simulated_data": simulated_trajectories, "df": df},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logging.info("finished run successfully")


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(_main)
