from datetime import datetime

import jax
from absl import app, flags
from jax import random as jr
from ml_collections import config_flags

from cdpm._src.data import as_batch_iterators, read_data
from cdpm._src.model import make_model
from cdpm._src.train import train

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "training configuration", lock_config=False
)
flags.DEFINE_string(
    "infile", None, "the file with the sequence locations per user",
)
flags.DEFINE_string(
    "outfolder", None, "out directory, i.e., place where results are written to"
)
flags.mark_flags_as_required(["config", "infile", "outfolder"])


def main(argv):
    del argv
    _, data_fn, unique_locations = read_data(
        FLAGS.infile, FLAGS.config.output_size
    )
    del _
    FLAGS.config.num_categories = len(unique_locations)

    train_iter, val_iter = as_batch_iterators(
        rng_key=jr.PRNGKey(FLAGS.config.data.rng_key),
        data=data_fn(),
        batch_size=FLAGS.config.training.batch_size,
        split=FLAGS.config.training.train_val_split,
        shuffle=FLAGS.config.training.shuffle_data,
    )

    tm = datetime.now().strftime("%Y-%m-%d-%H%M")
    run_name = f"{tm}-cdpm"

    model = make_model(FLAGS.config)
    _ = train(
        FLAGS=FLAGS,
        run_name=run_name,
        config=FLAGS.config,
        batch_fns=(train_iter, val_iter),
        model=model,
        rng_key=FLAGS.config.training.rng_key,
    )


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
