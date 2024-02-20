import pickle

import jax
from absl import app, flags, logging
from jax import random as jr
from ml_collections import config_flags

from dmma.data import read_data
from dmma.model import make_model
from dmma.simulate import simulate
from dmma.train import train

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "training configuration", lock_config=False
)
flags.DEFINE_enum("mode", "train", ["train", "simulate"], "command to run")
flags.DEFINE_string(
    "infile",
    None,
    "csv-separated input file containing pen-ultimate layer features",
)
flags.DEFINE_string("checkpoint", None, "pickle file with training parameters")
flags.DEFINE_string("outfile", None, "name of the output file")
flags.DEFINE_integer("n_seqs", 10, "number of generated sequences")
flags.DEFINE_integer("len_seqs", 50, "length of generated sequences")
flags.mark_flags_as_required(["config", "infile", "outfile"])


def _train():
    config = FLAGS.config
    _, data_fn, unique_locations = read_data(
        FLAGS.infile, FLAGS.config.output_size
    )
    config.num_categories = len(unique_locations)
    model = make_model(config)
    obj = train(
        rng_key=jr.PRNGKey(FLAGS.config.training.rng_key),
        data=data_fn(),
        model=model,
        config=config,
    )
    with open(FLAGS.outfile, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _simulate():
    with open(FLAGS.checkpoint, "rb") as fh:
        obj = pickle.load(fh)
        config = obj["config"]

    model = make_model(config)
    seqs = simulate(
        rng_key=jr.PRNGKey(config.simulation.rng_key),
        params=obj["params"],
        model=model,
        n_samples=FLAGS.n_seqs,
        len_trajectory=FLAGS.len_seqs,
        out_size=config.output_size,
        batch_size=config.simulation.batch_size,
    )

    seqs.tofile(FLAGS.outfile, sep=",")


def run(argv):
    del argv
    if FLAGS.mode == "train":
        _train()
    else:
        _simulate()
    logging.info("success")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    jax.config.config_with_absl()
    app.run(run)
