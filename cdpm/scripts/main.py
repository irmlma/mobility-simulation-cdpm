import pickle
from datetime import datetime

import jax
import pandas as pd
from absl import app, flags
from jax import random as jr
from ml_collections import config_flags
from sklearn.svm._libsvm import predict

from cdpm.data import read_data
from cdpm.model import make_model
from cdpm.train import train

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "training configuration", lock_config=False
)
flags.DEFINE_enum("mode", "train", ["train", "predict"], "command to run")
flags.DEFINE_string(
    "infile",
    None,
    "csv-separated input file containing pen-ultimate layer features",
)
flags.DEFINE_string("checkpoint", None, "pickle file with training parameters")
flags.DEFINE_string("outfile", None, "name of the output file")
flags.mark_flags_as_required(["config", "infile", "outfile"])


def _train():
    _, data_fn, unique_locations = read_data(
        FLAGS.infile, FLAGS.config.output_size
    )
    FLAGS.config.num_categories = len(unique_locations)

    model = make_model(FLAGS.config)
    _ = train(
        rng_key=jr.PRNGKey(FLAGS.config.training.rng_key),
        data=data_fn(),
        model=model,
        config=FLAGS.config,
        outfile=FLAGS.outfile,
    )

    return 0


def _predict():
    with open(FLAGS.checkpoint, "rb") as fh:
        obj = pickle.load(fh)
    config = obj["config"]

    features = pd.read_csv(FLAGS.infile).values
    model = make_model(config.model, features.shape[1])
    lps = predict(
        rng_key=config.prediction.rng_seq_key,
        params=obj["params"],
        model=model,
        data=features,
        batch_size=config.prediction.batch_size,
    )

    with open(FLAGS.outfile, "wb") as fh:
        pickle.dump(lps, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


def run(argv):
    del argv
    if FLAGS.mode == "train":
        _train()
    else:
        _predict()
    logging.info("success")
    return 0


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    jax.config.config_with_absl()
    app.run(run)
