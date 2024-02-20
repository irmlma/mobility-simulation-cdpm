# Synthetic location trajectories generation using categorical diffusion models

[![ci](https://github.com/irmlma/mobility-simulation-cdpm/actions/workflows/ci.yaml/badge.svg)](https://github.com/irmlma/mobility-simulation-cdpm/actions/workflows/ci.yaml)
[![arXiv](https://img.shields.io/badge/arXiv-2402.12242-b31b1b.svg)](https://arxiv.org/abs/2402.12242)

## About

This repository contains library code for training a categorical diffusion probabilistic model for generation of synthetic location trajectories.

## Installation

To install the latest GitHub <TAG>, just call the following on the command line:

```bash
docker build https://github.com/irmlma/ mobility-simulation-cdpm.git#<TAG> -t dmma
```

where <TAG> is, e.g., `v0.1.0`.

## Example usage

Having installed as described above you can train and generate trajectories using the provided Docker image.

First test the container using:

```bash
docker run dmma --help
```

Train the model using the provided config file via:
dock
```bash
docker run -v <<some path>>:/mnt \
  dmma /
  --mode=train \
  --config=/mnt/<<config.py>> \
  --infile=/mnt/<<train_dataset.csv>> \
  --outfile=/mnt/<<outfile.pkl>>
```

where
- `<<some path>` is a local path you want to mount to `/mnt/` to make it accessible to Docker,
- `<<config.py>>` is a config file that is following the template in `configs/config.py`,
- `<<train_dataset.csv>>` is a comma-separated file of numerical values which correspond to the features obtained from transforming inputs through a neural network,
- `<<outfile.pkl>>` is the outfile to which parameter and meta data is saved.

To simulate some data, use

```bash
docker run -v <<some path>>:/mnt \
  --mode=simulate \
  --config=/mnt/<<config.py>> \
  --outfile=/mnt/<<outfile.csv>> \
  --checkpoint=/mnt/<<checkpoint>> \
  --n_seqs=10 \
  --len_seqs=32
```

where
- `<<some path>` is the same as above,
- `<<config.py>>` is the same as above,
- `<<test_dataset.csv>>` is a data set for which you want to evaluate if it is OoD,
- `<<outfile.pkl>>` is the name of the outfile,
- `<<checkpoint>>` is the parameter file obtained through the training (i.e., in this case `<<outfolder>>/params.pkl`),
- `n_seqs` determines the number of generated location trajectories,
- `len_seqs` is the length of each trajectory.

## Citation

If you find our work relevant to your research, please consider citing

```
@article{dirmeier2024cdpm,
  title={Synthetic location trajectories generation using categorical diffusion models},
  author={Simon Dirmeier and Ye Hong and Fernando Perez-Cruz},
  year={2024},
  journal={arXiv preprint arXiv:2402.12242}
}
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
