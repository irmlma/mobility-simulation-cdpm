# Synthetic location trajectories generation using categorical diffusion models

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/irmlma/mobility-simulation-cdpm/actions/workflows/ci.yaml/badge.svg)](https://github.com/irmlma/mobility-simulation-cdpm/actions/workflows/ci.yaml)

## About

This repository implements a categorical diffusion probabilistic model for generation of synthetic location trajectories.

## Installation

To install the latest GitHub <TAG>, just call the following on the command line:

```bash
docker build https://github.com/irmlma/ mobility-simulation-cdpm.git#<TAG> -t uqma
```

where <TAG> is, e.g., `v0.1.0`.

## Example usage

Having installed as described above you can train and generate trajectories using the provided Docker image.

First test the container using:

```bash
docker run dmma --help
```

Train the model using the provided config file via:
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

To make predictions for epistemic uncertainty estimates, call:
```bash
docker run -v <<some path>>:/mnt \
  --mode=predict \
  --config=/mnt/<<config.py>> \
  --infile=/mnt/<<test_dataset.csv>> \
  --outfile=/mnt/<<outfile.pkl>> \
  --checkpopint=/mnt/<<checkpoint>>
```

where
- `<<some path>` is the same as above,
- `<<config.py>>` is the same as above,
- `<<test_dataset.csv>>` is a data set for which you want to evaluate if it is OoD,
- `<<outfile.pkl>>` is the name of the outfile,
- `<<checkpoint>>` is the parameter file obtained through the training (i.e., in this case `<<outfolder>>/params.pkl`).

```

## Citation

If you find our work relevant to your research, please consider citing

```
@article{dirmeier2024cdpm,
  title={Synthetic location trajectories generation using categorical diffusion models},
  author={Simon Dirmeier and Ye Hong and Fernando Perez-Cruz},
  year={2024}
}
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd@pm.me">sfyrbnd@pm.me</a>
