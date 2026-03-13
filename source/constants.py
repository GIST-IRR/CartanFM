from torch.optim import Adam, SGD

"""Project-wide constants for architecture defaults and optimizer mapping."""

DATA_HIDDEN_DIM = {
    # Fully-connected hidden dimensions used after CNN feature extraction.
    "dsprites": [256, 128],
    "shapes3d": [256, 256],
    "mpi3d": [256, 256],
}

OPTIMIZER = {
    # String-to-class map used by CLI argument `--optimizer`.
    "adam": Adam,
    "sgd": SGD,
}
