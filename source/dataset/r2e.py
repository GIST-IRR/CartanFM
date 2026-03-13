from copy import deepcopy

import torch
import torch.utils.data as data
import numpy as np

from source.dataset.dsprites import DSprites
from source.dataset.shapes3d import Shapes3d
from source.dataset.mpi3d import MPI3d
from typing import Tuple


"""Train/test split builders for random-to-extrapolation (r2e) settings."""


def r2e_dsprites(case: int, dataset: DSprites) -> Tuple[DSprites, DSprites]:
    """Create dSprites r2e split by masking case-specific unseen regions."""

    drop_list = list()
    if case == 0:
        shape = 1
        scale = 0
        for orientation in range(14, 28, 1):
            for x in range(20, 32, 1):
                for y in range(20, 32, 1):
                    to_drop = np.array([shape, scale, orientation, x, y])
                    drop_list.append(dataset.factor_to_idx(to_drop))
    elif case == 1:
        scale = 0
        orientation = 0
        for shape in range(3):
            for x in range(0, 8, 1):
                for y in range(0, 8, 1):
                    to_drop = np.array([shape, scale, orientation, x, y])
                    drop_list.append(dataset.factor_to_idx(to_drop))

    elif case == 2:
        shape = 2
        orientation = 0
        for scale in range(0, 6, 1):
            for x in range(16, 32, 1):
                for y in range(16, 32, 1):
                    to_drop = np.array([shape, scale, orientation, x, y])
                    drop_list.append(dataset.factor_to_idx(to_drop))

    drop_list = np.array(drop_list)

    # Train split excludes the held-out region.
    train_dataset = deepcopy(dataset)
    

    train_dataset.data = np.delete(train_dataset.data, drop_list, axis=0)
    train_dataset.latents_values = np.delete(
        train_dataset.latents_values, drop_list, axis=0
    )

    # Test split keeps only the held-out region.
    test_dataset = dataset
    mask = np.zeros(len(dataset), dtype=bool)
    mask[drop_list] = True
    test_dataset.data = test_dataset.data[mask]
    test_dataset.latents_values = test_dataset.latents_values[mask]

    return train_dataset, test_dataset


def r2e_shape3d(case: int, dataset: Shapes3d) -> Tuple[Shapes3d, Shapes3d]:
    """Create 3D Shapes r2e split by masking case-specific unseen regions."""
    drop_list = list()
    if case == 0:
        scale = 7
        shape = 1
        orientation = 7
        for floor_hue in range(6, 10, 1):
            for wall_hue in range(6, 10, 1):
                for object_hue in range(6, 10, 1):
                    to_drop = np.array(
                        [
                            floor_hue,
                            wall_hue,
                            object_hue,
                            scale,
                            shape,
                            orientation,
                        ]
                    )
                    drop_list.append(dataset.factor_to_idx(to_drop))
    elif case == 1:
        shape = 2
        scale = 7
        orientation = 7
        for floor_hue in range(0, 6, 1):
            for wall_hue in range(0, 6, 1):
                for object_hue in range(0, 6, 1):
                    to_drop = np.array(
                        [
                            floor_hue,
                            wall_hue,
                            object_hue,
                            scale,
                            shape,
                            orientation,
                        ]
                    )
                    drop_list.append(dataset.factor_to_idx(to_drop))
    elif case == 2:
        scale = 0
        orientation = 0
        object_hue = 0
        for floor_hue in range(0, 6, 1):
            for wall_hue in range(6, 10, 1):
                for shape in range(2):
                    to_drop = np.array(
                        [
                            floor_hue,
                            wall_hue,
                            object_hue,
                            scale,
                            shape,
                            orientation,
                        ]
                    )
                    drop_list.append(dataset.factor_to_idx(to_drop))

    drop_list = np.array(drop_list)

    # Train split excludes the held-out region.
    train_dataset = deepcopy(dataset)
    
    train_dataset.data.sort()
    

    train_dataset.data = np.delete(train_dataset.data, drop_list, axis=0).tolist()
    train_dataset.latents_values = np.delete(
        train_dataset.latents_values, drop_list, axis=0
    )

    # Test split keeps only the held-out region.
    test_dataset = dataset
    test_dataset.data.sort()

    mask = np.zeros(len(dataset), dtype=bool)
    mask[drop_list] = True
    test_dataset.data = np.array(test_dataset.data)[mask].tolist()
    test_dataset.latents_values = test_dataset.latents_values[mask]

    return train_dataset, test_dataset
