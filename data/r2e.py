from copy import deepcopy

import torch
import torch.utils.data as data
import numpy as np

from data.dsprites import DSprites
from data.shapes3d import Shapes3d
from data.mpi3d import MPI3d
from typing import Tuple

"""
factors = [
    shape=(square, ellipse, heart),
    scale=(0.5, 0.6, 0.7, 0.8, 0.9, 1),
    orientation=range(40),
    x=range(32),
    y=range(32)
"""

def r2e_dsprites(case: int, dataset: DSprites) -> Tuple[DSprites, DSprites]:

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

    train_dataset = deepcopy(dataset)
    
    train_dataset.data = np.delete(train_dataset.data, drop_list, axis=0)
    train_dataset.latents_values = np.delete(
        train_dataset.latents_values, drop_list, axis=0
    )

    test_dataset = dataset
    mask = np.zeros(len(dataset), dtype=bool)
    mask[drop_list] = True
    test_dataset.data = test_dataset.data[mask]
    test_dataset.latents_values = test_dataset.latents_values[mask]

    return train_dataset, test_dataset

"""
factors = [
    floor_hue=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    wall_hue=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    object_hue=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    scale=(0, 1, 2, 3, 4, 5, 6, 7),
    shape=(sphere, cube, cylinder, oblong),
    orientation=(-30, 30, 4)
"""

def r2e_shape3d(case: int, dataset: Shapes3d) -> Tuple[Shapes3d, Shapes3d]:
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

    train_dataset = deepcopy(dataset)
    
    train_dataset.data.sort()
    
    train_dataset.data = np.delete(train_dataset.data, drop_list, axis=0).tolist()
    train_dataset.latents_values = np.delete(
        train_dataset.latents_values, drop_list, axis=0
    )

    test_dataset = dataset
    test_dataset.data.sort()

    mask = np.zeros(len(dataset), dtype=bool)
    mask[drop_list] = True
    test_dataset.data = np.array(test_dataset.data)[mask].tolist()
    test_dataset.latents_values = test_dataset.latents_values[mask]

    return train_dataset, test_dataset

"""
factors = [
    object_color=(white, green, red, blue, brown, olive), 
    object_shape=(cone, cube, cylinder, hexagonal, pyramid, sphere), 
    object_size=(small, large),
    camera_height=(top, center, bottom),
    background_color=(purple, sea green, salmon)
    horizontal_axis=40, 
    vertical_axis=40
]
case 0: except small white cone with purple background (0, 0, 0, 0, 0, 10~19, 10~19)
case 1: except large red hexagonal with salmon background (2, 3, 1, 1, 2, 0~9, 0~9)
case 2: except small brown pyramid with sea green background (4, 4, 0, 2, 1, 30~39, 30~39)
"""
def r2e_mpi3d(case: int, dataset: MPI3d) -> Tuple[MPI3d, MPI3d]:
    drop_list = list()
    if case == 0:
        object_color = 0
        object_shape = 0
        object_size = 0
        camera_height = 0
        background_color = 0
        for horizontal_axis in range(10, 20, 1):
            for vertical_axis in range(10, 20, 1):
                to_drop = np.array(
                    [
                        object_color,
                        object_shape,
                        object_size,
                        camera_height,
                        background_color,
                        horizontal_axis,
                        vertical_axis,
                    ]
                )
                drop_list.append(dataset.factor_to_idx(to_drop))

    elif case == 1:
        object_color = 2
        object_shape = 3
        object_size = 1
        camera_height = 1
        background_color = 2
        for horizontal_axis in range(0, 10, 1):
            for vertical_axis in range(0, 10, 1):
                to_drop = np.array(
                    [
                        object_color,
                        object_shape,
                        object_size,
                        camera_height,
                        background_color,
                        horizontal_axis,
                        vertical_axis,
                    ]
                )
                drop_list.append(dataset.factor_to_idx(to_drop))

    elif case == 2:
        object_color = 4
        object_shape = 4
        object_size = 0
        camera_height = 2
        background_color = 1
        for horizontal_axis in range(30, 40, 1):
            for vertical_axis in range(30, 40, 1):
                to_drop = np.array(
                    [
                        object_color,
                        object_shape,
                        object_size,
                        camera_height,
                        background_color,
                        horizontal_axis,
                        vertical_axis,
                    ]
                )
                drop_list.append(dataset.factor_to_idx(to_drop))

    drop_list = np.array(drop_list)

    train_mask = np.ones(len(dataset), dtype=bool)
    test_mask = np.zeros(len(dataset), dtype=bool)
    train_mask[drop_list] = False
    test_mask[drop_list] = True

    train_dataset = data.Subset(dataset, np.where(train_mask)[0])
    test_dataset = data.Subset(dataset, np.where(test_mask)[0])

    return train_dataset, test_dataset
