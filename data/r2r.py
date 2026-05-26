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

def r2r_dsprites(case: int, dataset: DSprites) -> Tuple[DSprites, DSprites]:
    drop_list = list()

    if case == 0:
        shape = 2
        for scale in range(0, 6, 1):
            for orientation in range(0, 40, 1):
                for x in range(16, 32, 1):
                    for y in range(0, 32, 1):
                        to_drop = np.array([shape, scale, orientation, x, y])
                        drop_list.append(dataset.factor_to_idx(to_drop))

    elif case == 1:
        shape = 0
        for scale in range(0, 6, 1):
            for orientation in range(0, 40, 1):
                for x in range(0, 16, 1):
                    for y in range(0, 32, 1):
                        to_drop = np.array([shape, scale, orientation, x, y])
                        drop_list.append(dataset.factor_to_idx(to_drop))

    elif case == 2:
        shape = 1
        for scale in range(3, 6, 1):
            for orientation in range(0, 40, 1):
                for x in range(0, 32, 1):
                    for y in range(0, 16, 1):
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

def r2r_shape3d(case: int, dataset: Shapes3d) -> Tuple[Shapes3d, Shapes3d]:
    drop_list = list()

    if case == 0:
        shape = 3
        for floor_hue in range(0, 10, 1):
            for wall_hue in range(0, 10, 1):
                for object_hue in range(6, 10, 1):
                    for scale in np.linspace(0, 8, 1):
                        for orientation in range(0, 15, 1):
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
        shape = 0
        for floor_hue in range(0, 10, 1):
            for wall_hue in range(0, 10, 1):
                for object_hue in range(0, 10, 1):
                    for scale in np.linspace(0, 3, 1):
                        for orientation in range(0, 15, 1):
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
        shape = 2
        for floor_hue in range(0, 5, 1):
            for wall_hue in range(0, 10, 1):
                for object_hue in range(0, 10, 1):
                    for scale in np.linspace(0, 8, 1):
                        for orientation in range(0, 8, 1):
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

    drop_list = np.array(drop_list).astype(int)

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
case 0: except green cylinder with sea green background (1, 2, 0~1, 0~2, 1, 20~39, 20~39)
case 1: except blue small sphere from top view (3, 5, 0, 0, 0~2, 0~19, 0~19)
case 2: except olibe large cube from bottom view (5, 1, 1, 2, 0~2, 10~29, 10~29)
"""

def r2r_mpi3d(case: int, dataset: MPI3d) -> Tuple[MPI3d, MPI3d]:
    drop_list = list()
    if case == 0:
        object_color = 1
        object_shape = 2
        background_color = 1
        for object_size in range(2):
            for camera_height in range(3):
                for horizontal_axis in range(20, 40):
                    for vertical_axis in range(20, 40):
                        to_drop = np.array([object_color, object_shape, object_size, camera_height, background_color, horizontal_axis, vertical_axis])
                        drop_list.append(dataset.factor_to_idx(to_drop))

    elif case == 1:
        object_color = 3
        object_shape = 5
        object_size = 0
        camera_height = 0
        for background_color in range(3):
            for horizontal_axis in range(20):
                for vertical_axis in range(20):
                    to_drop = np.array([object_color, object_shape, object_size, camera_height, background_color, horizontal_axis, vertical_axis])
                    drop_list.append(dataset.factor_to_idx(to_drop))

    elif case == 2:
        object_color = 5
        object_shape = 1
        object_size = 1
        camera_height = 2
        for background_color in range(3):
            for horizontal_axis in range(20, 40):
                for vertical_axis in range(20, 40):
                    to_drop = np.array([object_color, object_shape, object_size, camera_height, background_color, horizontal_axis, vertical_axis])
                    drop_list.append(dataset.factor_to_idx(to_drop))

    drop_list = np.array(drop_list)

    train_dataset = deepcopy(dataset)
    
    train_dataset.data = np.delete(train_dataset.data, drop_list, axis=0)

    test_dataset = dataset

    mask = np.zeros(len(dataset), dtype=bool)
    mask[drop_list] = True
    test_dataset.data = test_dataset.data[mask]

    return train_dataset, test_dataset
