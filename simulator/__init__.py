import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from simulator.fake_simulator import FakeSimulator
from simulator.oracles import Oracle


def get_simulator(cfg) -> FakeSimulator:
    simulator_cfg = cfg["simulator"]

    if isinstance(simulator_cfg, dict):
        print("creating simulation world")
        if simulator_cfg["name"] == "rit18":
            rit18_world = get_rit18_world(simulator_cfg["rit18"])
            return FakeSimulator(simulator_cfg["rit18"], rit18_world)
        elif simulator_cfg["name"] == "potsdam":
            potsdam_world = get_potsdam_world(simulator_cfg["potsdam"])
            cv2.imwrite("potsdam_world.png", potsdam_world)
            return FakeSimulator(simulator_cfg["potsdam"], potsdam_world)
        elif simulator_cfg["name"] == "weedmap":
            # weedmap_world =get_weedmap_world(simulator_cfg['weedmap'])
            weedmap_world = 0
            return FakeSimulator(simulator_cfg["weedmap"], weedmap_world)
    else:
        raise RuntimeError(f"{type(simulator_cfg)} not a valid config file")


def get_oracle(cfg) -> Oracle:
    simulator_cfg = cfg["simulator"]

    if isinstance(simulator_cfg, dict):
        print("creating oracle annotations")
        if simulator_cfg["name"] == "rit18":
            rit18_anno = get_rit18_anno(simulator_cfg["rit18"])
            return Oracle(simulator_cfg["rit18"], rit18_anno)
        elif simulator_cfg["name"] == "potsdam":
            potsdam_anno = get_potsdam_anno(simulator_cfg["potsdam"])
            cv2.imwrite("potsdam_anno.png", potsdam_anno)
            return Oracle(simulator_cfg["potsdam"], potsdam_anno)
        elif simulator_cfg["name"] == "weedmap":
            # weedmap_world =get_weedmap_world(simulator_cfg['weedmap'])
            weedmap_anno = 0
            return Oracle(simulator_cfg["weedmap"], weedmap_anno)
    else:
        raise RuntimeError(f"{type(simulator_cfg)} not a valid config file")


def get_rit18_world(cfg):
    path_to_orthomosaic = cfg["path_to_orthomosaic"]
    resize_flag = cfg["resize_flag"]
    resize = cfg["resize"]

    if not os.path.exists(path_to_orthomosaic):
        raise FileNotFoundError

    orthomosaic = cv2.imread(path_to_orthomosaic)
    orthomosaic = cv2.cvtColor(orthomosaic, cv2.COLOR_BGR2RGB)

    if resize_flag:
        orthomosaic = cv2.resize(orthomosaic, tuple(resize))

    return orthomosaic


def get_rit18_anno(cfg):
    path_to_orthomosaic = cfg["path_to_anno"]
    resize_flag = cfg["resize_flag"]
    resize = cfg["resize"]

    if not os.path.exists(path_to_orthomosaic):
        raise FileNotFoundError

    orthomosaic = cv2.imread(path_to_orthomosaic)
    orthomosaic = cv2.cvtColor(orthomosaic, cv2.COLOR_BGR2RGB)

    if resize_flag:
        orthomosaic = cv2.resize(orthomosaic, tuple(resize))

    return orthomosaic


def get_weedmap_world(cfg):
    path_to_orthomosaic = cfg["path_to_anno"]
    resize_flag = cfg["resize_flag"]
    resize = cfg["resize"]

    if not os.path.exists(path_to_orthomosaic):
        raise FileNotFoundError

    orthomosaic = cv2.imread(path_to_orthomosaic)

    if resize_flag:
        orthomosaic = cv2.resize(orthomosaic, tuple(resize))

    return orthomosaic


def get_potsdam_world(cfg):
    path_to_orthomosaic = cfg["path_to_orthomosaic"]
    resize_flag = cfg["resize_flag"]
    resize = cfg["resize"]
    orth_tile_list = cfg["orth_tile_list"]
    orth_rgb = []

    for row in orth_tile_list:
        orth_rgb_row = []
        for orth_num in row:
            path_to_tile = f"{path_to_orthomosaic}/potsdam_{orth_num}_RGB.tif"
            if not os.path.exists(path_to_tile):
                raise FileNotFoundError

            rgb = cv2.imread(path_to_tile)
            if resize_flag:
                rgb = cv2.resize(rgb, tuple(resize))
            orth_rgb_row.append(rgb)
        orth_rgb.append(np.concatenate(orth_rgb_row, axis=1))

    orthomosaic = np.concatenate(orth_rgb, axis=0)

    return orthomosaic


def get_potsdam_anno(cfg):
    path_to_orthomosaic = cfg["path_to_anno"]
    resize_flag = cfg["resize_flag"]
    resize = cfg["resize"]
    orth_tile_list = cfg["orth_tile_list"]
    orth_rgb = []

    for row in orth_tile_list:
        orth_rgb_row = []
        for orth_num in row:
            path_to_tile = f"{path_to_orthomosaic}/potsdam_{orth_num}_label.tif"
            if not os.path.exists(path_to_tile):
                raise FileNotFoundError

            rgb = cv2.imread(path_to_tile)
            if resize_flag:
                rgb = cv2.resize(rgb, tuple(resize))
            orth_rgb_row.append(rgb)
        orth_rgb.append(np.concatenate(orth_rgb_row, axis=1))

    orthomosaic = np.concatenate(orth_rgb, axis=0)

    return orthomosaic
