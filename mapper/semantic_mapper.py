from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch
from agri_semantics.utils.utils import toOneHot

from utils import utils


class SemanticMapper:
    def __init__(self, cfg_mapper: Dict, cfg_sensor: Dict):
        self.map_name = cfg_mapper["map_name"]
        self.map_boundary = cfg_mapper["map_boundary"]
        self.ground_resolution = cfg_mapper["ground_resolution"]
        self.class_num = cfg_mapper["class_number"]

        self.world_range = cfg_sensor["world_range"]
        self.sensor_angle = cfg_sensor["sensor"]["angle"]
        self.gsd = cfg_sensor["gsd"]

        self.logit_map, self.uncertainty_prior, self.epistemic_map, self.hit_map = self.init_map()

    def init_map(self) -> Tuple[np.array, np.array, np.array, np.array]:
        logit_map = (1 / self.class_num) * np.ones((self.class_num, self.map_boundary[1], self.map_boundary[0]))
        uncertainty_prior = 0.1 * np.ones((self.map_boundary[1], self.map_boundary[0]))
        epistemic_map = np.zeros((self.map_boundary[1], self.map_boundary[0]))
        hit_map = np.zeros((self.map_boundary[1], self.map_boundary[0]))
        return logit_map, uncertainty_prior, epistemic_map, hit_map

    def find_map_index(self, data_point) -> Tuple[float, float]:
        x_index = np.floor(data_point[0] / self.ground_resolution[0]).astype(int)
        y_index = np.floor(data_point[1] / self.ground_resolution[1]).astype(int)

        return x_index, y_index

    def map_update(self, data_source: Dict):
        logits = data_source["logits"]
        uncertainty = data_source["uncertainty"]
        fov = data_source["fov"]
        gsd = data_source["gsd"]
        _c, m_y_dim, m_x_dim = logits.shape

        measurement_indices = np.array(np.meshgrid(np.arange(m_y_dim), np.arange(m_x_dim))).T.reshape(-1, 2).astype(int)
        x_ground = fov[0][0] + (0.5 + np.arange(m_x_dim)) * gsd[0]
        y_ground = fov[0][1] + (0.5 + np.arange(m_y_dim)) * gsd[1]
        ground_coords = np.array(np.meshgrid(y_ground, x_ground)).T.reshape(-1, 2)
        map_indices = np.floor(ground_coords / np.array(self.ground_resolution)).astype(int)

        f_prior = self.logit_map[:, map_indices[:, 0], map_indices[:, 1]]
        sigma_prior = self.uncertainty_prior[map_indices[:, 0], map_indices[:, 1]]
        f = logits[:, measurement_indices[:, 0], measurement_indices[:, 1]]
        sigma = uncertainty[measurement_indices[:, 0], measurement_indices[:, 1]]

        self.hit_map[map_indices[:, 0], map_indices[:, 1]] += 1
        self.epistemic_map[map_indices[:, 0], map_indices[:, 1]] = (
            self.epistemic_map[map_indices[:, 0], map_indices[:, 1]] + sigma
        ) / self.hit_map[map_indices[:, 0], map_indices[:, 1]]

        f_post, sigma_post = self.kalman_filter(f, f_prior, sigma, sigma_prior)
        self.logit_map[:, map_indices[:, 0], map_indices[:, 1]] = f_post
        self.uncertainty_prior[map_indices[:, 0], map_indices[:, 1]] = sigma_post

    def get_map_state(self, pose: np.array) -> Tuple[np.array, np.array, np.array, np.array, List]:
        fov_corners, _ = utils.get_fov(pose, self.sensor_angle, self.gsd, self.world_range)
        lu, _, rd, _ = fov_corners
        lu_x, lu_y = self.find_map_index(lu)
        rd_x, rd_y = self.find_map_index(rd)

        return (
            self.logit_map[:, lu_y:rd_y, lu_x:rd_x].copy(),
            self.uncertainty_prior[lu_y:rd_y, lu_x:rd_x].copy(),
            self.epistemic_map[lu_y:rd_y, lu_x:rd_x].copy(),
            self.hit_map[lu_y:rd_y, lu_x:rd_x].copy(),
            [lu_x, lu_y, rd_x, rd_y],
        )

    def semantic_output(self, ax):
        semantic_map = toOneHot(torch.from_numpy(self.logit_map).unsqueeze(0), self.map_name)
        ax.imshow(semantic_map)

    def epistemic_output(self, ax):
        sb.heatmap(self.epistemic_map, ax=ax, cmap="plasma")

    def visualizer(self):
        fig, ax = plt.subplots(1, 2)
        self.semantic_output(ax[0])
        self.epistemic_output(ax[1])
        plt.show()

    @staticmethod
    def kalman_filter(
        f: np.array, f_prior: np.array, sigma: np.array, sigma_prior: np.array
    ) -> Tuple[np.array, np.array]:
        kalman_gain = sigma_prior / (sigma_prior + sigma + 10 ** (-8))
        f_post = f_prior + kalman_gain * (f - f_prior)
        sigma_post = (1 - kalman_gain) * sigma_prior

        return f_post, sigma_post
