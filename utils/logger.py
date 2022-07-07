import os
from typing import Dict

import cv2
import numpy as np
import torch
import yaml
from agri_semantics.utils.utils import toOneHot

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, logger_name: str):
        self.logger_name = logger_name
        self.setup_log_dir()
        self.poses_list = np.empty((0, 3), float)

    def setup_log_dir(self):
        if os.path.exists(self.logger_name):
            raise ValueError(f"{self.logger_name} log directory already exists!")

        os.makedirs(self.logger_name)

    def reset_poses(self):
        self.poses_list = np.empty((0, 3), float)

    def update_poses(self, pose: np.array):
        self.poses_list = np.append(self.poses_list, [pose], axis=0)

    @staticmethod
    def save_train_data_to_disk(image: np.array, anno: np.array, dataset_path: str):
        anno_dir = os.path.join(dataset_path, "training_set", "anno")
        image_dir = os.path.join(dataset_path, "training_set", "image")

        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        train_data_id = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
        image_filepath = os.path.join(image_dir, f"rgb_{str(train_data_id).zfill(5)}.png")
        anno_filepath = os.path.join(anno_dir, f"gt_{str(train_data_id).zfill(5)}.png")

        cv2.imwrite(image_filepath, image)
        cv2.imwrite(anno_filepath, anno)

    def save_maps_to_disk(self, logit_map: np.array, epistemic_map: np.array, file_id: str, map_name: str):
        plt.imsave(
            os.path.join(self.logger_name, f"semantics_{file_id}.png"),
            toOneHot(torch.from_numpy(logit_map).unsqueeze(0), map_name),
        )
        plt.imsave(os.path.join(self.logger_name, f"uncertainty_{file_id}.png"), epistemic_map, cmap="plasma")

        with open(os.path.join(self.logger_name, f"semantic_map_{file_id}.npy"), "wb") as file:
            np.save(file, logit_map)

        with open(os.path.join(self.logger_name, f"epistemic_map_{file_id}.npy"), "wb") as file:
            np.save(file, epistemic_map)

    def save_path_to_disk(self, file_id: str):
        plt.plot(self.poses_list[:, 0], self.poses_list[:, 1], "-ok")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(self.logger_name, f"path_{file_id}.png"))

        with open(os.path.join(self.logger_name, f"path_poses_{file_id}.npy"), "wb") as file:
            np.save(file, self.poses_list)

    def save_config_files_to_disk(self, cfg: Dict, model_cfg: Dict):
        with open(os.path.join(self.logger_name, "config.yaml"), "w") as file:
            yaml.dump(cfg, file)

        with open(os.path.join(self.logger_name, "model_config.yaml"), "w") as file:
            yaml.dump(model_cfg, file)

    def save_evaluation_metrics_to_disk(self, test_statistics: Dict):
        with open(os.path.join(self.logger_name, "evaluation_metrics.yaml"), "w") as file:
            yaml.dump(test_statistics, file)
