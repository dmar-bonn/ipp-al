from typing import Dict

import cv2
import numpy as np
import torch
from agri_semantics.utils.utils import infer_segmentation_and_epistemic_uncertainty_from_image
from pytorch_lightning import LightningModule

from mapper.semantic_mapper import SemanticMapper
from planner.common import Planner
from planner.common import compute_flight_time
from simulator.fake_simulator import FakeSimulator
from simulator.oracles import Oracle
from utils.logger import Logger


class Mission:
    def __init__(
        self,
        planner: Planner,
        mapper: SemanticMapper,
        fake_simulator: FakeSimulator,
        oracle: Oracle,
        model: LightningModule,
        init_pose: np.array,
        cfg: Dict,
        model_cfg: Dict,
        logger: Logger,
    ):
        self.logger = logger
        self.planner = planner
        self.mapper = mapper
        self.fake_simulator = fake_simulator
        self.oracle = oracle
        self.model = model
        self.init_pose = init_pose
        self.budget = cfg["planner"]["budget"]
        self.uav_specifications = cfg["planner"]["uav_specifications"]
        self.cfg = cfg
        self.model_cfg = model_cfg

    def execute(self, mission_id: int):
        previous_pose = self.init_pose

        while self.budget > 0:
            # Get image and annotation
            measurement = self.fake_simulator.get_measurement(previous_pose)
            image = measurement["image"]
            anno = self.oracle.get_anno(previous_pose)["anno"]

            # Log current pose and save train data
            self.logger.update_poses(previous_pose)
            self.logger.save_train_data_to_disk(image, anno, self.model_cfg["data"]["path_to_dataset"])

            # Forward pass through network
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(1, 0, 2)
            probs, uncertainty = infer_segmentation_and_epistemic_uncertainty_from_image(
                self.model,
                image,
                num_mc_epistemic=self.cfg["network"]["num_mc_epistemic"],
                resize_image=False,
                aleatoric_model=self.cfg["network"]["aleatoric_model"],
            )

            _, preds = torch.max(torch.from_numpy(probs), dim=0)
            image, preds, uncertainty, probs = (
                image.transpose(1, 0, 2),
                preds.transpose(1, 0),
                uncertainty.transpose(1, 0),
                probs.transpose(0, 2, 1),
            )

            map_data = {
                "logits": probs,
                "uncertainty": uncertainty,
                "fov": measurement["fov"],
                "gsd": measurement["gsd"],
            }
            self.mapper.map_update(map_data)

            pose = self.planner.replan(self.budget, previous_pose, epistemic_image=uncertainty)
            if pose is None:
                print(f"FINISHED '{self.planner.planner_name}' PLANNING MISSION")
                print(f"CHOSEN PATH: {self.logger.poses_list}")
                break

            self.budget -= compute_flight_time(pose, previous_pose, uav_specifications=self.uav_specifications)
            print(f"NEXT POSE: {pose}, REMAINING BUDGET: {self.budget}")

            previous_pose = pose

        file_id = f"{self.cfg['simulator']['name']}_{self.cfg['planner']['type']}_{mission_id}"
        self.logger.save_maps_to_disk(
            self.mapper.logit_map, self.mapper.epistemic_map, file_id, self.cfg["mapper"]["map_name"]
        )
        self.logger.save_path_to_disk(file_id)
