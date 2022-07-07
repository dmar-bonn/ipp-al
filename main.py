from typing import Dict

import numpy as np
import yaml

from active_learning.missions import Mission
from active_learning.model_learners import ModelLearner
from mapper import get_mapper
from planner import get_planner
from simulator import get_oracle, get_simulator
from utils.logger import Logger


def main(cfg: Dict, model_cfg: Dict):
    experiment_name = f"{cfg['simulator']['name']}_{cfg['planner']['type']}"
    logger = Logger(experiment_name)
    logger.save_config_files_to_disk(cfg, model_cfg)
    init_pose = np.array([30, 30, 30], dtype=np.float32)

    fake_simulator = get_simulator(cfg)
    oracle = get_oracle(cfg)
    model_learner = ModelLearner(model_cfg, cfg["network"]["path_to_checkpoint"])
    trained_model = model_learner.setup_model()

    for mission_id in range(cfg["planner"]["num_missions"]):
        mapper = get_mapper(cfg)
        planner = get_planner(cfg, mapper, mission_id=mission_id)
        mission = Mission(planner, mapper, fake_simulator, oracle, trained_model, init_pose, cfg, model_cfg, logger)

        mission.execute(mission_id)
        trained_model = model_learner.train(mission_id)
        test_statistics = model_learner.evaluate()

        logger.save_evaluation_metrics_to_disk(test_statistics)
        logger.reset_poses()


if __name__ == "__main__":
    with open("config/config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    with open(cfg["network"]["path_to_config"], "r") as config_file:
        model_cfg = yaml.safe_load(config_file)

    main(cfg, model_cfg)
