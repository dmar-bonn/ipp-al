from typing import Dict, List, Optional

import numpy as np

from mapper.semantic_mapper import SemanticMapper


def compute_flight_time(action: np.array, previous_action: np.array, uav_specifications: Dict = None) -> float:
    dist_total = np.linalg.norm(action - previous_action, ord=2)
    dist_acc = min(dist_total * 0.5, np.square(uav_specifications["max_v"]) / (2 * uav_specifications["max_a"]))
    dist_const = dist_total - 2 * dist_acc

    time_acc = np.sqrt(2 * dist_acc / uav_specifications["max_a"])
    time_const = dist_const / uav_specifications["max_v"]
    time_total = time_const + 2 * time_acc

    return time_total


class Planner:
    def __init__(self, mapper: SemanticMapper, altitude: float, sensor_angle: List, uav_specifications: Dict):
        self.planner_name = "planner"
        self.mapper = mapper
        self.altitude = altitude
        self.sensor_angle = sensor_angle
        self.uav_specifications = uav_specifications

    def setup(self, **kwargs):
        pass

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        raise NotImplementedError("Replan function not implemented!")
