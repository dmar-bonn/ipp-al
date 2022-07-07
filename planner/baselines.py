from typing import List, Optional, Dict

import numpy as np

from mapper.semantic_mapper import SemanticMapper
from planner.common import compute_flight_time, Planner


class RandomPlanner(Planner):
    def __init__(self, mapper: SemanticMapper, altitude: float, sensor_angle: List, uav_specifications: Dict):
        super(RandomPlanner, self).__init__(mapper, altitude, sensor_angle, uav_specifications)

        self.planner_name = "random"

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        sample_trials = 100
        while sample_trials > 0:
            sampled_y = np.random.uniform(low=boundary_space[1], high=max_y)
            sampled_x = np.random.uniform(low=boundary_space[0], high=max_x)
            sampled_pose = np.array([sampled_y, sampled_x, self.altitude], dtype=np.float32)
            sample_trials -= 1
            if compute_flight_time(sampled_pose, previous_pose, self.uav_specifications) <= budget:
                return sampled_pose

        return None


class LawnmowerPlanner(Planner):
    def __init__(
        self,
        mapper: SemanticMapper,
        altitude: float,
        sensor_angle: List,
        uav_specifications: Dict,
        step_sizes: List[float],
    ):
        super(LawnmowerPlanner, self).__init__(mapper, altitude, sensor_angle, uav_specifications)

        self.planner_name = "lawnmower"
        self.step_sizes = step_sizes
        self.waypoints = []
        self.step_counter = 0
        self.step_size = step_sizes[0]

    def setup(self, **kwargs):
        mission_id = kwargs["mission_id"]
        self.step_size = self.step_sizes[mission_id % len(self.step_sizes)]
        self.waypoints = self.create_lawnmower_pattern(flip_orientation=(mission_id % 2))

    def create_lawnmower_pattern(self, flip_orientation: bool = False) -> np.array:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle)) + 1
        min_y, min_x = boundary_space[1], boundary_space[0]
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        x_positions = np.linspace(min_x, max_x, int((max_x - min_x) / self.step_size) + 1)
        y_positions = np.linspace(min_y, max_y, int((max_y - min_y) / self.step_size) + 1)
        waypoints = np.zeros((len(y_positions) * len(x_positions), 3))

        for j, y_pos in enumerate(y_positions):
            for k, x_pos in enumerate(x_positions):
                if j % 2 == 1:
                    x_pos = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - x_pos

                if flip_orientation:
                    waypoints[j * len(x_positions) + k] = np.array([y_pos, x_pos, self.altitude], dtype=np.float32)
                else:
                    waypoints[j * len(x_positions) + k] = np.array([x_pos, y_pos, self.altitude], dtype=np.float32)

        return waypoints

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        if self.step_counter >= len(self.waypoints):
            return None

        pose = self.waypoints[self.step_counter, :]
        self.step_counter += 1

        if compute_flight_time(pose, previous_pose, self.uav_specifications) > budget:
            return None

        return pose
