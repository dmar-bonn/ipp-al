from typing import Dict, List, Optional, Tuple

import cma
import cv2
import numpy as np

from mapper.semantic_mapper import SemanticMapper
from planner.common import compute_flight_time, Planner


class RecedingHorizonPlanner(Planner):
    def __init__(
        self,
        mapper: SemanticMapper,
        altitude: float,
        sensor_angle: List,
        uav_specifications: Dict,
        budget: float,
        max_iter: int,
        population_size: int,
        sigma0: List,
        horizon_length: int,
        lattice_step_size: float,
    ):
        super(RecedingHorizonPlanner, self).__init__(mapper, altitude, sensor_angle, uav_specifications)

        self.planner_name = "receding_horizon"
        self.sigma0 = sigma0
        self.max_iter = max_iter
        self.population_size = population_size
        self.horizon_length = horizon_length
        self.lattice_step_size = lattice_step_size
        self.budget = budget
        self.remaining_budget = budget

    @staticmethod
    def stacked_poses(poses: List) -> List:
        stacked_poses = []
        for i in range(len(poses) // 3):
            stacked_poses.append(np.array([poses[3 * i], poses[3 * i + 1], poses[3 * i + 2]]))

        return stacked_poses

    @staticmethod
    def flatten_poses(poses: List):
        flattened_poses = []
        for poses in poses:
            flattened_poses.extend([poses[0], poses[1], poses[2]])

        return flattened_poses

    def objective_function(self, flattened_poses: List) -> float:
        poses = self.stacked_poses(flattened_poses)
        total_epistemic_uncertainty = 0
        total_hit_count = 0
        total_budget_used = 0
        simulated_hit_map = np.zeros(self.mapper.hit_map.shape)

        for i, pose in enumerate(poses):
            if i > 0:
                total_budget_used += compute_flight_time(pose, poses[i - 1], self.uav_specifications)

            _, _, epistemic_submap, hit_submap, fov_indices = self.mapper.get_map_state(pose)
            lu_x, lu_y, rd_x, rd_y = fov_indices
            simulated_unknown_space_mask = (simulated_hit_map[lu_y:rd_y, lu_x:rd_x] == 0) & (hit_submap == 0)
            epistemic_submap[simulated_unknown_space_mask] = 0.15
            simulated_hit_map[lu_y:rd_y, lu_x:rd_x] += 1

            total_epistemic_uncertainty += np.sum(epistemic_submap)
            total_hit_count += np.sum(hit_submap)

        if total_budget_used == 0 or total_budget_used > self.remaining_budget:
            return 0

        total_hit_count += np.sum(simulated_hit_map)
        return -(total_epistemic_uncertainty / (total_hit_count + 1)) / ((total_budget_used + 1) / self.budget)

    def calculate_parameter_bounds_and_scales(self, num_waypoints: int) -> Tuple[List, List, List]:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        upper_x = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1] - 1
        upper_y = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0] - 1
        lower_x, lower_y = boundary_space[1] + 1, boundary_space[0] + 1
        lower_z, upper_z = self.altitude - 1.0, self.altitude + 1.0

        lower_bounds = []
        upper_bounds = []
        sigma_scales = []

        for i in range(num_waypoints):
            lower_bounds.extend([lower_y, lower_x, lower_z])
            upper_bounds.extend([upper_y, upper_x, upper_z])
            sigma_scales.extend(self.sigma0)

        return lower_bounds, upper_bounds, sigma_scales

    def cma_es_optimization(self, init_waypoints: np.array) -> List:
        lower_bounds, upper_bounds, sigma_scales = self.calculate_parameter_bounds_and_scales(self.horizon_length)
        cma_es = cma.CMAEvolutionStrategy(
            self.flatten_poses(init_waypoints),
            sigma0=1,
            inopts={
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": self.max_iter,
                "popsize": self.population_size,
                "CMA_stds": sigma_scales,
                "verbose": -9,
            },
        )
        cma_es.optimize(self.objective_function)

        return self.stacked_poses(list(cma_es.result.xbest))

    def create_planning_lattice(self) -> np.array:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        upper_x = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1] - 1
        upper_y = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0] - 1
        lower_x, lower_y = boundary_space[1] + 1, boundary_space[0] + 1

        lattice_steps = int(self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] / self.lattice_step_size)
        y_candidates = np.linspace(lower_y, upper_y, lattice_steps)
        x_candidates = np.linspace(lower_x, upper_x, lattice_steps)
        pose_candidates = np.array(np.meshgrid(y_candidates, x_candidates)).T.reshape(-1, 2)

        return np.hstack((pose_candidates, self.altitude * np.ones((len(pose_candidates), 1))))

    def init_poses(self, previous_pose: np.array, budget: float) -> List:
        pose_candidates = self.create_planning_lattice()
        init_poses = []
        simulated_hit_map = np.zeros(self.mapper.hit_map.shape)

        for i in range(self.horizon_length):
            best_pose = previous_pose
            lu_x_best, lu_y_best, rd_x_best, rd_y_best = 0, 0, 0, 0
            best_pose_value = -np.inf
            for pose_candidate in pose_candidates:
                if np.linalg.norm(pose_candidate - previous_pose, ord=2) < self.lattice_step_size:
                    continue

                pose_costs = compute_flight_time(pose_candidate, previous_pose, self.uav_specifications)
                if pose_costs > budget:
                    continue

                _, _, epistemic_submap, hit_submap, fov_indices = self.mapper.get_map_state(pose_candidate)
                lu_x, lu_y, rd_x, rd_y = fov_indices
                simulated_unknown_space_mask = (simulated_hit_map[lu_y:rd_y, lu_x:rd_x] == 0) & (hit_submap == 0)
                epistemic_submap[simulated_unknown_space_mask] = 0.15

                simulated_hit_submap_count = np.sum(simulated_hit_map[lu_y:rd_y, lu_x:rd_x]) + np.sum(hit_submap)
                pose_value = (np.sum(epistemic_submap) / (simulated_hit_submap_count + 1)) / (
                    (pose_costs + 1) / self.budget
                )

                if pose_value > best_pose_value:
                    best_pose = pose_candidate
                    best_pose_value = pose_value
                    lu_x_best, lu_y_best, rd_x_best, rd_y_best = lu_x, lu_y, rd_x, rd_y

            budget -= compute_flight_time(best_pose, previous_pose, self.uav_specifications)
            init_poses.append(best_pose)
            previous_pose = best_pose
            simulated_hit_map[lu_y_best:rd_y_best, lu_x_best:rd_x_best] += 1

        return init_poses

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        self.remaining_budget = budget
        init_poses = self.init_poses(previous_pose, budget)
        poses = self.cma_es_optimization(init_poses)

        greedy_value = -self.objective_function(self.flatten_poses(init_poses))
        cmaes_value = -self.objective_function(self.flatten_poses(poses))
        if greedy_value > cmaes_value:
            poses = init_poses

        next_best_pose = poses[0]
        next_best_pose[2] = self.altitude

        if compute_flight_time(next_best_pose, previous_pose, self.uav_specifications) > budget:
            return None

        return next_best_pose


class FrontierPlanner(Planner):
    def __init__(
        self,
        mapper: SemanticMapper,
        altitude: float,
        sensor_angle: List,
        uav_specifications: Dict,
        budget: float,
        frontier_step_size: float,
    ):
        super(FrontierPlanner, self).__init__(mapper, altitude, sensor_angle, uav_specifications)

        self.planner_name = "frontier"
        self.budget = budget
        self.frontier_step_size = frontier_step_size

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        hit_map_img = self.mapper.hit_map.astype(np.uint8)
        hit_map_img[hit_map_img > 1] = 1
        contours, hierarchy = cv2.findContours(hit_map_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        frontiers = contours[0][:, 0][:: self.frontier_step_size, :]

        best_pose = previous_pose
        best_frontier_value = -np.inf
        for frontier_candidate in frontiers:
            frontier_candidate = frontier_candidate * self.mapper.ground_resolution
            frontier_candidate = np.append(frontier_candidate, self.altitude)

            frontier_costs = compute_flight_time(frontier_candidate, previous_pose, self.uav_specifications)
            if frontier_costs > budget:
                continue

            frontier_candidate[0] = np.clip(frontier_candidate[0], boundary_space[0], max_x)
            frontier_candidate[1] = np.clip(frontier_candidate[1], boundary_space[1], max_y)

            if np.allclose(previous_pose, frontier_candidate):
                continue

            _, _, epistemic_submap, hit_submap, fov_indices = self.mapper.get_map_state(frontier_candidate)
            # epistemic_submap[hit_submap == 0] = 0.1

            total_epistemic_uncertainty = np.sum(epistemic_submap)
            total_hit_count = np.sum(hit_submap)

            if (total_epistemic_uncertainty / total_hit_count) > best_frontier_value:
                best_frontier_value = total_epistemic_uncertainty / total_hit_count
                best_pose = frontier_candidate

        if np.allclose(previous_pose, best_pose):
            return None

        if compute_flight_time(best_pose, previous_pose, self.uav_specifications) > budget:
            return None

        return best_pose
