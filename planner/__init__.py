from typing import Dict

from mapper.semantic_mapper import SemanticMapper
from planner.baselines import LawnmowerPlanner, RandomPlanner
from planner.common import Planner
from planner.global_planners import FrontierPlanner, RecedingHorizonPlanner
from planner.local_planners import EpistemicImagePlanner, EpistemicImageGradientPlanner


def get_planner(
    cfg: Dict,
    mapper: SemanticMapper,
    **kwargs,
) -> Planner:
    simulator_name = cfg["simulator"]["name"]
    planner_type = cfg["planner"]["type"]

    if planner_type == "random":
        planner = RandomPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"]["angle"],
            cfg["planner"]["uav_specifications"],
        )
        planner.setup()
        return planner
    elif planner_type == "lawnmower":
        planner = LawnmowerPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"]["angle"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["step_sizes"],
        )
        planner.setup(mission_id=kwargs["mission_id"])
        return planner
    elif planner_type == "receding_horizon":
        planner = RecedingHorizonPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"]["angle"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["budget"],
            cfg["planner"]["max_iter"],
            cfg["planner"]["population_size"],
            cfg["planner"]["sigma0"],
            cfg["planner"]["horizon_length"],
            cfg["planner"]["lattice_step_size"],
        )
        planner.setup()
        return planner
    elif planner_type == "frontier":
        planner = FrontierPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"]["angle"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["budget"],
            cfg["planner"]["frontier_step_size"],
        )
        planner.setup()
        return planner
    elif planner_type == "epistemic_image":
        planner = EpistemicImagePlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"]["angle"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["step_size"],
            cfg["planner"]["edge_width"],
        )
        planner.setup()
        return planner
    elif planner_type == "epistemic_image_gradient":
        planner = EpistemicImageGradientPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"]["angle"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["step_size"],
            cfg["planner"]["edge_width"],
        )
        planner.setup()
        return planner
    else:
        raise ValueError(f"Planner type '{planner_type}' unknown!")
