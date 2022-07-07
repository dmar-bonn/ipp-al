from typing import Dict

from mapper.semantic_mapper import SemanticMapper


def get_mapper(cfg: Dict) -> SemanticMapper:
    simulator_name = cfg["simulator"]["name"]

    if simulator_name not in cfg["simulator"].keys():
        raise KeyError(f"No simulation with name '{simulator_name}' specified in config file!")

    return SemanticMapper(cfg["mapper"], cfg["simulator"][simulator_name])
