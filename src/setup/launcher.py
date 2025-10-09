# src/launcher/run_system.py
from src.setup.defaults import DEFAULT_PARAMS
from src.setup.input_check import setup_simulation
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_params(defaults, updates, path="root"):
    """
    Update default params, check if all used keywords are in the defaults
    """
    for key, value in updates.items():
        if key not in defaults:
            raise KeyError(f"Unknown config key '{key}' at path '{path}'")
        if isinstance(value, dict) and isinstance(defaults[key], dict):
            merge_params(defaults[key], value, path=f"{path}.{key}")
        else:
            defaults[key] = value
    return defaults


if __name__ == "__main__":
    #input_file = 'systems/icewater/test_interval1.yaml'
    input_file = 'systems/gete/test_interval1.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)

    input_file = 'systems/gete/test_interval100.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)

    input_file = 'systems/gete/test_interval250.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)

    exit()
    input_file = 'systems/icewater/test_interval250.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)