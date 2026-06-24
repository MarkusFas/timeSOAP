# src/launcher/run_system.py

from importlib.resources import files
from pathlib import Path
import argparse
from smoothsoap.setup.defaults import DEFAULT_PARAMS
from smoothsoap.setup.input_check import setup_simulation
import yaml
import sys

def load_config(yaml_path):
    with yaml_path.open("r") as f:
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


def main():
    parser = argparse.ArgumentParser(
    description="Run SmoothSOAP simulation with a YAML config."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config file, if not given, user can specify default system and runfiles in launcher",
    )
    args = parser.parse_args()
    input_file = args.config
 
    
    #input_file = 'systems/icewater/test.yaml'
    #input_file = 'systems/icewater/test_interval1.yaml'
    #input_file = 'systems/cycloAE/test_sep_tests.yaml'
    #input_file = 'systems/cycloAE/test_intervalpca.yaml'
    #input_file = 'systems/ice_water_sep/test_interval1.yaml'
    #input_file = 'systems/smallcell_interface_350/test_interval_lf.yaml'
    #input_file = 'systems/icewater/test_interval1.yaml'
    #input_file = 'systems/smallcell_interface_350/test_intervaltemp.yaml'
    #input_file = 'systems/cycloAE/test_interval_hf0.yaml'
    #input_file = 'systems/smallcell_interface_350/test_interval_lf0.yaml'
    #input_file = 'systems/iron/run.yaml'
    #input_file = 'systems/smallcell_interface_350/test_metad_trj.yaml'
    #input_file = 'systems/ice_water_sep/test_interval1.yaml'
    #input_file = 'systems/GeTe/test_interval1.yaml' 
    #input_file = 'systems/cycloAE/test_interval_hf0.yaml'
    #input_file = 'systems/test_hannah/test_interval1.yaml'
    #input_file = 'systems/smallcell_interface_350/test_intervaltica.yaml'
    #input_file= 'systems/BaTiO3/test.yaml'

    if len(sys.argv)>1:
        input_file=sys.argv[1]
    else:
       print('Please provide default file')

    #input_file = 'systems/ice_water_sep/test_intervaltemp.yaml'
    #input_file = 'systems/chignolin/run0.yaml'
    #input_file = 'systems/GeTe/run0.yaml'
    #input_file = 'systems/ala/run0.yaml'

    if input_file is None:
        systems = ['batio3']#['smallcell_interface_350', 'smallcell_interface_350', 'smallcell_interface_350', 'smallcell_interface_350', 'smallcell_interface_350'] #['iron'] 
        runfiles = ['separateallfour']#['test_interval_lf0', 'test_interval_lf1', 'test_interval_lf2', 'test_interval_lf3', 'test_interval_lf4'] #['run'] 

        yaml_paths = []
        for system, runfile in zip(systems, runfiles):
            print(f"Starting setup for {system} {runfile}...")
            yaml_paths.append(files(f"smoothsoap.systems.{system}").joinpath(f"{runfile}.yaml"))
    else:
        input_file = Path(input_file).resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"Config file does not exist: {input_file}")
        yaml_paths = [input_file]

    for yaml_path in yaml_paths:
        user_cfg = load_config(yaml_path)
        params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
        params["base_path"] = Path.cwd()
        print("Parameters loaded. Starting simulation setup...")
        setup_simulation(**params)


if __name__ == "__main__":
    main()
