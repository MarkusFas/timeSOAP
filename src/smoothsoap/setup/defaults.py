DEFAULT_PARAMS = {
    "system": None,
    "version": "v0",
    "specifier": "default",

    "interval": 100,
    "lag": 0,
    "max_lag": 1000,
    "min_lag": 100,
    "lag_step": 20,
    "sigma": 0,
    "n_cumulants": 1,
    "methods": "PCA",
    "spatial_cutoff": 0,
    "train_selected_atoms": 1,
    "test_selected_atoms": None,

    "input_params": {
        "fname": None,
        "indices": ":",
        "concatenate": True,
    },
    "output_per_structure": True,
    "output_params": {
        "fname": None,
        "indices": ":",
        "concatenate": True,
    },

    "descriptor": "SOAP",
    "SOAP_params": {
        "centers": None,
        "neighbors": None,
        "cutoff": 5.0,
        "max_angular": 6,
        "max_radial": 6,
    },

    "plots": ["projection"],
    "ridge": False,
    "predict_avg": False,
    "ridge_alpha": 1e-5,
    "ridge_save": True,
    "model_save": False,
    "model_proj_dims": [0],
    "model_load": False,
    "trafo_load": False,
    "i_pca": 0,
    "log": False,
    "classify": {
        "request": False,
        "switch_index": None,
    }
}
