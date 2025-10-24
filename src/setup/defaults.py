DEFAULT_PARAMS = {
    "system": None,
    "version": "v0",
    "specifier": "default",

    "interval": 100,
    "lag": 0,
    "max_lag": 1000,
    "min_lag": 100,
    "lag_step": 20,
    "sigma": None,
    "methods": "PCA",
    "train_selected_atoms": 1,
    "test_selected_atoms": 1,

    "input_params": {
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
}