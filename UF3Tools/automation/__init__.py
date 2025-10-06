"""

#Automation Sub-package

# Import key classes to make them directly accessible from UF3Tools.automation
from . import dask_handler
from .data_logger import DataLogger
from .ModelAutomater import ModelAutomater
#from .optuna_cma_handler import * # Assuming a class name
#from .optuna_handler import *
#from .OptunaSearch import *
#from .pso_handler import * # Assuming a class name

# Define what `from UF3Tools.automation import *` imports
__all__ = [
    'DaskHandler',
    'DataLogger',
    'ModelAutomater',
    #'OptunaCMAHandler',
    #'OptunaHandler',
    #'OptunaSearch',
    #'PSOHandler',
]
"""

"""
UF3Tools.automation subpackage initializer
"""

import importlib

__all__ = [
    "dask_handler",
    "data_logger",
    "ModelAutomater",
    "optuna_cma_handler",
    "optuna_handler",
    "OptunaSearch",
    "pso_handler",
]

def __getattr__(name):
    """Lazily import submodules/classes on first access."""
    if name == "dask_handler":
        return importlib.import_module(f"{__name__}.dask_handler")
    if name == "data_logger":
        return importlib.import_module(f"{__name__}.data_logger").DataLogger
    if name == "ModelAutomater":
        return importlib.import_module(f"{__name__}.ModelAutomater").ModelAutomater
    # Lazy-load function-only modules
    if name in ["optuna_cma_handler", "optuna_handler", "OptunaSearch", "pso_handler"]:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)