"""
Automation Sub-package
"""

# Import key classes to make them directly accessible from UF3Tools.automation
from .dask_handler import DaskHandler  # Assuming a class name
from .data_logger import DataLogger
from .ModelAutomater import ModelAutomater
from .optuna_cma_handler import OptunaCMAHandler # Assuming a class name
from .optuna_handler import OptunaHandler
from .OptunaSearch import OptunaSearch
from .pso_handler import PSOHandler # Assuming a class name

# Define what `from UF3Tools.automation import *` imports
__all__ = [
    'DaskHandler',
    'DataLogger',
    'ModelAutomater',
    'OptunaCMAHandler',
    'OptunaHandler',
    'OptunaSearch',
    'PSOHandler',
]
