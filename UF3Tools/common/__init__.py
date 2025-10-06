# UF3Tools/common/__init__.py
"""
UF3Tools common utilities subpackage.
"""

# Optional: lazy-load modules if you want them accessible via UF3Tools.common.<name>
import importlib

_submodules = [
    "HiperGatorHelper",
    "LAMMPSModifier",
    "OUTCARHandler",
    "StructureDownloader",
    "UF3Utility",
    "VASPModifier",
]

def __getattr__(name):
    if name in _submodules:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    return list(globals().keys()) + _submodules
