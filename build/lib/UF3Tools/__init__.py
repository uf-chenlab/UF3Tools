import os
import glob
import importlib

def import_all_classes_from_directory(package_dir, package_name):
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Create the module name relative to the package
                relative_path = os.path.relpath(os.path.join(root, file), package_dir)
                module_name = os.path.splitext(relative_path)[0].replace(os.path.sep, ".")

                # Import the module
                module = importlib.import_module(f"{package_name}.{module_name}")

                # Import all classes from the module into the package namespace
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        globals()[attr_name] = attr

# Define the package directory and package name
package_dir = os.path.dirname(__file__)
package_name = __name__

# Import all classes
#import_all_classes_from_directory(package_dir, package_name)

# This tells Python that 'automation', 'simulation', and 'common' are
# part of the UF3Tools package and makes them importable.
from . import automation
from . import simulation
from . import common

# It's good practice to define __all__ to control what 'from UF3Tools import *' does
__all__ = [
    'automation',
    'simulation',
    'common',
]