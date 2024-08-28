from .SimulationManager import SimulationManager
import os
import shutil
from pymatgen.io.vasp import Poscar
from pymatgen.io.vasp.outputs import Outcar
import numpy as np
from pymatgen.core import Element
from ..simulations.SurfaceEnergySimulation import DFTSurfaceEnergySimulation, DFTSurfaceEnergyBulkSimulation, DFTSurfaceEnergyFreeSimulation
from ..simulations.RelaxationSimulation import DFTRelaxationSimulation
from ..common.DFTDataParser import DFTDataParser


        
class DFTSimulationManager(SimulationManager):
    def __init__(self):
        # Automatically set the parent directory to "/dft"
        super().__init__("dft")
    def create_simulation(self, simulation_type, **kwargs):
        """
        Create a DFT simulation based on the simulation type.

        :param simulation_type: The type of simulation to set up (e.g., "relaxation").
        :param kwargs: Additional arguments needed for the specific simulation.
        """
        if simulation_type.lower() == "relaxation":
            template_directory = os.path.join(self.parent_directory, "templates/lattice_parameters/")
            self.simulation = DFTRelaxationSimulation(self.simulation_dir, template_directory, **kwargs)
        if simulation_type.lower() == "surface_energy":
            template_directory = os.path.join(self.parent_directory, "templates/surface_energy/")
            self.simulation = DFTSurfaceEnergySimulation(self.simulation_dir, template_directory, **kwargs)
        else:
            raise ValueError(f"Simulation type '{simulation_type}' is not supported.")

        # Set up the input file for the simulation
        self.simulation.setup_input_file()

    def start_simulation(self, overwrite=False, local=False):
        """
        Start the DFT simulation with the appropriate command.
        """
        if(local == True):
            command = "chmod 777 run_dft_local.sh && ./run_dft_local.sh"
        else:
            command = "chmod 777 run_dft.sh && ./run_dft.sh"
        
        return super().start_simulation(command, overwrite)
    def set_model_file(self, model_file):
        """
        Set the model file for the DFT simulation.

        :param model_file: The path to the model file.
        """
        # Define the target file name in the simulation directory
        target_file = "POSCAR"

        # Check if the source model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' does not exist.")

        # Copy and rename the model file to 'data.lammps' in the simulation directory
        shutil.copy(model_file, target_file)
        

        # Register the renamed file in the simulation directory
        super().register_simulation_file(target_file)
        print(f"Model file '{model_file}' copied and renamed to {target_file} in the simulation directory.")
        #remove the temporary data.lammps
        os.remove(target_file)
     