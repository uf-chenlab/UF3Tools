from .SimulationManager import SimulationManager
import os
import shutil
import numpy as np
from ..simulations.Simulation import LammpsSimulation

class LammpsSimulationFileManager(SimulationManager):
    def __init__(self):
        # Automatically set the parent directory to "/lammps"
        self.original_directory = os.getcwd()
        self.base_directory = os.path.join(os.getcwd(), "lammps")
        super().__init__("lammps")
    def setup_simulation_directory(self, simulation_type):
        sim_dir = os.path.join(self.base_directory, simulation_type)
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
        self.simulation_dir = sim_dir
        return sim_dir
    
    def set_model_file(self, model_file):
        """
        Set the model file for the LAMMPS simulation.

        :param model_file: The path to the model file.
        """
        # Define the target file name in the simulation directory
        target_file = "data.lammps"

        # Check if the source model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' does not exist.")

        # Copy and rename the model file to 'data.lammps' in the simulation directory
        shutil.copy(model_file, target_file)
        

        # Register the renamed file in the simulation directory
        super().register_simulation_file(target_file)
        #remove the temporary data.lammps
        os.remove(target_file)
        
        
    def set_potential_file(self, potential_file, simulation):
        """
        Set the potential file for the LAMMPS simulation.

        :param potential_file: The path to the potential file.
        """
        self.potential_file = potential_file
        self.register_simulation_file(potential_file)
        if isinstance(simulation, type):  # If `simulation` is a class, create an instance
            raise TypeError(f"Expected an instance of LammpsSimulation, but got a class {simulation}")
        simulation.set_potential_file(potential_file)
        