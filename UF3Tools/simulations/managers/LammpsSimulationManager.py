from .SimulationManager import SimulationManager
import os
import shutil
import numpy as np
from ..simulations.RelaxationSimulation import RelaxationSimulation
from ..simulations.LangevinSimulation import LangevinSimulation
from ..simulations.SurfaceEnergySimulation import SurfaceEnergySimulation
from ..simulations.ElasticTensorSimulation import ElasticTensorSimulation
from ..simulations.CohesiveEnergySimulation import CohesiveEnergySimulation
from ..simulations.BaseRelaxationSimulation import *
class LammpsSimulationManager(SimulationManager):
    def __init__(self):
        # Automatically set the parent directory to "/lammps"
        super().__init__("lammps")
    def create_simulation(self, simulation_type, **kwargs):
        """
        Create a LAMMPS simulation based on the simulation type.

        :param simulation_type: The type of simulation to set up (e.g., "relaxation").
        :param kwargs: Additional arguments needed for the specific simulation.
        """
        print("creating simulation")
        if simulation_type.lower() == "relaxation":
            template_directory = os.path.join(self.parent_directory, "templates/lattice_parameters/")
            self.simulation = RelaxationSimulation(self.simulation_dir, template_directory, **kwargs)
        elif simulation_type.lower() == "langevin":
            template_directory = os.path.join(self.parent_directory, "templates/langevin/")
            self.simulation = LangevinSimulation(self.simulation_dir, template_directory, **kwargs)
        elif simulation_type.lower() == "surface_energy":
            template_directory = os.path.join(self.parent_directory, "templates/surface_energy/")
            self.simulation = SurfaceEnergySimulation(self.simulation_dir, template_directory, **kwargs)
        elif simulation_type.lower() == "elastic_tensor":
            template_directory = os.path.join(self.parent_directory, "templates/elastic_tensor/")
            self.simulation = ElasticTensorSimulation(self.simulation_dir, template_directory, **kwargs)
        elif simulation_type.lower() == "cohesive_energy":
            template_directory = os.path.join(self.parent_directory, "templates/cohesive_energy/")
            self.simulation = CohesiveEnergySimulation(self.simulation_dir, template_directory, **kwargs)
        elif simulation_type.lower() == "base_relaxation":
            template_directory = os.path.join(self.parent_directory, "templates/base_relaxation/")
            self.simulation = BaseRelaxationSimulation(self.simulation_dir, template_directory, **kwargs)
        else:
            raise ValueError(f"Simulation type '{simulation_type}' is not supported.")
        print("object created")
        # Set up the input file for the simulation
        self.simulation.setup_input_file()
        return self.simulation

    def start_simulation(self, overwrite=False, local=False):
        """
        Start the LAMMPS simulation with the appropriate command.
        """
        if isinstance(self.simulation, BaseSimulation):
            overwrite = False
        if(local == True):
            command = "chmod 777 startSimulation.sh && ./startSimulation.sh"
        else:
            command = "chmod 777 start_simulation_hpg.sh && ./start_simulation_hpg.sh"
        return super().start_simulation(command, overwrite)
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
        
        
    def set_potential_file(self, potential_file):
        """
        Set the potential file for the LAMMPS simulation.

        :param potential_file: The path to the potential file.
        """
        self.potential_file = potential_file
        self.register_simulation_file(potential_file)
        self.simulation.set_potential_file(potential_file)
        
    def retrieve_results(self):
        os.chdir(self.simulation_dir)
        try:
            results = self.simulation.process_simulation_results()
        except:
            results = {}
        return results
