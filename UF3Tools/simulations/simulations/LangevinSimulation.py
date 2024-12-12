import os 
import shutil
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser

class LangevinSimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the LangevinSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the Langevin simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "in.langevin"
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "lammps.in")
        self.parameters = {
            'atom_style': 'atomic',
            'uf3_degree': '3',
            'uf3_file': 'AlN.uf3',
            'uf3_elements': 'Al N',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000',
            'timestep': '0.001',
            'force_dump_steps': '400000',
            'dump_steps': '200000',
            'temperature': '300',
            'pressure': '1',
            'run_steps': '200000',
            'npt_enable': '#',
        }
        self.parameters.update(kwargs)
        self.lattice_parameters = {}
    def set_potential_file(self, potential_file):
        """
        Set the potential file for the LAMMPS simulation.
        """
        self.parameters['uf3_file'] = potential_file
        self.setup_input_file()
    def setup_input_file(self):
        """
        Set up the LAMMPS input file for the relaxation simulation.
        """
        handler = LammpsInputFileHandler(self.template_file, self.output_file) 
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
        
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        init_lattice_exists = os.path.exists(os.path.join(self.simulation_directory, "initial_lattice.txt"))
        final_lattice_exists = os.path.exists(os.path.join(self.simulation_directory, "final_lattice.txt"))
        force_exists = os.path.exists(os.path.join(self.simulation_directory, "force_dump.txt"))
        return init_lattice_exists and final_lattice_exists and force_exists
       
    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
        

        return {'result': '1'}
        
