import os 
import shutil
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser
        
class CohesiveEnergySimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "in.cohesive"
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "lammps.in")
        self.parameters = {
            'atom_style': 'atomic',
            'x_boundary': 'p',
            'y_boundary': 'p',
            'z_boundary': 'p',
            'change_box': '',
            'uf3_degree': '3',
            'uf3_file': 'AlN.uf3',
            'uf3_elements': 'Al N',
            'box_relax_settings': 'fix 1 all box/relax iso 0.0 vmax 0.001',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000'
        }
        self.parameters.update(kwargs)
        self.cohesive_energy = {}
       
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

        :return: True if output file exists
        """
        
        print(f"existing cohesive results: {os.path.exists(self.simulation_directory + '/energy.txt')}")
        return os.path.exists(self.simulation_directory + "/energy.txt")
       
    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
         # Paths to the necessary files
        energy_file_path = os.path.join(self.simulation_directory, "energy.txt")
        results = {}

        try:
            # Read the contents of the energy.txt file
            with open(energy_file_path, "r") as file:
                energy_text = file.read()

            # Regular expression pattern to match the surface energy result
            energy_pattern = r'ecoh\s*=\s*([-.\d]+)'  # Modified pattern to handle negative numbers

            energy_match = re.search(energy_pattern, energy_text)

            if energy_match:
                energy = float(energy_match.group(1))
                results['cohesive_energy'] = energy
            else:
                print("Error: Surface energy could not be extracted. Please check the format of energy.txt.")
                results['cohesive_energy'] = None

            return results

        except:
            return {
                "cohesive_energy": 999999
            }
    
class DFTCohesiveEnergySimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the DFTRelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "INCAR"
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "INCAR")
        self.parameters = {
            'algo': 'Fast',
            'prec': 'Accurate',
            'ibrion': '2',
            'isif': '3',
            'lcharg': 'FALSE',
            'lwave': 'FALSE',
            'lreal': 'FALSE',
            'ismear': '0',
            'sigma': '0.05',
            'npar': '4',
            'nsw': '1',
            'ncore': '8',
            'encut': '600',
            'extras': ''
        }
        self.parameters.update(kwargs)
    def setup_input_file(self):
        """
        Set up the VASP INCAR file for the relaxation simulation.
        """
        handler = LammpsInputFileHandler(self.template_file, self.output_file) 
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        outcar_exists = os.path.exists(os.path.join(self.simulation_directory, "OUTCAR"))
        contcar_exists = os.path.exists(os.path.join(self.simulation_directory, "CONTCAR"))
        return outcar_exists and contcar_exists
    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
        
        results = {}
        return results