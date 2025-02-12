import os
import re
from .Simulation import LammpsSimulation
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser

class RelaxationSimulation(LammpsSimulation):
    def __init__(self, simulation_directory, template_directory, **kwargs):
        parameters = {
            'atom_style': 'atomic',
            'uf3_degree': '3',
            'uf3_file': 'AlN.uf3',
            'uf3_elements': 'Al N',
            'change_box': '',
            'box_relax_settings': 'fix 1 all box/relax iso 0.0 vmax 0.001',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000'
        }
        super().__init__(simulation_directory, template_directory, "in.relax", "lammps.in", parameters, **kwargs)
        self.lattice_parameters = {}
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        return os.path.exists(os.path.join(self.simulation_directory, "lattice.txt"))
    def process_simulation_results(self):
        lattice_file_path = os.path.join(self.simulation_directory, "lattice.txt")
        lattice_parameters = {}
        try:
            with open(lattice_file_path, "r") as file:
                lattice_text = file.read()
            pattern = r'a:\s*([\d.]+)\s*b:\s*([\d.]+)\s*c:\s*([\d.]+)\s*alpha:\s*([\d.]+)\s*beta:\s*([\d.]+)\s*gamma:\s*([\d.]+)'
            match = re.search(pattern, lattice_text)
            if match:
                lattice_parameters = {key: float(match.group(i+1)) for i, key in enumerate(['a', 'b', 'c', 'alpha', 'beta', 'gamma'])}
                self.lattice_parameters = lattice_parameters
            else:
                print("Error: Lattice parameters could not be extracted.")
        except:
            lattice_parameters = {key: 999 for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']}
        self.result = lattice_parameters
        return lattice_parameters
    

class DFTRelaxationSimulation():
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
            'lreal': '.FALSE.',
            'ismear': '0',
            'sigma': '0.05',
            'npar': '4',
            'nsw': '1',
            'ncore': '8',
            'encut': '600',
            'extras': ''
        }
        self.parameters.update(kwargs)
        self.lattice_parameters = {}
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
        
        parser = DFTDataParser(self.simulation_directory)
        lattice_parameters = parser.get_lattice_constants()

        return lattice_parameters