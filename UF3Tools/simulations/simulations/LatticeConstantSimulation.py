import os
from .Simulation import CompoundSimulation
from .BaseRelaxationSimulation import *

class LatticeConstantSimulation(CompoundSimulation):
    required_sub_simulations = [BaseRelaxationSimulation]
    simulation_type = "LatticeConstant"
    result_type = "lattice_constant"
    def __init__(self, simulation_directory, input_file, sub_simulations=None, parameters=None):
        self.original_directory = os.getcwd()
        if(self.simulation_type not in simulation_directory):
            simulation_directory = os.path.join(simulation_directory, (self.simulation_type+'Simulation'))
        self.simulation_directory = os.path.abspath(simulation_directory)
        self.input_file = os.path.join(simulation_directory, input_file)

        self.parameters = {
            "atom_style": "atomic",
            "x_boundary": "p",
            "y_boundary": "p",
            "z_boundary": "p",
            "change_box": "",
            "uf3_degree": "3",
            "uf3_file": "AlN.uf3",
            "uf3_elements": "Al N",
            "box_relax_settings": "fix 1 all box/relax iso 0.0 vmax 0.001",
            "min_style": "cg",
            "etol": "1e-25",
            "ftol": "1e-25",
            "maxiter": "10000",
            "maxeval": "50000",
        }
        if parameters is not None:
            self.parameters.update(parameters)
        super().__init__(self.simulation_directory, input_file, sub_simulations, self.parameters)

    def process_simulation_results(self):
        if not self.sub_simulations:
            return {}
        if "BaseRelaxation" in self.sub_simulations:
            sim = self.sub_simulations["BaseRelaxation"]
            self.result = {
                self.result_type: sim.get_result()[self.result_type]
            }
        return self.result
    

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