import os
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser
from .Simulation import CompleteLammpsSimulation

class ElasticTensorSimulation(CompleteLammpsSimulation):
    simulation_type = "ElasticTensor"
    result_type = "elastic_tensor"
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
        self.parameters = {
            'atom_style': 'atomic',
            'change_box': '',
            'box_relax_settings': '',
            'uf3_degree': '3',
            'uf3_file': 'AlN.uf3',
            'uf3_elements': 'Al N',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000'
        }
        if parameters is not None:
            self.parameters.update(parameters)
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, parameters=self.parameters)
    def process_simulation_results(self):
        #initialize 
        lammps_data = {
            "elastic_tensor": None,
            "bulk_modulus_voigt": None,
            "bulk_modulus_reuss": None,
            "bulk_modulus_hill": None,
            "shear_modulus_voigt": None,
            "shear_modulus_reuss": None,
            "shear_modulus_hill": None,
            }
        file_path = os.path.join(self.simulation_directory, "results.txt")
        # Initialize an empty elastic tensor
        elastic_tensor = {}

        # Regex to capture the elastic constant and its value
        elastic_pattern = re.compile(r"Elastic Constant (\w+) = ([\d\.\-\+eE]+) GPa")
        
        # Regex to capture bulk modulus, shear moduli, and Poisson ratio
        bulk_modulus_pattern = re.compile(r"Bulk Modulus = ([\d\.\-\+eE]+) GPa")
        shear_modulus1_pattern = re.compile(r"Shear Modulus 1 = ([\d\.\-\+eE]+) GPa")
        shear_modulus2_pattern = re.compile(r"Shear Modulus 2 = ([\d\.\-\+eE]+) GPa")
        poisson_ratio_pattern = re.compile(r"Poisson Ratio = ([\d\.\-\+eE]+)")
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()
            print(f"read file: {file_path}: {lines}")
            for line in lines:
                # Match elastic tensor components
                elastic_match = elastic_pattern.search(line)
                if elastic_match:
                    elastic_constant, value = elastic_match.groups()
                    elastic_tensor[elastic_constant] = float(value)
                
                # Match bulk modulus
                bulk_modulus_match = bulk_modulus_pattern.search(line)
                if bulk_modulus_match:
                    lammps_data["bulk_modulus_voigt"] = float(bulk_modulus_match.group(1))
                
                # Match shear moduli
                shear_modulus1_match = shear_modulus1_pattern.search(line)
                if shear_modulus1_match:
                    lammps_data["shear_modulus_voigt"] = float(shear_modulus1_match.group(1))
                
                shear_modulus2_match = shear_modulus2_pattern.search(line)
                if shear_modulus2_match:
                    lammps_data["shear_modulus_reuss"] = float(shear_modulus2_match.group(1))
                
                # We aren't extracting Poisson ratio for now, but you can include it if needed
                # poisson_ratio_match = poisson_ratio_pattern.search(line)

            # Store the elastic tensor as a dictionary
            if elastic_tensor:
                lammps_data["elastic_tensor"] = elastic_tensor
            
            # Calculate the Hill averages for bulk modulus and shear modulus if both Voigt and Reuss are available
            if lammps_data["bulk_modulus_voigt"] and lammps_data["bulk_modulus_reuss"]:
                lammps_data["bulk_modulus_hill"] = (lammps_data["bulk_modulus_voigt"] + lammps_data["bulk_modulus_reuss"]) / 2
            
            if lammps_data["shear_modulus_voigt"] and lammps_data["shear_modulus_reuss"]:
                lammps_data["shear_modulus_hill"] = (lammps_data["shear_modulus_voigt"] + lammps_data["shear_modulus_reuss"]) / 2
            self.result = lammps_data
            return lammps_data
        except: 
            return {}
            