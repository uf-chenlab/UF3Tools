import os 
import shutil 
import numpy as np
from ..managers.DFTSimulationManager import DFTSimulationManager
from ..managers.LammpsSimulationFileManager import LammpsSimulationFileManager  
from ..common.LammpsInputFileHandler import LammpsInputFileHandler 
import ast

class PotentialValidation:
    computational_weight = 0
    required_base_simulations = []
    required_single_crystal_simulations = []
    simulation = None
    required_supercell = (1, 1, 1)

    @classmethod
    def get_computational_weight(cls):
        return cls.computational_weight

    @classmethod
    def get_required_base_simulations(cls):
        return cls.required_base_simulations
    
    @classmethod
    def get_required_single_crystal_simulations(cls):
        return cls.required_single_crystal_simulations

    @classmethod
    def get_required_supercell(cls):
        return cls.required_supercell

    def __init__(self, simulation_type, potential_file, lammps_model_file, dft_model_file, element_list, root_directory, run_scripts, use_dft=False):
        # Ensure all file paths are absolute
        self.potential_file = os.path.abspath(potential_file)
        self.lammps_model_file = os.path.abspath(lammps_model_file)
        self.dft_model_file = os.path.abspath(dft_model_file)
        self.root_directory = os.path.abspath(root_directory)

        print(f"Potential file: {self.potential_file}")
        print(f"LAMMPS model file: {self.lammps_model_file}")
        print(f"DFT model file: {self.dft_model_file}")
        print(f"Root directory: {self.root_directory}")

        self.element_list = element_list
        print(f"Element list: {self.element_list}")

        self.use_dft = use_dft
        print(f"Use DFT: {self.use_dft}")

        self.has_dft_data = False
        self.simulation_type = simulation_type
        self.simulation = None
        self.run_scripts = run_scripts

        # Ensure root directory exists
        os.makedirs(self.root_directory, exist_ok=True)

        # Initialize simulation managers with absolute paths
        self.lammps_manager = None
        self.dft_manager = None
        self.dft_data = {}
        self.lammps_results = []
    def setup_input_file(self, template_file, output_file, parameters=None):
        handler = LammpsInputFileHandler(template_file, output_file, parameters)
        handler.generate_input_file()
    def get_simulation(self):
        return self.simulation

    def start_validation(self, overwrite=True, local=False):
        if not self.lammps_results:
            print(f"Running LAMMPS simulation in: {self.simulation.get_directory()}")
            self.lammps_results = self.simulation.run(overwrite=overwrite, local=local)

        print(f"LAMMPS results: {self.lammps_results}")

        if self.lammps_results and ((self.use_dft and self.dft_data) or self.has_dft_data):
            if self.has_dft_data:
                if isinstance(self.lammps_results, list):
                    self.lammps_results = self.lammps_results[0]

                print(f"LAMMPS result: {self.lammps_results.get(self.simulation.get_result_type(), 'N/A')}")
                print(f"DFT result: {self.dft_data}")

                percent_error = {}
                result_type = self.simulation.get_result_type()

                if result_type in self.lammps_results:
                    self.lammps_results = self.lammps_results[result_type]

                for key in self.lammps_results:
                    dft_value = self.dft_data.get(key)
                    lammps_value = self.lammps_results[key]

                    if dft_value is not None and dft_value != 0:
                        percent_error[key] = self.calculate_percent_error(dft_value, lammps_value)
                    else:
                        percent_error[key] = None

                return percent_error

        print("No DFT data found")
        return {self.simulation.get_result_type(): None}

    def retrieve_dft_result(self, path):
        raise NotImplementedError("Subclasses should implement this method")

    def retrieve_lammps_result(self):
        return self.simulation.get_result()

    def import_base_simulations(self, simulations):
        self.base_results = simulations

    def get_lammps_validation_results(self):
        return self.lammps_results
    def get_dft_validation_results(self):
        return self.dft_data

    def calculate_percent_error(self, dft_data, lammps_data):
        if dft_data is None or dft_data == 0:
            return None
        if lammps_data is None:
            return None
        return abs(abs(lammps_data - dft_data) / dft_data * 100)

class ValidationData:
    def __init__(self, potential_version, potential_file, structure):
        self.potential_version = potential_version
        self.potential_file = potential_file
        self.structure = structure
        self.validation_data = {}
        self.lammps_data = {}
        self.relative_order_results = {}
        self.normalized_error = None

        # Storing raw data separately
        self.raw_lammps_data = {}
        self.raw_dft_data = {}

    def add_validation(self, validation_type, validation_results):
        self.validation_data[validation_type] = validation_results

    def add_raw_data(self, validation_type, lammps_raw, dft_raw):
        if validation_type not in self.raw_lammps_data:
            self.raw_lammps_data[validation_type] = []
        if validation_type not in self.raw_dft_data:
            self.raw_dft_data[validation_type] = []
        self.raw_lammps_data[validation_type].append(lammps_raw)
        self.raw_dft_data[validation_type].append(dft_raw)

    def get_structure_id(self):
        return self.structure

    def get_potential_version(self):
        return self.potential_version

    def get_potential_file(self):
        return self.potential_file

    def get_validation_data(self):
        return self.validation_data

    def getValidation(self, validation_type):
        return self.validation_data.get(validation_type, None)

    def set_normalized_error(self, error):
        self.normalized_error = error

    def get_normalized_error(self):
        return self.normalized_error

    def set_lammps_validation_results(self, lammps_data):
        self.lammps_data = lammps_data

    def add_lammps_validation_result(self, validation_type, lammps_data):
        self.lammps_data[validation_type] = lammps_data

    def get_lammps_validation_results(self):
        return self.lammps_data

    def get_relative_order_results(self):
        return self.relative_order_results

    def set_relative_order_results(self, relative_order_results):
        self.relative_order_results = relative_order_results

    def __str__(self):
        output = f"Potential version: {self.potential_version}\n"
        output += f"Structure: {self.structure}\n"
        output += f"Normalized error: {self.normalized_error}\n"
        output += f"Relative Order: {self.relative_order_results}\n"
        for validation_type, validation_results in self.validation_data.items():
            output += f"{validation_type} validation results:\n"
            if isinstance(validation_results, list):
                for result in validation_results:
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, (int, float)):
                                output += f"  {key}: {value:.6f}%\n"
                            else:
                                output += f"  {key}: {value}\n"
                    else:
                        output += f"  {result}\n"
            else:
                for key, value in validation_results.items():
                    if isinstance(value, (int, float)):
                        output += f"  {key}: {value:.6f}%\n"
                    else:
                        output += f"  {key}: {value}\n"
        return output

    def save_to_file(self, directory):
        path = os.path.join(directory, f"{self.potential_file}_{self.potential_version}_validation.txt")
        with open(path, "w") as file:
            file.write(str(self))

    def save_raw_data_to_file(self, directory):
        base_filename = base_filename = self.potential_version.split("_")[0]
        #trim the filename so it doenst have all of the parameters
        
        # Save LAMMPS data
        lammps_path = os.path.join(directory, f"{base_filename}_lammps_raw_data.txt")
        with open(lammps_path, "w") as file:
            file.write(f"Potential version: {self.potential_version}\n")
            file.write(f"Structure: {self.structure}\n\n")
            file.write("Raw LAMMPS Data:\n")
            for validation_type, data_list in self.raw_lammps_data.items():
                file.write(f"  {validation_type}:\n")
                for data in data_list:
                    for k, v in data.items():
                        file.write(f"    {k}: {v}\n")

        # Save DFT data
        dft_path = os.path.join(directory, f"{base_filename}_dft_raw_data.txt")
        with open(dft_path, "w") as file:
            file.write(f"Potential version: {self.potential_version}\n")
            file.write(f"Structure: {self.structure}\n\n")
            file.write("Raw DFT Data:\n")
            for validation_type, data_list in self.raw_dft_data.items():
                file.write(f"  {validation_type}:\n")
                for data in data_list:
                    for k, v in data.items():
                        file.write(f"    {k}: {v}\n")
        
class ValidationDataCompare:
    def __init__(self, validation_1, validation_2):
        self.validation_1 = validation_1
        self.validation_2 = validation_2
    
    def calculate_difference(self):
        difference = {}
        validation_data_1 = self.validation_1.get_validation_data()
        validation_data_2 = self.validation_2.get_validation_data()

        for validation_type in validation_data_1:
            difference[validation_type] = []
            # Check if both validation results are lists (for handling multiple entries)
            if isinstance(validation_data_1[validation_type], list) and isinstance(validation_data_2[validation_type], list):
                for val1, val2 in zip(validation_data_1[validation_type], validation_data_2[validation_type]):
                    diff_dict = {}
                    for key in val1:
                        value1 = val1[key]
                        value2 = val2[key]
                        try:
                            # Attempt to convert both values to float
                            value1 = float(value1)
                            value2 = float(value2)
                            # Calculate the absolute difference
                            diff_dict[key] = abs(value1 - value2)
                        except (ValueError, TypeError):
                            # If conversion fails, set the difference to None
                            diff_dict[key] = None
                            print(f"Warning: Could not convert '{key}' values to float in '{validation_type}'. Difference set to None.")
                    difference[validation_type].append(diff_dict)
            else:
                # Handle the case where validation results are single dictionaries
                diff_dict = {}
                for key in validation_data_1[validation_type]:
                    value1 = validation_data_1[validation_type][key]
                    value2 = validation_data_2[validation_type][key]
                    try:
                        # Attempt to convert both values to float
                        value1 = float(value1)
                        value2 = float(value2)
                        # Calculate the absolute difference
                        diff_dict[key] = abs(value1 - value2)
                    except (ValueError, TypeError):
                        # If conversion fails, set the difference to None
                        diff_dict[key] = None
                        print(f"Warning: Could not convert '{key}' values to float in '{validation_type}'. Difference set to None.")
                difference[validation_type] = diff_dict
        
        return difference
