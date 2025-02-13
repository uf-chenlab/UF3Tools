import os 
import shutil 
import numpy as np
from ..managers.DFTSimulationManager import DFTSimulationManager
from ..managers.LammpsSimulationFileManager import LammpsSimulationFileManager   
from ..simulations.Simulation import LammpsSimulation
from ..common.DFTDataParser import DFTDataParser
import ast

class PotentialValidation:
    computational_weight = 0
    required_base_simulations = []
    simulation = None
    
    @classmethod
    def get_computational_weight(cls):
        return cls.computational_weight
    
    @classmethod
    def get_required_base_simulations(cls):
        return cls.required_base_simulations
    
    def __init__(self, simulation_type, potential_file, lammps_model_file, dft_model_file, element_list, root_directory, use_dft=False):
        #initialize variables
        print("potentialvalidation init")
        self.potential_file = potential_file
        print("potential file: ", self.potential_file)
        self.lammps_model_file = lammps_model_file
        print("lammps model file: ", self.lammps_model_file)
        self.dft_model_file = dft_model_file
        print("dft model file: ", self.dft_model_file)
        self.element_list = element_list
        print("element list: ", self.element_list)
        self.use_dft = use_dft
        print("use dft: ", self.use_dft)
        self.has_dft_data = False
        
        self.simulation_type = simulation_type
        self.simulation = None
        print("variables set up")
        
        #initialize folder structures
        self.original_directory = os.getcwd()
        print("printing original directory")
        print(self.original_directory)
        self.root_directory = root_directory
        os.makedirs(self.root_directory, exist_ok=True)
        os.chdir(self.root_directory)
        print("printing root directory")
        print(self.root_directory)
        #initialize simulation managers
        self.lammps_manager = LammpsSimulationFileManager()
        self.dft_manager = DFTSimulationManager()
        self.dft_result = {}
        self.lammps_results = []
    def get_simulation(self):
        return self.simulation
    def start_validation(self, overwrite=True, local=False):
        """
        Start the validation process by running the LAMMPS and DFT simulations.
        """
        if(self.lammps_results == []):
            self.lammps_results = self.simulation.run_simulation()
        
        # Initialize DFT results
        dft_results = None
        if self.use_dft and not self.has_dft_data:
            dft_results = self.dft_manager.start_simulation(overwrite=overwrite, local=local)
        elif self.has_dft_data:
            dft_results = self.dft_result.get(self.simulation_type, None)
            if dft_results is None:
                raise ValueError(f"DFT results missing for simulation type: {self.simulation_type}")

        if self.lammps_results and ((self.use_dft and dft_results) or self.has_dft_data):
            percent_error = []
            
            for lammps in self.lammps_results:
                
                # Ensure lammps is a dictionary
                if not isinstance(lammps, dict):
                    raise TypeError(f"Expected dict for LAMMPS, got {type(lammps)}")
                if not isinstance(dft_results, dict):
                    raise TypeError(f"Expected dict for DFT, got {type(dft_results)}")

                # Calculate percent error
                error_dict = {}
                for key in lammps:
                    if key in dft_results:
                        error_dict[key] = abs(lammps[key] - dft_results[key]) / dft_results[key] * 100
                    else:
                        print(f"Key '{key}' not found in DFT results, skipping.")
                percent_error.append(error_dict)

            return percent_error
        elif self.lammps_results and not self.use_dft:
            return self.lammps_results
        else:
            raise ValueError("LAMMPS or DFT results are missing.")
    def retrieve_dft_result(self, path):
        # subclass should implement this
        raise NotImplementedError("Subclasses should implement this method")
    def retrieve_lammps_result(self):
        return self.simulation.get_result()
    def import_base_simulations(self, simulations):
        self.base_results = simulations      

class ValidationData:
    def __init__(self, potential_version, potential_file, structure):
        self.potential_version = potential_version
        self.potential_file = potential_file
        self.structure = structure
        self.validation_data = {}
        self.normalized_error = None
    def add_validation(self, validation_type, validation_results):
        self.validation_data[validation_type] = validation_results
    def get_potential_version(self):
        return self.potential_version
    def get_validation_data(self):
        return self.validation_data
    def get_potential_file(self):
        return self.potential_file
    def getValidation(self, validation_type):
        return self.validation_data.get(validation_type, None)
    def set_normalized_error(self, error):
        self.normalized_error = error
    def get_normalized_error(self):
        return self.normalized_error
    def __str__(self):
        output = f"Potential version: {self.potential_version}\n"
        output += f"Structure: {self.structure}\n"
        output += f'Normalized error: {self.normalized_error}\n'
        for validation_type, validation_results in self.validation_data.items():
            output += f"{validation_type} validation results:\n"
            # Check if validation_results is a list of dictionaries
            if isinstance(validation_results, list):
                for result in validation_results:
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, (int, float)):  # Check if the value is numeric
                                output += f"  {key}: {value:.6f}%\n"
                            else:
                                output += f"  {key}: {value}\n"  # Handle non-numeric values differently
                    else:
                        output += f"  {result}\n"  # Handle non-dictionary list items
            else:
                # Handle case where validation_results is a single dictionary
                for key, value in validation_results.items():
                    if isinstance(value, (int, float)):  # Check if the value is numeric
                        output += f"  {key}: {value:.6f}%\n"
                    else:
                        output += f"  {key}: {value}\n"  # Handle non-numeric values differently
        return output
    def save_to_file(self, directory):
        with open(os.path.join(directory, f"{self.potential_file}_{self.potential_version}_validation.txt"), "w") as file:
            file.write(str(self))
        
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
