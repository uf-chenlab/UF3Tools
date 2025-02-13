from .PotentialValidation import *
from ..simulations.CohesiveEnergySimulation import CohesiveEnergySimulation
import re

class CohesiveEnergyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return CohesiveEnergySimulation.get_required_sub_simulations()
    computational_weight=1
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path,**kwargs):
        self.simulation_type = "CohesiveEnergySimulation"
        super().__init__(self.simulation_type, potential_file, lammps_model_file, dft_model_file, element_list,  validation_folder_path,use_dft=False)
        self.default_kwargs = {
            'min_style': 'cg',
            'uf3_elements': element_list
        }
        # Update the default keyword arguments with the user-specified arguments
        self.kwargs = self.default_kwargs
        self.kwargs.update(kwargs)
        self.dft_data = {
            "cohesive_energy": None,
        }
        self.lammps_data = {
            "bulk_energy": None,
            "num_atoms": None,
            "cohesive_energy": None
        }
        self.use_dft = False
        
        self.lammps_manager.setup_simulation(self.simulation_type)
        self.simulation = CohesiveEnergySimulation(self.lammps_manager.get_root_directory(), self.lammps_manager.get_template_directory(), None, **kwargs)
        self.required_base_simulations = self.simulation.get_required_sub_simulations()
        self.initialize_simulation_files()
    def initialize_simulation_files(self):
        os.chdir(self.original_directory)
        self.lammps_manager.set_potential_file(self.potential_file, self.simulation)
        self.lammps_manager.set_model_file(self.lammps_model_file)
    def retrieve_dft_result(self, path):
        """
        Retrieve DFT results from the specified path.
        """
        print(f"looking for dft results in: {path}")
        dft_parser = DFTDataParser(path)
        self.dft_data = dft_parser.get_cohesive_energy(path)
        print(f"cohesive energy: {self.dft_data['cohesive_energy']}")
        self.has_dft_data = True
        
    def start_validation(self, overwrite=True, local=False):
        """
        Start the validation process by running the LAMMPS and DFT simulations and comparing results.
        """
        if self.lammps_results == []:
            os.chdir(self.original_directory)
            self.lammps_manager.set_potential_file(self.potential_file, self.simulation)
            self.lammps_manager.set_model_file(self.lammps_model_file)
            os.chdir(self.root_directory)
            self.lammps_results = self.simulation.run_simulation(overwrite=overwrite, local=local)
        print(f"lammps results: {self.lammps_results}")
        # Proceed if both LAMMPS and DFT data are available (either newly run or from stored data)
        if self.lammps_results and ((self.use_dft and self.dft_data) or self.has_dft_data == True):
            print("entered if")
            if self.has_dft_data:
                if isinstance(self.lammps_results, list):
                    self.lammps_results = self.lammps_results[0]
                print("enetered dft data int ")
                print(f"lammps cohesive energy: {self.lammps_results['cohesive_energy']}")
                print(f'dft result: {self.dft_data}')
                print(f"dft cohesive energy: {self.dft_data['cohesive_energy']}")
                # Initialize the dictionary for percent errors
                percent_error = {}

                # Compare the LAMMPS and DFT elastic tensors by calculating percent errors
                lammps_cohesive_energy = self.lammps_results["cohesive_energy"]

                percent_error["cohesive_energy"] = abs((lammps_cohesive_energy - self.dft_data["cohesive_energy"]) / self.dft_data["cohesive_energy"]) * 100
                return percent_error

        # If results are missing, throw an error
        print("NO DFT DATA FOUND IDIOT")
        return {
            'cohesive_energy': 99999999
        }


    