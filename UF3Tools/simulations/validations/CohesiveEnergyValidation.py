from .PotentialValidation import *
import re

class CohesiveEnergyValidation(PotentialValidation):
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path,**kwargs):
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
        super().__init__(potential_file, lammps_model_file, dft_model_file, element_list,  validation_folder_path,use_dft=False)
        
    def initialize_simulation_files(self):
        self.lammps_manager.setup_simulation("cohesive_energy", **self.kwargs)
        os.chdir(self.original_directory)
        self.lammps_manager.set_potential_file(self.potential_file)
        self.lammps_manager.set_model_file(self.lammps_model_file)
    def start_validation(self, overwrite=True, local=False):
        """
        Start the validation process by running the LAMMPS and DFT simulations and comparing results.
        """
        if self.lammps_results == []:
            self.lammps_manager.set_potential_file(self.potential_file)
            self.lammps_manager.set_model_file(self.lammps_model_file)
            self.lammps_results = self.lammps_manager.start_simulation(overwrite=overwrite, local=local)

        # Proceed if both LAMMPS and DFT data are available (either newly run or from stored data)
        if self.lammps_results and ((self.use_dft and dft_results) or self.has_dft_data == True):
            if self.has_dft_data:
                dft_results = self.dft_result[self.simulation_type]
            print(f"lammps cohesive energy: {self.lammps_results[0]['cohesive_energy']}")
            print(f"dft cohesive energy: {dft_results['cohesive_energy']}")
            # Initialize the dictionary for percent errors
            percent_error = {}

            # Compare the LAMMPS and DFT elastic tensors by calculating percent errors
            lammps_cohesive_energy = self.lammps_results[0]["cohesive_energy"]

            percent_error["cohesive_energy"] = abs((lammps_cohesive_energy - dft_results["cohesive_energy"]) / dft_results["cohesive_energy"]) * 100
            return percent_error

        # If results are missing, throw an error
        else:
            return {
                'cohesive_energy': 99999999
            }


    