from .PotentialValidation import *
import re

class ElasticTensorValidation(PotentialValidation):
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path,**kwargs):
        self.default_kwargs = {
            'min_style': 'cg',
            'uf3_elements': element_list
        }
        # Update the default keyword arguments with the user-specified arguments
        self.kwargs = self.default_kwargs
        self.kwargs.update(kwargs)
        self.dft_data = {
            "elastic_tensor": None,
            "bulk_modulus_voigt": None,
            "bulk_modulus_reuss": None,
            "bulk_modulus_hill": None,
            "shear_modulus_voigt": None,
            "shear_modulus_reuss": None,
            "shear_modulus_hill": None,
        }
        self.lammps_data = {
            "elastic_tensor": None,
            "bulk_modulus_voigt": None,
            "bulk_modulus_reuss": None,
            "bulk_modulus_hill": None,
            "shear_modulus_voigt": None,
            "shear_modulus_reuss": None,
            "shear_modulus_hill": None,
        }
        self.use_dft = False
        super().__init__(potential_file, lammps_model_file, dft_model_file, element_list,  validation_folder_path,use_dft=False)
        
    def initialize_simulation_files(self):
        self.lammps_manager.setup_simulation("elastic_tensor", **self.kwargs)
        os.chdir(self.original_directory)
        self.lammps_manager.set_potential_file(self.potential_file)
        self.lammps_manager.set_model_file(self.lammps_model_file)
    def start_validation(self, overwrite=True, local=False):
        """
        Start the validation process by running the LAMMPS and DFT simulations and comparing results.
        """
        self.lammps_manager.set_potential_file(self.potential_file)
        self.lammps_manager.set_model_file(self.lammps_model_file)
        lammps_results = self.lammps_manager.start_simulation(overwrite=overwrite, local=local)

        # Proceed if both LAMMPS and DFT data are available (either newly run or from stored data)
        if lammps_results and ((self.use_dft and dft_results) or self.has_dft_data == True):
            if self.has_dft_data:
                dft_results = self.dft_result[self.simulation_type]
            # Initialize the list for percent errors
            percent_error = []

            # Convert the DFT elastic tensor matrix into a dictionary format similar to LAMMPS
            dft_elastic_tensor_dict = {
                "C11all": dft_results["elastic_tensor"][0][0],
                "C12all": dft_results["elastic_tensor"][0][1],
                "C13all": dft_results["elastic_tensor"][0][2],
                "C22all": dft_results["elastic_tensor"][1][1],
                "C23all": dft_results["elastic_tensor"][1][2],
                "C33all": dft_results["elastic_tensor"][2][2],
                "C44all": dft_results["elastic_tensor"][3][3],
                "C55all": dft_results["elastic_tensor"][4][4],
                "C66all": dft_results["elastic_tensor"][5][5],
            }

            # Compare the LAMMPS and DFT elastic tensors by calculating percent errors
            lammps_elastic_tensor = lammps_results[0]["elastic_tensor"]

            # Calculate percent error for the elastic tensor
            for key in dft_elastic_tensor_dict:
                lammps_value = lammps_elastic_tensor.get(key, None)
                dft_value = dft_elastic_tensor_dict.get(key, None)

                # Calculate percent error if both LAMMPS and DFT values exist
                if lammps_value is not None and dft_value is not None:
                    error = abs(lammps_value - dft_value) / dft_value * 100
                    percent_error.append({key: error})

            # Compare bulk moduli and shear moduli
            def calculate_modulus_error(lammps_modulus, dft_modulus, name):
                if lammps_modulus is not None and dft_modulus is not None:
                    error = abs(lammps_modulus - dft_modulus) / dft_modulus * 100
                    percent_error.append({name: error})

            # Bulk moduli comparison
            calculate_modulus_error(lammps_results[0].get("bulk_modulus_voigt"), dft_results.get("bulk_modulus_voigt"), "bulk_modulus_voigt")
            calculate_modulus_error(lammps_results[0].get("bulk_modulus_reuss"), dft_results.get("bulk_modulus_reuss"), "bulk_modulus_reuss")
            calculate_modulus_error(lammps_results[0].get("bulk_modulus_hill"), dft_results.get("bulk_modulus_hill"), "bulk_modulus_hill")

            # Shear moduli comparison
            calculate_modulus_error(lammps_results[0].get("shear_modulus_voigt"), dft_results.get("shear_modulus_voigt"), "shear_modulus_voigt")
            calculate_modulus_error(lammps_results[0].get("shear_modulus_reuss"), dft_results.get("shear_modulus_reuss"), "shear_modulus_reuss")
            calculate_modulus_error(lammps_results[0].get("shear_modulus_hill"), dft_results.get("shear_modulus_hill"), "shear_modulus_hill")
            
            return percent_error
        
        # If results are missing, throw an error
        else:
            raise ValueError("LAMMPS or DFT results are missing.")


    