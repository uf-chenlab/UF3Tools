from .PotentialValidation import *
from ..simulations.ElasticTensorSimulation import ElasticTensorSimulation

class ElasticTensorValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return None
    computational_weight=2
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path,**kwargs):
        self.simulation_type = "ElasticTensorSimulation"
        print("validation folder path in elastictensor: ", validation_folder_path)
        super().__init__(self.simulation_type, potential_file, lammps_model_file, dft_model_file, element_list,  validation_folder_path,use_dft=False)
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
        print("validation folder path in elastictensor: ", self.root_directory)
        print("my root directory is: ", self.lammps_manager.get_root_directory())
        print("my template directory is: ", self.lammps_manager.get_template_directory())
        self.lammps_manager.setup_simulation(self.simulation_type)
        self.simulation = ElasticTensorSimulation(self.lammps_manager.get_root_directory(), self.lammps_manager.get_template_directory(), **kwargs)
        self.initialize_simulation_files()
    def retrieve_dft_result(self, path):
        """
        Retrieve DFT results from the specified path.
        """
        print(f"looking for dft results in: {path}")
        dft_parser = DFTDataParser(path)
        self.dft_data = dft_parser.get_elastic_tensor(path)
        print(f"dft lattice: {self.dft_data}")
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
            self.lammps_results = self.simulation.run(overwrite=overwrite, local=local)

        # Proceed if both LAMMPS and DFT data are available (either newly run or from stored data)
        if self.lammps_results and ((self.use_dft and self.dft_data) or self.has_dft_data == True):
            if self.has_dft_data:
                dft_results = self.dft_data
            print("dft results: ", dft_results)
            # Initialize the dictionary for percent errors
            percent_error = {}

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
            print("lammps result: ", self.lammps_results)
            # Compare the LAMMPS and DFT elastic tensors by calculating percent errors
            if(isinstance(self.lammps_results, list)):
                self.lammps_results = self.lammps_results[0]
            lammps_elastic_tensor = self.lammps_results["elastic_tensor"]

            # Calculate percent error for the elastic tensor
            for key in dft_elastic_tensor_dict:
                lammps_value = lammps_elastic_tensor.get(key, None)
                dft_value = dft_elastic_tensor_dict.get(key, None)

                # Calculate percent error if both LAMMPS and DFT values exist
                if lammps_value is not None and dft_value is not None:
                    error = abs(lammps_value - dft_value) / dft_value * 100
                    percent_error[key] = error

            # Compare bulk moduli and shear moduli
            def calculate_modulus_error(lammps_modulus, dft_modulus, name):
                if lammps_modulus is not None and dft_modulus is not None:
                    error = abs(lammps_modulus - dft_modulus) / dft_modulus * 100
                    percent_error[name] = error
            
            # Bulk moduli comparison
            calculate_modulus_error(self.lammps_results.get("bulk_modulus_voigt"), dft_results.get("bulk_modulus_voigt"), "bulk_modulus_voigt")
            calculate_modulus_error(self.lammps_results.get("bulk_modulus_reuss"), dft_results.get("bulk_modulus_reuss"), "bulk_modulus_reuss")
            calculate_modulus_error(self.lammps_results.get("bulk_modulus_hill"), dft_results.get("bulk_modulus_hill"), "bulk_modulus_hill")

            # Shear moduli comparison
            calculate_modulus_error(self.lammps_results.get("shear_modulus_voigt"), dft_results.get("shear_modulus_voigt"), "shear_modulus_voigt")
            calculate_modulus_error(self.lammps_results.get("shear_modulus_reuss"), dft_results.get("shear_modulus_reuss"), "shear_modulus_reuss")
            calculate_modulus_error(self.lammps_results.get("shear_modulus_hill"), dft_results.get("shear_modulus_hill"), "shear_modulus_hill")
            
            return percent_error

        # If results are missing, throw an error
        else:
            print("missing elastic tensor results")
            return {
                "C11all": None,
                "C12all": None,
                "C13all": None,
                "C22all": None,
                "C23all": None,
                "C33all": None,
                "C44all": None,
                "C55all": None,
                "C66all": None,
                "bulk_modulus_voigt": None,
                "bulk_modulus_reuss": None,
                "bulk_modulus_hill": None,
                "shear_modulus_voigt":None,
                "shear_modulus_reuss":None,
                "shear_modulus_hill":None
            }


    