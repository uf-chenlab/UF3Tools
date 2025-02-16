from .PotentialValidation import *
from ..simulations.ElasticTensorSimulation import ElasticTensorSimulation

class ElasticTensorValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return None
    computational_weight=2
    required_supercell = (4,4,4)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, **kwargs):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "ElasticTensorSimulation",
            potential_file,
            lammps_model_file,
            dft_model_file,
            element_list,
            abs_validation_path,
            use_dft=False
        )
        self.default_kwargs = {
            "min_style": "cg",
            "uf3_elements": element_list
        }
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
        self.lammps_data = dict(self.dft_data)
        self.use_dft = False

        self.lammps_manager.setup_simulation(self.simulation_type)
        self.simulation = ElasticTensorSimulation(
            self.lammps_manager.get_root_directory(),
            self.lammps_manager.get_template_directory(),
            **kwargs
        )
        self.initialize_simulation_files()

    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        print(f'path: {path}')
        abs_path = os.path.abspath(path)
        print(f'abs_path: {abs_path}')
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = dft_parser.get_elastic_tensor(abs_path)
        print(f'dft data: {self.dft_data}')
        self.has_dft_data = True

    def start_validation(self, overwrite=True, local=False):
        if self.lammps_results == []:
            self.lammps_manager.set_potential_file(self.potential_file, self.simulation)
            self.lammps_manager.set_model_file(self.lammps_model_file)
            self.lammps_results = self.simulation.run(overwrite=overwrite, local=local)
        print(f'lammps results: {self.lammps_results}')
        print(f'has dft data: {self.dft_data}')
        if self.lammps_results and ((self.use_dft and self.dft_data) or self.has_dft_data):
            if self.has_dft_data:
                dft_results = self.dft_data
            percent_error = {}

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

            if isinstance(self.lammps_results, list):
                self.lammps_results = self.lammps_results[0]

            lammps_elastic_tensor = self.lammps_results["elastic_tensor"]
            print(f'lammps elastic tensor: {lammps_elastic_tensor}')
            
            for key in dft_elastic_tensor_dict:
                lammps_value = lammps_elastic_tensor.get(key, None)
                dft_value = dft_elastic_tensor_dict[key]
                if lammps_value is not None and dft_value is not None:
                    percent_error[key] = self.calculate_percent_error(dft_value, lammps_value)

            percent_error["bulk_modulus_voigt"] = self.calculate_percent_error(
                dft_results["bulk_modulus_voigt"], self.lammps_results["bulk_modulus_voigt"]
            )
            percent_error["bulk_modulus_reuss"] = self.calculate_percent_error(
                dft_results["bulk_modulus_reuss"], self.lammps_results["bulk_modulus_reuss"]
            )
            percent_error["bulk_modulus_hill"] = self.calculate_percent_error(
                dft_results["bulk_modulus_hill"], self.lammps_results["bulk_modulus_hill"]
            )
            percent_error["shear_modulus_voigt"] = self.calculate_percent_error(
                dft_results["shear_modulus_voigt"], self.lammps_results["shear_modulus_voigt"]
            )
            percent_error["shear_modulus_reuss"] = self.calculate_percent_error(
                dft_results["shear_modulus_reuss"], self.lammps_results["shear_modulus_reuss"]
            )
            percent_error["shear_modulus_hill"] = self.calculate_percent_error(
                dft_results["shear_modulus_hill"], self.lammps_results["shear_modulus_hill"]
            )
            return percent_error
        else:
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
                "shear_modulus_voigt": None,
                "shear_modulus_reuss": None,
                "shear_modulus_hill": None,
            }


    