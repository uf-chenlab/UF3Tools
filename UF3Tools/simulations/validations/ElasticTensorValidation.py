from .PotentialValidation import *
from ..simulations.ElasticTensorSimulation import ElasticTensorSimulation
import os
import shutil

class ElasticTensorValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return None
    computational_weight=2

    @classmethod
    def get_required_supercell(cls, params=None):
        if params:
            abs_path = os.path.abspath(params)
        if os.path.exists(os.path.join(abs_path, "2D_material")):
            print("[ElasticTensorValidation] Detected 2D material — using (2,2,1)")
            cls.required_supercell = (2, 2, 1)
        else:
            cls.required_supercell = (2, 2, 2)       
        return cls.required_supercell              
    


    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, run_scripts, template_directory, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        self.validation_folder_path = abs_validation_path

        super().__init__(
            "ElasticTensorSimulation",
            potential_file,
            lammps_model_file,
            dft_model_file,
            element_list,
            abs_validation_path,
            run_scripts,
            use_dft=False
        )

        print(f'parameters: {parameters}')
        self.default_kwargs = {
            "min_style": "cg",
            "pair_style": "uf3",
            "uf3_elements": element_list
        }
        self.kwargs = self.default_kwargs
        print(f'Kwargs: {self.kwargs}')
        if parameters is not None:
            self.kwargs.update(parameters)
        print(f'Kwargs: {self.kwargs}')
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
        self.lammps_model_file = os.path.abspath(lammps_model_file)
        self.use_dft = False

        self.sim_template_file = os.path.join(template_directory, "in.elastic")
        self.input_file = "lammps.in"
        print("self lammps model file: ", self.lammps_model_file)

        self.simulation = ElasticTensorSimulation(
            self.sim_template_file,
            self.lammps_model_file,
            self.potential_file,
            os.path.join(validation_folder_path, 'lammps', 'simulations'),
            run_scripts['lammps'],
            parameters=parameters
        )

        self.setup_input_file(
            os.path.join(self.simulation.get_directory(), self.sim_template_file),
            os.path.join(self.simulation.get_directory(), self.input_file),
            self.simulation.get_parameters()
        )
        shutil.copy(
            os.path.join(self.simulation.get_directory(), "in.elastic"),
            os.path.join(self.simulation.get_directory(), "lammps.in")
        )

    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        print(f'path: {path}')
        abs_path = os.path.abspath(path)
        print(f'abs_path: {abs_path}')
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = dft_parser.get_elastic_tensor(abs_path)
        print(f'dft data: {self.dft_data}')
        self.has_dft_data = True

    
    def get_monolayer_thickness_from_atoms(self):
        """Calculate the physical thickness of a 2D monolayer from atomic z-coordinates in the LAMMPS model file."""
        z_coords = []
        reading_atoms = False
        with open(self.lammps_model_file, 'r') as f:
            for line in f:
                if "Atoms" in line:
                    reading_atoms = True
                    next(f)  # skip the blank line after 'Atoms'
                    continue
                if reading_atoms:
                    if line.strip() == "":
                        break  # end of atom section
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Assumes format: atom-ID atom-type x y z [optional extra columns]
                        z = float(parts[4])  # z-coordinate is 5th column
                        z_coords.append(z)
        
        if not z_coords:
            raise RuntimeError("Could not extract z-coordinates from the LAMMPS model file.")

        min_z = min(z_coords)
        max_z = max(z_coords)
        thickness_angstrom = max_z - min_z
        thickness_nm = thickness_angstrom * 0.1  # convert from Å to nm
        print(f"[Monolayer thickness] min z = {min_z:.4f}, max z = {max_z:.4f}, thickness = {thickness_nm:.4f} nm")
        return thickness_nm


    def convert_lammps_elastic_tensor_to_2d(self, tensor_3d, thickness_nm):
        """Convert all LAMMPS elastic constants from GPa to N/m using box thickness."""
        tensor_2d = {}
        for key, value in tensor_3d.items():
            if value is not None:
                tensor_2d[key] = value * thickness_nm  # GPa * nm = N/m
            else:
                tensor_2d[key] = None
        return tensor_2d

    


    def extract_independent_elastic_constants(self, elastic_tensor, type="hex"):
        """
        Determines the crystal structure type based on the symmetry of the elastic tensor 
        and extracts the independent elastic constants accordingly.

        Parameters:
            elastic_tensor (dict): Dictionary containing elastic tensor components.

        Returns:
            tuple: (structure_type, dict of independent elastic constants)
        """
        C11 = elastic_tensor.get("C11all", None)
        C12 = elastic_tensor.get("C12all", None)
        C13 = elastic_tensor.get("C13all", None)
        C22 = elastic_tensor.get("C22all", None)
        C23 = elastic_tensor.get("C23all", None)
        C33 = elastic_tensor.get("C33all", None)
        C44 = elastic_tensor.get("C44all", None)
        C55 = elastic_tensor.get("C55all", None)
        C66 = elastic_tensor.get("C66all", None)


        # # **2D Material**: 3 independent constants (C11, C12, C66)  ## all i want to do is is say that if C11, C12, and C66 are present and only present (other values are none or 0) then it is a 2D material
        # if C11 and C12 and C66 and all(x is not None for x in [C11, C12, C66]):
        #     return "2D Material", {
        #         "C11": C11,
        #         "C12": C12,
        #         "C66": C66,
        #     }
        
         # **2D Material**: Only C11, C12, and C66 are defined
        if all(x is not None and abs(x) > 1e-8 for x in [C11, C12, C66]) and all(
            x is None or abs(x) < 1e-8
            for x in [C13, C22, C23, C33, C44, C55]
        ):
            return "2D Material", {
                "C11": C11,
                "C12": C12,
                "C66": C66,
            }

        
        # **Cubic**: 3 independent constants
        # Conditions: C11 = C22 = C33, C12 = C13 = C23, C44 = C55 = C66
        if (C11 == C22 == C33) and (C12 == C13 == C23) and (C44 == C55 == C66):
            return "cubic", {
                "C11": C11,
                "C12": C12,
                "C44": C44,
            }

 
        # **Hexagonal/Wurtzite**: 5 independent constants
        # Conditions: C11 = C22, C13 = C23, C44 = C55, and C66 = (C11 - C12) / 2
        elif type=="hex" or ((C11 == C22) and (C13 == C23) and (C44 == C55) and (C66 == (C11 - C12) / 2)):
            return "hexagonal/wurtzite", {
                "C11": C11,
                "C12": C12,
                "C13": C13,
                "C33": C33,
                "C44": C44,
            }

        # **Tetragonal**: 6 independent constants
        # Conditions: C11 = C22, C13 = C23, C44 = C55 (but C66 is independent)
        elif (C11 == C22) and (C13 == C23) and (C44 == C55):
            return "tetragonal", {
                "C11": C11,
                "C12": C12,
                "C13": C13,
                "C33": C33,
                "C44": C44,
                "C66": C66,
            }

        # **Orthorhombic**: 9 independent constants
        # No strict symmetry between components; all are independent.
        elif C11 and C22 and C33 and C12 and C13 and C23 and C44 and C55 and C66:
            return "orthorhombic", {
                "C11": C11,
                "C12": C12,
                "C13": C13,
                #"C22": C22,
                #"C23": C23,
                "C33": C33,
                "C44": C44,
                #"C55": C55,
                "C66": C66,
            }

        # **Monoclinic**: 13 independent constants
        # Usually, C16, C26, and C36 are nonzero, making it more complex.
        elif "C16" in elastic_tensor or "C26" in elastic_tensor or "C36" in elastic_tensor:
            return "monoclinic", {
                key: elastic_tensor[key]
                for key in elastic_tensor
                if key in ["C11", "C12", "C13", "C15", "C22", "C23", "C25", "C33", "C35", "C44", "C46", "C55", "C66"]
            }

        # **Triclinic**: 21 independent constants (fully asymmetric case)
        # If no symmetry is detected, assume a fully anisotropic triclinic structure.
        return "triclinic", elastic_tensor


    def start_validation(self, overwrite=True, local=False):
        self.lammps_results = self.simulation.run(overwrite=overwrite, local=local)
        print(f'lammps results: {self.lammps_results}')
        print(f'has dft data: {self.dft_data}')
        if self.lammps_results and ((self.use_dft and self.dft_data) or self.has_dft_data):
            if self.has_dft_data:
                dft_results = self.dft_data
            percent_error = {}
            dft_elastic_tensor_dict = {}
            for i in range(6):
                for j in range(6):
                    key = f"C{i+1}{j+1}all"
                    dft_elastic_tensor_dict[key] = dft_results["elastic_tensor"][i][j]
            #make sure no results are none
            print(f'dft elastic tensor dict: {dft_elastic_tensor_dict}')
            for key in dft_elastic_tensor_dict:
                if dft_elastic_tensor_dict[key] is None:
                    dft_elastic_tensor_dict[key] = 0
            if isinstance(self.lammps_results, list):
                self.lammps_results = self.lammps_results[0]

            lammps_elastic_tensor = self.lammps_results["elastic_tensor"]

            # Convert LAMMPS elastic tensor from GPa to N/m for 2D materials
            # # This is necessary because LAMMPS outputs elastic constants in GPa,
            # # while we need them in N/m for 2D materials.
            # Convert to 2D units if needed
# Convert to 2D units if needed
            if os.path.exists(os.path.join(self.validation_folder_path, "2D_material")):
                print("Converting LAMMPS elastic constants from GPa to N/m for 2D comparison")
                thickness_nm = self.get_monolayer_thickness_from_atoms()
                print(f"Using monolayer thickness: {thickness_nm:.4f} nm")
                self.lammps_results["elastic_tensor"] = self.convert_lammps_elastic_tensor_to_2d(
                    self.lammps_results["elastic_tensor"], thickness_nm
                )
                for key in ["bulk_modulus_voigt", "bulk_modulus_reuss", "bulk_modulus_hill",
                            "shear_modulus_voigt", "shear_modulus_reuss", "shear_modulus_hill"]:
                    if self.lammps_results[key] is not None:
                        self.lammps_results[key] *= thickness_nm

            # Now assign converted tensor
            lammps_elastic_tensor = self.lammps_results["elastic_tensor"]


            
            print(f'lammps elastic tensor: {lammps_elastic_tensor}')
            
            for key in dft_elastic_tensor_dict:
                lammps_value = lammps_elastic_tensor.get(key, None)
                dft_value = dft_elastic_tensor_dict[key]
                if (lammps_value is not None and dft_value is not None) and (lammps_value != 0 and dft_value != 0):
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
            structure, indep_percent_error = self.extract_independent_elastic_constants(percent_error)
            indep_error = 0
            for key in indep_percent_error:
                indep_error += indep_percent_error[key]
            indep_error /= len(indep_percent_error)
            percent_error['independent_elastic_constants'] = indep_error
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
                'independent_elastic_constants': None
            }


    