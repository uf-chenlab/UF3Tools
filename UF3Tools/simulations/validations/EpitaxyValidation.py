from .PotentialValidation import *
from ..simulations.EpitaxySimulation import EpitaxySimulation
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.vasp.inputs import Poscar
import re

class EpitaxyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return None
    computational_weight=4
    required_supercell = (8,8,6)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, run_scripts, template_directory, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "EpitaxySimulation",
            potential_file,
            lammps_model_file,
            dft_model_file,
            element_list,
            abs_validation_path,
            run_scripts,
            use_dft=False
        )
        self.default_kwargs = {
            "min_style": "cg",
            "uf3_elements": element_list
        }
        self.kwargs = self.default_kwargs
        if parameters is not None:
            self.kwargs.update(parameters)
        self.dft_data = {}
        self.lammps_data = {}
        self.use_dft = False

        self.sim_template_file = os.path.join(
            template_directory,
            "in.defect_energy"
        )
        self.input_file = "lammps.in"
        self.simulation = EpitaxySimulation(
            self.sim_template_file,
            self.lammps_model_file,
            self.potential_file,
            os.path.join(validation_folder_path, 'lammps', 'simulations'),
            run_scripts['lammps'],
            parameters=parameters
            
        )
        self.setup_input_file(os.path.join(self.simulation.get_directory(), self.sim_template_file), os.path.join(self.simulation.get_directory(), self.input_file), self.simulation.get_parameters())

    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        self.dft_data = {}
        self.has_dft_data = True

    def start_validation(self, overwrite=True, local=False):
        self.lammps_data = self.simulation.run(overwrite=overwrite, local=local)