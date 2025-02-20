from .PotentialValidation import *
from ..simulations.LatticeConstantSimulation import LatticeConstantSimulation
class LatticeConstantValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return LatticeConstantSimulation.get_required_sub_simulations()
    computational_weight=1
    required_supercell = (1,1,1)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, **kwargs):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "LatticeConstantSimulation",
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
        self.dft_data = {"lattice_constant": None}
        self.lammps_data = {"lattice_constant": None}
        self.use_dft = False

        self.lammps_manager.setup_simulation(self.simulation_type)
        self.simulation = LatticeConstantSimulation(
            self.lammps_manager.get_root_directory(),
            self.lammps_manager.get_template_directory(),
            None,
            **kwargs
        )
        self.required_base_simulations = self.simulation.get_required_sub_simulations()
        self.initialize_simulation_files()

    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        abs_path = os.path.abspath(path)
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = dft_parser.get_lattice_constants(abs_path)
        self.has_dft_data = True