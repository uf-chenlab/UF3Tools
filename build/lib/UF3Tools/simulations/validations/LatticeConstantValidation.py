from .PotentialValidation import *
from ..simulations.LatticeConstantSimulation import LatticeConstantSimulation
class LatticeConstantValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return LatticeConstantSimulation.get_required_sub_simulations()
    computational_weight=1
    required_supercell = (1,1,1)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, run_scripts, template_directory, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "LatticeConstantSimulation",
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
            "pair_style": "uf3",
            "uf3_elements": element_list
        }
        self.kwargs = self.default_kwargs
        if parameters is not None:
            self.kwargs.update(parameters)
        self.dft_data = {"lattice_constant": None}
        self.lammps_data = {"lattice_constant": None}
        self.use_dft = False

        self.sim_template_file = os.path.join(
            template_directory,
            "in.lattice"
        )
        self.input_file = "lammps.in"
        self.simulation = LatticeConstantSimulation(
            self.sim_template_file,
            self.lammps_model_file,
            self.potential_file,
            os.path.join(validation_folder_path, 'lammps', 'simulations'),
            run_scripts['lammps'],
            parameters=parameters 
        )
        self.setup_input_file(os.path.join(self.simulation.get_directory(), self.sim_template_file), os.path.join(self.simulation.get_directory(), self.input_file), self.simulation.get_parameters())
        self.required_base_simulations = self.simulation.get_required_sub_simulations()

    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        abs_path = os.path.abspath(path)
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = dft_parser.get_lattice_constants(abs_path)
        self.has_dft_data = True