from .PotentialValidation import *
from ..simulations.FormationEnergySimulation import FormationEnergySimulation
class FormationEnergyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return FormationEnergySimulation.get_required_sub_simulations()
    @classmethod
    def get_required_single_crystal_simulations(cls):
        return FormationEnergySimulation.get_required_sub_simulations()
    computational_weight=1
    required_supercell = (1,1,1)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, run_scripts, template_directory, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "FormationEnergySimulation",
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
        self.dft_data = {"formation_energy": None}
        self.lammps_data = {"formation_energy": None}
        self.single_crystal_data = None
        self.use_dft = False

        self.sim_template_file = os.path.join(
            template_directory,
            "in.lattice"
        )
        self.input_file = "lammps.in"
        self.simulation = FormationEnergySimulation(
            self.sim_template_file,
            self.lammps_model_file,
            self.potential_file,
            os.path.join(validation_folder_path, 'lammps', 'simulations'),
            run_scripts['lammps'],
            parameters=parameters 
        )
        self.setup_input_file(os.path.join(self.simulation.get_directory(), self.sim_template_file), os.path.join(self.simulation.get_directory(), self.input_file), self.simulation.get_parameters())
        self.required_base_simulations = self.simulation.get_required_sub_simulations()

    def set_single_crystal_data(self, single_crystal_data):
        self.single_crystal_data = single_crystal_data
        print("single crystal data is: ", self.single_crystal_data)
    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        abs_path = os.path.abspath(path)
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = {}
        self.has_dft_data = True
        
    def start_validation(self, overwrite=True, local=False):
        return {'formation_energy': 100}