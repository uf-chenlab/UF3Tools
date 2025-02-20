from .PotentialValidation import *
from ..simulations.SurfaceEnergySimulation import SurfaceEnergySimulation

class SurfaceEnergyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return None
    computational_weight=1
    required_supercell = (1,1,1)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "SurfaceEnergySimulation",
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
        if parameters is not None:
            self.kwargs.update(parameters)
        self.dft_data = {
            "bulk_energy": None,
            "slab_energy": None,
            "surface_energy": None,
        }
        self.lammps_data = {
            "bulk_energy": None,
            "slab_energy": None,
            "surface_energy": None,
        }
        self.use_dft = False

        self.lammps_manager.setup_simulation(self.simulation_type)
        self.sim_template_file = os.path.join(
            self.lammps_manager.get_template_directory(),
            "in.surface"
        )
        self.input_file = "lammps.in"
        self.setup_input_file(self.sim_template_file, os.path.join(self.lammps_manager.get_root_directory(), self.input_file), self.kwargs)
        self.simulation = SurfaceEnergySimulation(
            self.lammps_manager.get_root_directory(),
            self.input_file,
            self.kwargs
        )
        self.initialize_simulation_files()

    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        abs_path = os.path.abspath(path)
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = dft_parser.get_surface_energy(abs_path)
        self.has_dft_data = True

    def start_validation(self, overwrite=True, local=False):
        if self.lammps_results == []:
            self.lammps_manager.set_potential_file(self.potential_file, self.simulation)
            self.lammps_manager.set_model_file(self.lammps_model_file)
            self.lammps_results = self.simulation.run(overwrite=overwrite, local=local)

        percent_error = {}
        if self.lammps_results and ((self.use_dft and self.dft_data) or self.has_dft_data):
            if self.has_dft_data:
                dft_results = self.dft_data
            dft_bulk = dft_results["bulk_energy"]
            dft_slab = dft_results["slab_energy"]
            dft_area = dft_results["area"]

            if isinstance(self.lammps_results, list):
                self.lammps_results = self.lammps_results[0]

            lammps_bulk = self.lammps_results["bulk_energy"]
            lammps_slab = self.lammps_results["slab_energy"]
            lammps_area = self.lammps_results["bulk_area"]

            lammps_surface_energy = (lammps_slab - lammps_bulk) / (2 * lammps_area)
            dft_surface_energy = (dft_slab - dft_bulk) / (2 * dft_area)

            percent_error["bulk_energy"] = self.calculate_percent_error(
                (dft_bulk / dft_area), (lammps_bulk / lammps_area)
            )
            percent_error["slab_energy"] = self.calculate_percent_error(
                (dft_slab / dft_area), (lammps_slab / lammps_area)
            )
            percent_error["surface_energy"] = self.calculate_percent_error(
                dft_surface_energy, lammps_surface_energy
            )

        return percent_error
    
