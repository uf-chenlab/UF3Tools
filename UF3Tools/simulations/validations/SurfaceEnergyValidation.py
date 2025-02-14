from .PotentialValidation import *
from ..simulations.SurfaceEnergySimulation import SurfaceEnergySimulation

class SurfaceEnergyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return None
    computational_weight=1
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path,**kwargs):
        self.simulation_type = "SurfaceEnergySimulation"
        super().__init__(self.simulation_type, potential_file, lammps_model_file, dft_model_file, element_list,  validation_folder_path,use_dft=False)
        self.default_kwargs = {
            'min_style': 'cg',
            'uf3_elements': element_list
        }
        # Update the default keyword arguments with the user-specified arguments
        self.kwargs = self.default_kwargs
        self.kwargs.update(kwargs)
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
        print("validation folder path in elastictensor: ", self.root_directory)
        print("my root directory is: ", self.lammps_manager.get_root_directory())
        print("my template directory is: ", self.lammps_manager.get_template_directory())
        self.lammps_manager.setup_simulation(self.simulation_type)
        self.simulation = SurfaceEnergySimulation(self.lammps_manager.get_root_directory(), self.lammps_manager.get_template_directory(), **kwargs)
        self.initialize_simulation_files()
    def retrieve_dft_result(self, path):
        """
        Retrieve DFT results from the specified path.
        """
        print(f"looking for dft results in: {path}")
        dft_parser = DFTDataParser(path)
        self.dft_data = dft_parser.get_surface_energy(path)
        print(f"surface energy: {self.dft_data}")
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
            print(f"lammps results: {self.lammps_results}")
        percent_error = {}
        # Proceed if both LAMMPS and DFT data are available (either newly run or from stored data)
        if self.lammps_results and ((self.use_dft and self.dft_data) or self.has_dft_data == True):
            if self.has_dft_data:
                dft_results = self.dft_data
            print("dft results: ", dft_results)
            dft_bulk = dft_results["bulk_energy"]
            dft_slab = dft_results["slab_energy"]
            dft_area = dft_results["area"]
            # Initialize the dictionary for percent errors
            percent_error = {}
            print("lammps result: ", self.lammps_results)
            # Compare the LAMMPS and DFT elastic tensors by calculating percent errors
            if(isinstance(self.lammps_results, list)):
                self.lammps_results = self.lammps_results[0]
            lammps_bulk = self.lammps_results["bulk_energy"]
            lammps_slab = self.lammps_results["slab_energy"]
            lammps_area = self.lammps_results["bulk_area"]
            
            lammps_surface_energy = (lammps_slab - lammps_bulk) / (2*lammps_area)
            dft_surface_energy = (dft_slab - dft_bulk) / (2*dft_area)
            percent_error['bulk_energy'] = self.calculate_percent_error(dft_bulk, lammps_bulk)
            percent_error['slab_energy'] = self.calculate_percent_error(dft_slab, lammps_slab)
            percent_error['surface_energy'] = self.calculate_percent_error(dft_surface_energy, lammps_surface_energy)

        return percent_error
    
