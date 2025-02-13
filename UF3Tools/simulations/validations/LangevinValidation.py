from .PotentialValidation import *
class LangevinValidation(PotentialValidation):
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list,**kwargs):
        self.default_kwargs = {
            'min_style': 'cg',
            'uf3_elements': element_list
        }
        # Update the default keyword arguments with the user-specified arguments
        self.kwargs = self.default_kwargs
        self.kwargs.update(kwargs)
        super().__init__(potential_file, lammps_model_file, dft_model_file, element_list)
    def initialize_simulation_files(self):
        """
        Initialize the simulation files needed for the potential file.
        """
        # LAMMPS SETUP
        self.lammps_manager.setup_simulation("langevin")
        os.chdir(self.original_directory)
        self.lammps_manager.set_potential_file(self.potential_file, self.simulation)
        self.lammps_manager.set_model_file(self.lammps_model_file)
  