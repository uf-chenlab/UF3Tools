from .PotentialValidation import *
    
class LatticeConstantValidation(PotentialValidation):
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, **kwargs):
        
        print("lattice constant constructor")
        
        print(f'intiailizing files:, {potential_file}, {lammps_model_file}, {dft_model_file}, {element_list}, {validation_folder_path}')
        
        self.default_kwargs = {
            'min_style': 'cg',
            'uf3_elements': element_list
        }
        # Update the default keyword arguments with the user-specified arguments
        self.kwargs = self.default_kwargs
        self.kwargs.update(kwargs)
        
        super().__init__(potential_file, lammps_model_file, dft_model_file, element_list,  validation_folder_path, use_dft=True)
    def initialize_simulation_files(self):
        """
        Initialize the simulation files needed for the potential file.
        """
        print("lattice constant initialize")
        # LAMMPS SETUP
        self.lammps_manager.setup_simulation("relaxation", **self.kwargs)
        print("lammps manager setup")
        os.chdir(self.original_directory)
        print("original directory: ", self.original_directory)
        print(f"potential and model files: {self.potential_file}, {self.lammps_model_file}")
        self.lammps_manager.set_potential_file(self.potential_file)
        self.lammps_manager.set_model_file(self.lammps_model_file)
        
        #DFT SETUP
        self.dft_manager.setup_simulation("relaxation", ncore=2)
        self.dft_manager.set_model_file(self.dft_model_file)
        os.chdir(self.root_directory)