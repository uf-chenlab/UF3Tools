from .PotentialValidation import *

class SurfaceEnergyValidation(PotentialValidation):
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, **kwargs):
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
        self.lammps_manager.setup_simulation("surface_energy", **self.kwargs)
        os.chdir(self.original_directory)
        self.lammps_manager.set_potential_file(self.potential_file)
        self.lammps_manager.set_model_file(self.lammps_model_file)
        
        #DFT SETUP
        self.dft_manager.setup_simulation("surface_energy", ncore=2)
        self.dft_manager.set_model_file(self.dft_model_file)
        self.dft_manager.perform_model_modification()
        
        os.chdir(self.root_directory)

    def start_validation(self, overwrite=True, local=False):
        """
        Start the validation process by running the LAMMPS and DFT simulations.
        """
        lammps_results = self.lammps_manager.start_simulation(overwrite=overwrite, local=local)
        percent_error = {}
        if isinstance(lammps_results, list):
            lammps_results = lammps_results[0]
        area = (lammps_results['bulk_area'] + lammps_results['free_area'])/2
        surface_energy = (lammps_results['free_energy'] - lammps_results['bulk_energy'])/(2*area)
        lammps_results.pop('bulk_area')
        lammps_results.pop('free_area')
        lammps_results['area'] = area
        lammps_results['surface_energy'] = surface_energy  
        dft_results = self.dft_result
        if isinstance(dft_results, list):
            dft_results = dft_results[0]
        dft_results['area'] = lammps_results['area']
        

        if self.use_dft and self.dft_result != {}:
            dft_results = self.dft_manager.start_simulation(overwrite=overwrite, local=local)

        dft_results['surface_energy'] = (dft_results['free'] - dft_results['bulk'])/(2*area)
        # Compare the results
        print(f"LAMMPS results: {lammps_results}")
        print(f"DFT results: {dft_results}")
        
        

        # Handle the case where results might be lists of dictionaries
        
        try:
            percent_error['surface_energy'] = abs(lammps_results['surface_energy'] - dft_results['surface_energy']) / dft_results['surface_energy'] * 100
        except (ZeroDivisionError) as e:
            percent_error['surface_energy'] = None
            print(f"Error calculating percent error for 'surface_energy': {e}")
       
        return percent_error
   
    
