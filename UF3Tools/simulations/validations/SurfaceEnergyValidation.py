from .PotentialValidation import *

class SurfaceEnergyValidation(PotentialValidation):
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, **kwargs):
        self.default_kwargs = {
            'min_style': 'cg',
            'uf3_elements': element_list
        }
        # Update the default keyword arguments with the user-specified arguments
        self.kwargs = self.default_kwargs
        self.kwargs.update(kwargs)
        super().__init__(potential_file, lammps_model_file, dft_model_file, element_list,validation_folder_path)

    def initialize_simulation_files(self):
        """
        Initialize the simulation files needed for the potential file.
        """
        # LAMMPS SETUP
        self.lammps_manager.setup_simulation("surface_energy", **self.kwargs)
        os.chdir(self.original_directory)
        print("looking for potential file in: ", self.original_directory)
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
        print("lammps results raw")
        print(lammps_results)
        surface_energy = (lammps_results['free_energy']/(2*lammps_results['free_area']) - lammps_results['bulk_energy']/(2*lammps_results['bulk_area']))
        lammps_results.pop('bulk_area')
        lammps_results.pop('free_area')
        lammps_results['area'] = area
        lammps_results['surface_energy'] = surface_energy  
        print(lammps_results)
        dft_results = self.dft_result
        if isinstance(dft_results, list):
            dft_results = dft_results[0]
        dft_results['area'] = lammps_results['area']
        

        if self.use_dft and self.dft_result != {}:
            dft_results = self.dft_manager.start_simulation(overwrite=overwrite, local=local)
            dft_results['surface_energy'] = (dft_results['free'] - dft_results['bulk'])/(2*area)
        else:
            
            array = dft_results['surface_energy']
            print("dft array")
            print(array)
            dft_results['bulk'] = array[0]/(2*area)
            dft_results['free'] = array[1]/(2*area)
            dft_results['surface_energy'] = array[1]/(2*area) - array[0]/(2*area)
            print(dft_results['surface_energy'])
        

        try:
            print(f"lammps {lammps_results['bulk_energy']} ; area: {2 * lammps_results['area']} ; dft {dft_results['bulk']}")
            percent_error['bulk_energy'] = abs(
                (lammps_results['bulk_energy'] /(2* lammps_results['area'])) - 
                (dft_results['bulk'])
            ) / abs((dft_results['bulk']) )* 100
        except ZeroDivisionError as e:
            percent_error['bulk_energy'] = None
            print(f"Error calculating percent error for 'bulk_energy': {e}")

        try:
            percent_error['free_energy'] = abs(
                (lammps_results['free_energy'] /( lammps_results['area']*2)) - 
                (dft_results['free'])
            ) / abs((dft_results['free']) )* 100
        except ZeroDivisionError as e:
            percent_error['free_energy'] = None
            print(f"Error calculating percent error for 'free_energy': {e}")

        # Calculate percent error for surface energy
        try:
            percent_error['surface_energy'] = abs(
                lammps_results['surface_energy'] - dft_results['surface_energy']
            ) / abs(dft_results['surface_energy'] )* 100
        except ZeroDivisionError as e:
            percent_error['surface_energy'] = None
            print(f"Error calculating percent error for 'surface_energy': {e}")

        return percent_error
    
