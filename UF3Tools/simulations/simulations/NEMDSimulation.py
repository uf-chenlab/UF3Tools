import os 
from .Simulation import CompleteLammpsSimulation
from pymatgen.core import Structure
from UF3Tools.common.LAMMPSModifier import LAMMPSModifier


class NEMDSimulation(CompleteLammpsSimulation):
    simulation_type = "NEMDSimulation"
    result_type = "conductivity"
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
        self.model_file = model_file
        print("model file: ", model_file)
        self.parameters = {
            'atom_style': 'atomic',
            'uf3_degree': '3',
            "pair_style": "uf3",
            'uf3_file': 'AlN.uf3',
            'uf3_elements': 'Al N',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000',
            'timestep': '0.001',
            'source_start': '5',
            'source_end': '10',
            'sink_start': '45',
            'sink_end': '50',
            'b1_end': '2',
            'b2_end': '55',
            'center_start': '20',
            'center_end': '25',
            'env_temp': '300',
            'env_pressure': '1',
            'env_run_steps': '20000',
            'source_flux': '0.0075',
            'sink_flux': '-0.0075',
            'restart_steps': '20000',
            'avg_time_1': '20000',
            'avg_time_2': '50000',
            'direct_method_run_steps': '200000',
        }
        if parameters is not None:
            self.parameters.update(parameters)
            
        self.set_position_regions_from_model()
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, parameters=self.parameters)
    def set_position_regions_from_model(self):
        structure = LAMMPSModifier(self.model_file).get_structure()
        print(structure)
        c_length = structure.lattice.c
        print('structure c length: ', c_length)
        # just set total bins to 100, assume region positions are in % of total length
        num_bins = 100
        bin_size = c_length / num_bins

        def region_bin(start_frac, end_frac, bin_size):
            start = int(start_frac * num_bins*bin_size)
            end = int(end_frac * num_bins*bin_size)
            return str(start), str(end)

        self.parameters['sink_start'], self.parameters['sink_end'] = region_bin(0.10, 0.15, bin_size)
        self.parameters['source_start'], self.parameters['source_end'] = region_bin(0.85, 0.90,bin_size)
        self.parameters['b1_end'] = str(int(0.02 * num_bins*bin_size))
        self.parameters['b2_end'] = str(int(0.98 * num_bins*bin_size))
        self.parameters['center_start'], self.parameters['center_end'] = region_bin(0.45, 0.55,bin_size)
        print(self.parameters)
    
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        init_lattice_exists = os.path.exists(os.path.join(self.simulation_directory, "initial_lattice.txt"))
        final_lattice_exists = os.path.exists(os.path.join(self.simulation_directory, "final_lattice.txt"))
        force_exists = os.path.exists(os.path.join(self.simulation_directory, "force_dump.txt"))
        return init_lattice_exists and final_lattice_exists and force_exists
       
    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
        

        
