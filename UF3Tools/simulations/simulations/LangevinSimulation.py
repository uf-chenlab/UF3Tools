import os 
from .Simulation import CompleteLammpsSimulation

class LangevinSimulation(CompleteLammpsSimulation):
    simulation_type = "LangevinExpansion"
    result_type = "lattice_expansion"
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
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
            'force_dump_steps': '400000',
            'dump_steps': '200000',
            'temperature': '300',
            'pressure': '1',
            'run_steps': '200000',
            'npt_enable': '#',
        }
        if parameters is not None:
            self.parameters.update(parameters)
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, parameters=self.parameters)
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
        

        
