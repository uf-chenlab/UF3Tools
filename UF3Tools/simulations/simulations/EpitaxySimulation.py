import os
from .Simulation import CompleteLammpsSimulation


class EpitaxySimulation(CompleteLammpsSimulation):
    simulation_type = "Epitaxy"
    result_type = "epitaxy"
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], sub_simulations=None, parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
        self.parameters = {
            'atom_style': 'atomic',
            "pair_style": "uf3",
            'boundary': 'p p f',
            'min_z_height': '0',
            'max_z_height': '80',
            'uf3_degree': '3',
            'timestep': '0.0005',
            'substrate_surface_min': '30',
            'substrate_surface_max': '50',
            'num_atoms_to_generate': '5',
            'growth_temp': '2000',
            'initial_eq_time': '20000',
            'rain_time': '20000',
            'eq_time': '20000'
        }
        if parameters is not None:
            self.parameters.update(parameters)
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, sub_simulations, parameters=self.parameters)
    def get_result(self):
        if(self.result != {}):
            return self.result
        else:
            return self.process_simulation_results()