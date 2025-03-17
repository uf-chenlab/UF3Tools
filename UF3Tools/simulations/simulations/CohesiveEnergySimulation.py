import os
from .Simulation import CompoundSimulation
from .BaseRelaxationSimulation import *


class CohesiveEnergySimulation(CompoundSimulation):
    required_sub_simulations = [BaseRelaxationSimulation]
    simulation_type = "CohesiveEnergy"
    result_type = "cohesive_energy"
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], sub_simulations=None, parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
        self.parameters = {
            'atom_style': 'atomic',
            'x_boundary': 'p',
            'y_boundary': 'p',
            'z_boundary': 'p',
            'change_box': '',
            'uf3_degree': '3',
            'box_relax_settings': 'fix 1 all box/relax iso 0.0 vmax 0.001',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000'
        }
        if parameters is not None:
            self.parameters.update(parameters)
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, sub_simulations, parameters=self.parameters)
    def process_simulation_results(self):
        if(self.sub_simulations == {}):
            return {}
        if('BaseRelaxation' in self.sub_simulations.keys()):
            sim = self.sub_simulations['BaseRelaxation']
            self.result = {
                self.result_type: sim.get_result()[self.result_type]
            }
        return self.result
        