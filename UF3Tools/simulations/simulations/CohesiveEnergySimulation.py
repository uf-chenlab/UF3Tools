import os
from .Simulation import CompoundSimulation
from .BaseRelaxationSimulation import *


class CohesiveEnergySimulation(CompoundSimulation):
    required_sub_simulations = [BaseRelaxationSimulation]
    simulation_type = "CohesiveEnergy"
    result_type = "cohesive_energy"
    def __init__(self, simulation_directory, input_file, sub_simulations=None, parameters=None):
        self.original_directory = os.getcwd()
        if(self.simulation_type not in simulation_directory):
            simulation_directory = os.path.join(simulation_directory, (self.simulation_type+'Simulation'))
        self.simulation_directory = os.path.abspath(simulation_directory)
        self.input_file = os.path.join(simulation_directory, input_file)
        self.parameters = {
            'atom_style': 'atomic',
            'x_boundary': 'p',
            'y_boundary': 'p',
            'z_boundary': 'p',
            'change_box': '',
            'uf3_degree': '3',
            'uf3_file': 'AlN.uf3',
            'uf3_elements': 'Al N',
            'box_relax_settings': 'fix 1 all box/relax iso 0.0 vmax 0.001',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000'
        }
        if parameters is not None:
            self.parameters.update(parameters)
        super().__init__(self.simulation_directory, input_file, sub_simulations, self.parameters)
    def process_simulation_results(self):
        if(self.sub_simulations == {}):
            return {}
        if('BaseRelaxation' in self.sub_simulations.keys()):
            sim = self.sub_simulations['BaseRelaxation']
            self.result = {
                self.result_type: sim.get_result()[self.result_type]
            }
        return self.result
        