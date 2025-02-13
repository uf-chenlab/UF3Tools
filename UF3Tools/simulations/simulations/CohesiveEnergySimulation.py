import os
from .Simulation import CompoundSimulation
from .BaseRelaxationSimulation import *


class CohesiveEnergySimulation(CompoundSimulation):
    required_sub_simulations = [BaseRelaxationSimulation]
    simulation_type = "CohesiveEnergy"
    result_type = "cohesive_energy"
    def __init__(self, simulation_directory, template_directory, sub_simulations, **kwargs):
        self.original_directory = os.getcwd()
        self.template_file_name = 'in.cohesive'
        self.simulation_directory = os.path.join(simulation_directory, (self.simulation_type+'Simulation'))
        self.output_file = os.path.join(simulation_directory, "lammps.in")
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
        self.parameters.update(kwargs)
        super().__init__(self.simulation_directory, os.path.join(template_directory, (self.simulation_type + 'Simulation')), self.template_file_name, "lammps.in", self.parameters, sub_simulations, **kwargs)
    def process_simulation_results(self):
        if(self.sub_simulations == {}):
            return {}
        if('BaseRelaxation' in self.sub_simulations.keys()):
            sim = self.sub_simulations['BaseRelaxation']
            self.result = {
                self.result_type: sim.get_result()[self.result_type]
            }
        return self.result
        