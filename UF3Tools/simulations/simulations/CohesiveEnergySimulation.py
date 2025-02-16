import os
from .Simulation import CompoundSimulation
from .BaseRelaxationSimulation import *


class CohesiveEnergySimulation(CompoundSimulation):
    required_sub_simulations = [BaseRelaxationSimulation]
    simulation_type = "CohesiveEnergy"
    result_type = "cohesive_energy"
    def __init__(self, simulation_directory, template_directory, sub_simulations, **kwargs):
        abs_sim_dir = os.path.abspath(simulation_directory)
        abs_template_dir = os.path.abspath(template_directory)

        final_sim_path = os.path.join(abs_sim_dir, self.simulation_type + "Simulation")
        self.simulation_directory = final_sim_path
        self.template_file_name = 'in.cohesive'
        self.output_file = os.path.join(final_sim_path, "lammps.in")

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

        super().__init__(
            final_sim_path,
            abs_template_dir,
            self.template_file_name,
            "lammps.in",
            self.parameters,
            sub_simulations,
            **kwargs
        )
    def process_simulation_results(self):
        if(self.sub_simulations == {}):
            return {}
        if('BaseRelaxation' in self.sub_simulations.keys()):
            sim = self.sub_simulations['BaseRelaxation']
            self.result = {
                self.result_type: sim.get_result()[self.result_type]
            }
        return self.result
        