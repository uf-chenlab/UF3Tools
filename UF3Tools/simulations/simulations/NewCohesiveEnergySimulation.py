import os
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser
from .Simulation import CompoundSimulation
from .BaseRelaxationSimulation import *


class NewCohesiveEnergySimulation(CompoundSimulation):
    def __init__(self, simulation_directory, template_directory, sub_simulations, **kwargs):
        self.required_sub_simulations = [BaseRelaxationSimulation]
        self.template_file = template_directory + "in.cohesive"
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
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
        super().__init__(simulation_directory, template_directory, "in.cohesive", "lammps.in", self.parameters, sub_simulations, **kwargs)