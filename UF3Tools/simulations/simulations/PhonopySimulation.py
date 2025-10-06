import os
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser
from .Simulation import CompoundSimulation
from .BaseRelaxationSimulation import *

class PhonopySimulation(CompoundSimulation):
    required_sub_simulations = [BaseRelaxationSimulation]
    simulation_type = "Phonopy"
    result_type = "phonopy"
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
        self.parameters = {
            'atom_style': 'atomic',
            'uf3_degree': '3',
            "pair_style": "uf3",
            'masses':''
        }
        if parameters is not None:
            self.parameters.update(parameters)
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, parameters=self.parameters)
    def process_simulation_results(self):
        if(self.sub_simulations == {}):
            return {}
        if('BaseRelaxation' in self.sub_simulations.keys()):
            sim = self.sub_simulations['BaseRelaxation']
            self.result = sim.get_result()
        return self.result