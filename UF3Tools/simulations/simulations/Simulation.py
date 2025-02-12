import os 
import shutil
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser

class LammpsSimulation:
    def __init__(self, 
                 simulation_directory, 
                 template_directory, 
                 template_file, 
                 output_file,
                 parameters=None,
                 **kwargs):
        self.template_file = os.path.join(template_directory, template_file)
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, output_file)
        self.parameters = parameters or {}
        self.parameters.update(kwargs)
        self.result = {}
        
    def set_potential_file(self, potential_file):
        self.parameters['uf3_file'] = potential_file
        self.setup_input_file()
    
    def setup_input_file(self):
        handler = LammpsInputFileHandler(self.template_file, self.output_file)
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
    
    def check_existing_results(self):
        if self.result != {}:
            return True
    def get_result(self):
        if(self.result != {}):
            return self.result
        else:
            print("No simulation results!")
            return {}

    
    def process_simulation_results(self):
        raise NotImplementedError("Subclasses must implement this method")
    
class BaseSimulation(LammpsSimulation):
    def __init__(self, 
                 simulation_directory, 
                 template_directory, 
                 template_file, 
                 output_file,
                 parameters=None,
                 **kwargs):
        
        super().__init__(simulation_directory, template_directory, template_file, output_file, parameters, **kwargs)
    def process_simulation_results(self):
        raise NotImplementedError("Subclasses must implement this method")
    def get_result(self):
        if(self.check_existing_results()):
            return self.result
        else:
            print("No base simulation results found. simulation not run")
        return self.result
    
    
class CompoundSimulation(LammpsSimulation):
    def __init__(self, 
                 simulation_directory, 
                 template_directory, 
                 template_file, 
                 output_file,
                 parameters=None,
                 sub_simulations=None,
                 **kwargs):
        
        super().__init__(simulation_directory, template_directory, template_file, output_file, parameters, **kwargs)
        self.sub_simulations = sub_simulations or {}
        self.required_sub_simulations = []
    def process_simulation_results(self):
        raise NotImplementedError("Subclasses must implement this method")
    def get_required_sub_simulations(self):
        return self.required_sub_simulations
    def get_sub_simulation(self, simulation_type):
        return self.sub_simulations.get(simulation_type, None)
    def get_sub_simulations(self):
        return self.sub_simulations
    def set_sub_simulation(self, simulation_type, simulation):
        self.sub_simulations[simulation_type] = simulation
    def set_sub_simulations(self, simulations):
        self.sub_simulations = simulations

    
        