import os 
import subprocess
import re

class LammpsSimulation:
    @classmethod
    def get_id(cls):
        return cls.simulation_type
    @classmethod
    def get_result_type(cls):
        return cls.result_type
    def __init__(self, 
                 simulation_directory, 
                 input_file,
                 parameters=None):
        self.simulation_directory = os.path.abspath(simulation_directory)
        self.input_file = os.path.abspath(input_file)
        self.parameters = parameters or {}
        self.result = {}
    def run(self, local=False, overwrite=False):
        # Parallel safe because it's only changing the cwd of the subprocess, not the process globally
        if local:
            command = "chmod 777 startSimulation.sh && ./startSimulation.sh"
        else:
            command = "chmod 777 start_simulation_hpg.sh && ./start_simulation_hpg.sh"

        subprocess.run(command, shell=True, cwd=self.simulation_directory)
        return self.process_simulation_results()
    def get_input_file_path(self):
        return self.input_file
    def set_potential_file(self, potential_file):
        print(f"setting potential file: {potential_file}, {os.getcwd()}")
        self.parameters['uf3_file'] = potential_file
        print(f'parameters total: {self.parameters}')
    def get_directory(self):
        return self.simulation_directory
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
    def read_result_file(self, result_file, pattern):
        results = {}
        try:
            with open(result_file, 'r') as f:
                lines = f.read()
            match = re.search(pattern, lines)
            return match
        except:
            print(f"Error: Could not open file {result_file}")
            return None
    def get_parameters(self):
        return self.parameters
    
class BaseSimulation(LammpsSimulation):
    def __init__(self, 
                 simulation_directory,
                 input_file,
                 parameters=None):
        
        super().__init__(simulation_directory, input_file, parameters)
    def process_simulation_results(self):
        raise NotImplementedError("Subclasses must implement this method")
    def get_result(self):
        if(self.check_existing_results()):
            return self.result
        else:
            print("No base simulation results found. simulation not run")
        return self.result    
    
class CompoundSimulation(LammpsSimulation):
    required_sub_simulations = []
    @classmethod
    def get_required_sub_simulations(cls):
        return cls.required_sub_simulations
    def __init__(self, 
                 simulation_directory, 
                 input_file,
                 sub_simulations = None,
                 parameters=None):
        
        super().__init__(simulation_directory, input_file, parameters)
        self.sub_simulations = sub_simulations or {}
    def process_simulation_results(self):
        raise NotImplementedError("Subclasses must implement this method")
    def get_sub_simulation(self, simulation_type):
        return self.sub_simulations.get(simulation_type, None)
    def get_sub_simulations(self):
        return self.sub_simulations
    def set_sub_simulation(self, simulation_type, simulation):
        self.sub_simulations[simulation_type] = simulation
    def set_sub_simulations(self, simulations):
        self.sub_simulations = simulations
    def run(self, local=False, overwrite=False):
        if(self.sub_simulations != {}):
            if(self.result != {}):
                return self.result
            else:
                return self.process_simulation_results()
        else:
            print(f"MISSING SUB SIMULATION RESULT")
            return None

    
        