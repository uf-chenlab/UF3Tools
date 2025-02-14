import os 
import subprocess
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser
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
                 template_directory, 
                 template_file, 
                 output_file,
                 parameters=None,
                 **kwargs):
        self.template_file = os.path.join(template_directory, template_file)
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        print(f"LAMMPSSimulation directory: {self.simulation_directory}")
        self.output_file = os.path.join(simulation_directory, output_file)
        self.parameters = parameters or {}
        self.parameters.update(kwargs)
        self.result = {}
        self.setup_input_file()
    def run(self, local=False, overwrite=False):
        print(f"running simulation: {self.simulation_directory}")
        os.chdir(self.simulation_directory)
        if(local == True):
            command = "chmod 777 startSimulation.sh && ./startSimulation.sh"
        else:
            #command = "chmod 777 start_simulation_hpg.sh && ./start_simulation_hpg.sh"
            command = "export OMP_NUM_THREADS=8 && mpirun -np 1 ./lmp_mpi_hpg < lammps.in"
        #subprocess.run(command, shell=True,stdout=subprocess.DEVNULL)
        subprocess.run(command, shell=True)
        result = self.process_simulation_results()
        os.chdir(self.original_directory)
        return result  
    def set_potential_file(self, potential_file):
        print(f"setting potential file: {potential_file}, {os.getcwd()}")
        self.parameters['uf3_file'] = potential_file
        self.setup_input_file()
    
    def setup_input_file(self):
        os.chdir(self.simulation_directory)
        handler = LammpsInputFileHandler(self.template_file, self.output_file)
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
        os.chdir(self.original_directory)
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
    required_sub_simulations = []
    @classmethod
    def get_required_sub_simulations(cls):
        return cls.required_sub_simulations
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

    
        