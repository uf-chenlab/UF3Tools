import os 
import subprocess
import re
import shutil

from UF3Tools.simulations.common.LammpsInputFileHandler import LammpsInputFileModifier, LammpsInputFileHandler
class LammpsSimulation:
    @classmethod
    def get_id(cls):
        return cls.simulation_type
    @classmethod
    def get_result_type(cls):
        return cls.result_type
    def __init__(self, 
                 input_file,
                 potential_file,
                 model_file,
                 parameters=None):
        self.parameters = parameters or {}
        self.set_potential_file(potential_file)
        self.set_model_file(model_file)
        self.result = {}
    def get_input_file_path(self):
        return self.input_file
    def set_model_file(self, model_file):
        self.parameters['model_file'] = model_file
    def set_potential_file(self, potential_file):
        self.parameters['uf3_file'] = potential_file
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
    def get_parameters(self):
        return self.parameters

class CompleteLammpsSimulation(LammpsSimulation):
    @classmethod
    def get_id(cls):
        return cls.simulation_type
    @classmethod
    def get_result_type(cls):
        return cls.result_type
    def __init__(self, simulation_name, input_file, model_file, potential_file, 
                 simulation_root_directory=os.getcwd(), run_script="start_simulation.sh", 
                 extra_files=[], simulation_type=None, parameters=None):
        """
        Initializes the simulation environment and copies necessary files.
        """
        self.simulation_directory = os.path.join(simulation_root_directory, simulation_name)
        if not self.simulation_directory.endswith("Simulation"):
            self.simulation_directory += "Simulation"
        self.simulation_directory = os.path.abspath(self.simulation_directory)
        os.makedirs(self.simulation_directory, exist_ok=True)

        self.copy_simulation_files(input_file, model_file, potential_file, run_script, extra_files)
        self.input_file = os.path.join(self.simulation_directory, os.path.basename(input_file))
        self.model_file = os.path.join(self.simulation_directory, os.path.basename(model_file))
        self.potential_file = os.path.join(self.simulation_directory, os.path.basename(potential_file))
        self.run_script = os.path.join(self.simulation_directory, os.path.basename(run_script))
        self.extra_files = [os.path.join(self.simulation_directory, os.path.basename(file)) for file in extra_files]
        super().__init__(self.input_file, self.potential_file, self.model_file, parameters)
        self.setup_input_file()
        if parameters is not None:
            print(f' simulation params: {self.get_parameters()}')
            self.set_input_file_variables()
    def set_input_file_variables(self):
        handler = LammpsInputFileHandler(self.input_file, self.input_file, self.parameters)
        handler.generate_input_file()
    def copy_simulation_files(self, input_file, model_file, potential_file, run_script, extra_files):
        shutil.copy(input_file, self.simulation_directory)
        shutil.copy(model_file, self.simulation_directory)
        shutil.copy(potential_file, self.simulation_directory)
        shutil.copy(run_script, self.simulation_directory)
        for file in extra_files:
            shutil.copy(file, self.simulation_directory)     
    def setup_input_file(self):
        handler = LammpsInputFileModifier(self.input_file)
        modify_input = {
            'read_data {parameter}': [self.model_file],
            'pair_coeff * * {parameter}': [self.potential_file],
        }
        handler.modify_input_file(modify_input)
        
    def run(self, local=False, overwrite=False):
        command = f"chmod 777 {self.run_script} && {self.run_script}"
        print("running simulation with command: ", command)
        print("simulation directory: ", self.simulation_directory)
        subprocess.run(command, shell=True, cwd=self.simulation_directory)
        return self.process_simulation_results()
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
   
class BaseSimulation(CompleteLammpsSimulation):
    def __init__(self, simulation_name, input_file, model_file, potential_file, 
                 simulation_root_directory=os.getcwd(), run_script="start_simulation.sh", 
                 extra_files=[], simulation_type=None, parameters=None):
        
        super().__init__(simulation_name, input_file, model_file, potential_file, simulation_root_directory, run_script, extra_files, simulation_type, parameters=parameters)

class BaseSingleCrystalSimulation(BaseSimulation):
    def __init__(self, simulation_name, input_file, model_file, potential_file, 
                 simulation_root_directory=os.getcwd(), run_script="start_simulation.sh", 
                 extra_files=[], simulation_type=None, parameters=None):
        super().__init__(simulation_name, input_file, model_file, potential_file, simulation_root_directory, run_script, extra_files, simulation_type, parameters=parameters)


class CompoundSimulation(CompleteLammpsSimulation):
    required_sub_simulations = []
    @classmethod
    def get_required_sub_simulations(cls):
        return cls.required_sub_simulations
    def __init__(self, simulation_name, input_file, model_file, potential_file, 
                 simulation_root_directory=os.getcwd(), run_script="start_simulation.sh", 
                 extra_files=[], simulation_type=None, sub_simulations=None, parameters=None):
        
        super().__init__(simulation_name, input_file, model_file, potential_file, simulation_root_directory, run_script, extra_files, simulation_type, parameters=parameters)
        self.sub_simulations = sub_simulations or {}
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

    
        