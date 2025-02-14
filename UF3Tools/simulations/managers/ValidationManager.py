import os 
from ..validations.CohesiveEnergyValidation import CohesiveEnergyValidation
from ..validations.ElasticTensorValidation import ElasticTensorValidation
from ..validations.LangevinValidation import LangevinValidation
from ..validations.LatticeConstantValidation import LatticeConstantValidation
from ..validations.SurfaceEnergyValidation import SurfaceEnergyValidation
from ..validations.PotentialValidation import ValidationData
from .LammpsSimulationFileManager import LammpsSimulationFileManager
from ...common.LAMMPSModifier import LAMMPSModifier
class ValidationManager:
    validation_lut = {
            "cohesive_energy": CohesiveEnergyValidation,
            "elastic_tensor": ElasticTensorValidation,
            "surface_energy": SurfaceEnergyValidation,
            "lattice_constant": LatticeConstantValidation,
            "langevin": LangevinValidation
    }
    dft_results_lut = {
        'lattice_constant': "dft/simulations/RelaxationSimulation/CONTCAR",
        'cohesive_energy': "dft/simulations/CohesiveEnergySimulation/results.txt",
        'elastic_tensor': "dft/simulations/ElasticTensor/results.txt",
        'surface_energy': "dft/simulations/SurfaceEnergySimulation/"
    }
    def __init__(self, potential_version, structure_id, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, validations_to_run, **kwargs):
        self.potential_file = potential_file
        self.lammps_model_file = lammps_model_file
        self.dft_model_file = dft_model_file
        self.element_list = element_list
        self.validation_folder_path = validation_folder_path
        self.validations_to_run = validations_to_run
        self.kwargs = kwargs
        self.base_simulations_required = []
        self.base_results = {}
        self.base_simulation_objects = {}
        self.validation_data = ValidationData(potential_version, potential_file, structure_id)
        self.current_directory = os.getcwd()
    def get_validation_data(self):
        return self.validation_data
    def run_validation(self, overwrite=True, local=False):
        """
        Run the validation process.
        """
        os.chdir(self.validation_folder_path)
        # determine the base simulations needed
        for validation in self.validations_to_run:
            req_list = self.validation_lut[validation].get_required_base_simulations()
            if(req_list != [] and req_list != None):
                self.base_simulations_required.append(req_list)
            
        # set up and run our base simulations
        #combine the base simulations into a single list
        base_simulations = []
        for sublist in self.base_simulations_required:
            base_simulations.extend(sublist)
        #remove duplicates
        base_simulations = list(set(base_simulations))
        #create the base simulations
        for base_simulation in base_simulations:      
            manager = LammpsSimulationFileManager()
            manager.setup_simulation(base_simulation.get_id())
            base_simulation = base_simulation(manager.get_simulation_directory(), manager.get_template_directory(), **self.kwargs)
            os.chdir(self.current_directory)
            manager.set_potential_file(self.potential_file, base_simulation)
            manager.set_model_file(self.lammps_model_file)
            os.chdir(self.validation_folder_path)
            result = base_simulation.run(overwrite=overwrite, local=local)
            self.base_results[base_simulation.get_id()] = result
            self.base_simulation_objects[base_simulation.get_id()] = base_simulation
        #check that the base results are valid
        for result in self.base_results.values():
            if(result == []):
                raise ValueError("Base simulation results are empty. Please check your potential file and model file.")
        
        #sort the validations to run by computational weight
        self.validations_to_run.sort(key=lambda x: self.validation_lut[x].get_computational_weight())
        for validation in self.validations_to_run:
            print(f'running validation: {validation}')
            os.chdir(self.current_directory)
            if self.validation_lut[validation].requires_supercell == True:
                modifier = LAMMPSModifier(self.lammps_model_file)
                lammps_file = modifier.generate_supercell((8,8,8))
            else:
                lammps_file = self.lammps_model_file
            validation_class = self.validation_lut[validation](
                self.potential_file,
                lammps_file,
                self.dft_model_file,
                self.element_list,
                self.validation_folder_path,
                **self.kwargs
            )
            required_base_simulations = validation_class.get_required_base_simulations()
            if(required_base_simulations != [] and required_base_simulations != None):
                for base_simulation in required_base_simulations:
                    validation_class.get_simulation().set_sub_simulation(base_simulation.get_id(), self.base_simulation_objects[base_simulation.get_id()])
                    
            validation_class.retrieve_dft_result(os.path.join(self.validation_folder_path, self.dft_results_lut[validation]))   
            os.chdir(self.validation_folder_path)
            result = validation_class.start_validation(overwrite=overwrite, local=local)  
            print(f"Validation: {validation} - Percent Error: {result}")
            self.validation_data.add_validation(validation, result)
        os.chdir(self.current_directory)
        return self.validation_data
            