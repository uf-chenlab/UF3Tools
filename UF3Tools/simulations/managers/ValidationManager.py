import os 
from ..validations import CohesiveEnergyValidation, ElasticTensorValidation, SurfaceEnergyValidation
from ..validations import PotentialValidation, LatticeConstantValidation, LangevinValidation
from .LammpsSimulationManager import LammpsSimulationManager

class ValidationManager:
    validation_lut = {
            "cohesive_energy": CohesiveEnergyValidation,
            "elastic_tensor": ElasticTensorValidation,
            "surface_energy": SurfaceEnergyValidation,
            "lattice_constant": LatticeConstantValidation,
            "langevin": LangevinValidation
    }
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, validations_to_run, **kwargs):
        self.potential_file = potential_file
        self.lammps_model_file = lammps_model_file
        self.dft_model_file = dft_model_file
        self.element_list = element_list
        self.validation_folder_path = validation_folder_path
        self.validations_to_run = validations_to_run
        self.kwargs = kwargs
        self.base_simulations_required = []
        self.base_results = []

    def run_validation(self, overwrite=True, local=False):
        """
        Run the validation process.
        """
        
        # determine the base simulations needed
        for validation in self.validations_to_run:
            req_list = self.validation_lut[validation].get_required_base_simulations()
            if(req_list != []):
                self.base_simulations.append(req_list)
            
        # set up and run our base simulations
        #combine the base simulations into a single list
        base_simulations = []
        for sublist in self.base_simulations_required:
            base_simulations.extend(sublist)
        #remove duplicates
        base_simulations = list(set(base_simulations))
        #create the base simulations
        for base_simulation in base_simulations:
            manager = LammpsSimulationManager()
            manager.setup_simulation(base_simulation)
            manager.set_potential_file(self.potential_file)
            manager.set_model_file(self.lammps_model_file)
            result = manager.start_simulation(overwrite=overwrite, local=local)
            self.base_results.append(result)
        #check that the base results are valid
        for result in self.base_results:
            if(result == []):
                raise ValueError("Base simulation results are empty. Please check your potential file and model file.")
        
        #sort the validations to run by computational weight
        self.validations_to_run.sort(key=lambda x: self.validation_lut[x].get_computational_weight())
        for validation in self.validations_to_run:
            validation_class = self.validation_lut[validation](
                self.potential_file,
                self.lammps_model_file,
                self.dft_model_file,
                self.element_list,
                self.validation_folder_path,
                **self.kwargs
            )
            validation_class.start_validation(overwrite=overwrite, local=local)  
            