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
        'elastic_tensor': "dft/simulations/ElasticTensorSimulation/results.txt",
        'surface_energy': "dft/simulations/SurfaceEnergySimulation/"
    }

    def __init__(self, potential_version, structure_id, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, validations_to_run, **kwargs):
        self.potential_file = os.path.abspath(potential_file)
        self.lammps_model_file = os.path.abspath(lammps_model_file)
        self.dft_model_file = os.path.abspath(dft_model_file)
        self.validation_folder_path = os.path.abspath(validation_folder_path)
        self.element_list = element_list
        self.validations_to_run = validations_to_run
        self.kwargs = kwargs
        self.base_simulations_required = []
        self.base_results = {}
        self.base_simulation_objects = {}
        self.validation_data = ValidationData(potential_version, potential_file, structure_id)

    def get_validation_data(self):
        return self.validation_data

    def run_validation(self, overwrite=True, local=False):
        """
        Run the validation process without using os.chdir().
        """

        # Determine the base simulations needed
        for validation in self.validations_to_run:
            required_simulations = self.validation_lut[validation].get_required_base_simulations()
            if required_simulations:
                self.base_simulations_required.extend(required_simulations)

        # Remove duplicates
        base_simulations = list(set(self.base_simulations_required))

        # Run base simulations
        for base_simulation in base_simulations:
            manager = LammpsSimulationFileManager(self.validation_folder_path)
            manager.setup_simulation(base_simulation.get_id())

            base_simulation_instance = base_simulation(
                manager.get_simulation_directory(),
                manager.get_template_directory(),
                **self.kwargs
            )
            print(f'potential file: {self.potential_file}')
            print(f'base simulation locations: {manager.get_simulation_directory()}')
            print(f'lammps model file: {self.lammps_model_file}')
            manager.set_potential_file(self.potential_file, base_simulation_instance)
            manager.set_model_file(self.lammps_model_file)

            # Run simulation
            result = base_simulation_instance.run(overwrite=overwrite, local=local)

            # Store results
            self.base_results[base_simulation.get_id()] = result
            self.base_simulation_objects[base_simulation.get_id()] = base_simulation_instance

        # Check if base simulation results are valid
        for result in self.base_results.values():
            if not result:
                raise ValueError("Base simulation results are empty. Please check your potential file and model file.")

        # Sort validations by computational weight
        self.validations_to_run.sort(key=lambda x: self.validation_lut[x].get_computational_weight())

        # Run validations
        for validation in self.validations_to_run:
            print(f'Running validation: {validation}')

            # Generate LAMMPS supercell
            modifier = LAMMPSModifier(self.lammps_model_file)
            lammps_file = modifier.generate_supercell(self.validation_lut[validation].get_required_supercell())

            # Create validation class instance
            validation_class = self.validation_lut[validation](
                self.potential_file,
                lammps_file,
                self.dft_model_file,
                self.element_list,
                self.validation_folder_path,
                **self.kwargs
            )

            # Assign required base simulations
            required_base_simulations = validation_class.get_required_base_simulations()
            if required_base_simulations:
                for base_simulation in required_base_simulations:
                    validation_class.get_simulation().set_sub_simulation(
                        base_simulation.get_id(),
                        self.base_simulation_objects[base_simulation.get_id()]
                    )

            # Retrieve DFT results
            dft_result_path = os.path.join(self.validation_folder_path, self.dft_results_lut[validation])
            validation_class.retrieve_dft_result(dft_result_path)

            # Run validation
            result = validation_class.start_validation(overwrite=overwrite, local=local)
            lammps_raw = validation_class.get_lammps_validation_results()

            print(f"Validation: {validation} - Percent Error: {result}")

            # Store results
            self.validation_data.add_validation(validation, result)
            self.validation_data.add_lammps_validation_result(validation, lammps_raw)

        return self.validation_data
            