import os

from UF3Tools.simulations.common.LammpsInputFileHandler import LammpsInputFileHandler 
from ..validations.DefectEnergyValidation import DefectEnergyValidation
from ..validations.CohesiveEnergyValidation import CohesiveEnergyValidation
from ..validations.ElasticTensorValidation import ElasticTensorValidation
from ..validations.LangevinValidation import LangevinValidation
from ..validations.LatticeConstantValidation import LatticeConstantValidation
from ..validations.SurfaceEnergyValidation import SurfaceEnergyValidation
from ..validations.PhonopyValidation import PhonopyValidation
from ..validations.PotentialValidation import ValidationData
from ..validations.FormationEnergyValidation import FormationEnergyValidation
from ..validations.EnergyPerAtomValidation import EnergyPerAtomValidation
from ..simulations.Simulation import BaseSingleCrystalSimulation, BaseSimulation
from ...common.LAMMPSModifier import LAMMPSModifier
class ValidationManager:
    validation_lut = {
        "cohesive_energy": CohesiveEnergyValidation,
        "elastic_tensor": ElasticTensorValidation,
        "surface_energy": SurfaceEnergyValidation,
        "lattice_constant": LatticeConstantValidation,
        "langevin": LangevinValidation,
        "phonopy": PhonopyValidation,
        "formation_energy": FormationEnergyValidation,
        'defect_energy': DefectEnergyValidation,
        'energy_per_atom': EnergyPerAtomValidation,
    }
    
    template_file_lut = {
        'LatticeConstant': "in.lattice",
        'CohesiveEnergy': "in.cohesive",
        'ElasticTensor': "in.elastic",
        'SurfaceEnergy': "in.surface",
        'Langevin': "in.langevin",
        'BaseRelaxation': "in.base_relax",
        'BaseSingleCrystalRelaxation': "in.base_relax",
        'Phonopy': "in.phonopy",
        'FormationEnergy': "in.formation_energy",
        'DefectEnergy': "in.defect_energy",
    }
    
    dft_results_lut = {
        'lattice_constant': "dft/simulations/RelaxationSimulation/CONTCAR",
        'cohesive_energy': "dft/simulations/CohesiveEnergySimulation/results.txt",
        'elastic_tensor': "dft/simulations/ElasticTensorSimulation/results.txt",
        'surface_energy': "dft/simulations/SurfaceEnergySimulation/",
        'phonopy': "dft/simulations/PhonopySimulation/",
        'formation_energy': "dft/simulations/FormationEnergySimulation/",
        'defect_energy': "dft/simulations/DefectEnergySimulation/",
        'energy_per_atom': "dft/simulations/EnergyPerAtomSimulation/results.txt"
    }

    def __init__(self, potential_version, structure_id, potential_file, lammps_model_file, dft_model_file, element_list, config_directory, validation_folder_path, dft_validation_folder = None, validations_to_run=None, kwargs=None):
        self.potential_file = os.path.abspath(potential_file)
        self.config_directory = os.path.abspath(config_directory)
        self.lammps_model_file = os.path.abspath(lammps_model_file)
        self.dft_model_file = os.path.abspath(dft_model_file)
        print(f'validation folder path: {validation_folder_path}')
        print(f'dft validation folder path: {dft_validation_folder}')
        self.validation_folder_path = os.path.abspath(validation_folder_path)
        self.dft_validation_folder = os.path.abspath(dft_validation_folder) if dft_validation_folder else self.validation_folder_path
        self.structure_id = structure_id
        self.element_list = element_list
        self.validations_to_run = validations_to_run
        self.kwargs = kwargs
        if self.kwargs is None:
            self.kwargs = {}
        self.kwargs.update({'uf3_elements': self.element_list})
        self.base_simulations_required = []
        self.base_results = {}
        self.base_simulation_objects = {}
        self.single_crystal_data = None
        self.lammps_directory = os.path.join(self.validation_folder_path, "lammps")
        self.lammps_simulation_root_directory = os.path.join(self.lammps_directory, "simulations")
        self.template_directory = os.path.join(self.config_directory, "lammps", "templates")
        self.basic_files_directory = os.path.join(self.config_directory, "lammps", "basic_files")
        self.run_scripts = {
            'lammps': os.path.join(self.basic_files_directory, "start_simulation_hpg.sh"),
            'dft': os.path.join(self.basic_files_directory, "start_simulation_dft.sh")
        }
        self.validation_data = ValidationData(potential_version, potential_file, structure_id)
        print(f' validation data element list: {self.element_list}')

    def get_validation_data(self):
        return self.validation_data
    def get_base_results(self):
        return self.base_results
    def setup_input_file(self, template_file, output_file, parameters=None):
        print('setup input file params: ', parameters)
        handler = LammpsInputFileHandler(template_file, output_file, parameters)
        handler.generate_input_file()
    def set_single_crystal_data(self, data):
        self.single_crystal_data = data
    def run_validation(self, overwrite=True, local=False):
        modifier = LAMMPSModifier(self.lammps_model_file)
        elements = modifier.get_elements()
        self.kwargs.update({'uf3_elements': elements})

        for validation in self.validations_to_run:
            if validation not in self.validation_lut:
                continue
            sim = self.validation_lut[validation]
            if issubclass(sim, BaseSimulation):
                continue
            required_simulations = sim.get_required_base_simulations()
            if required_simulations:
                self.base_simulations_required.extend(required_simulations)

        base_simulations = list(set(self.base_simulations_required))
        base_simulations = [sim for sim in base_simulations if not issubclass(sim, BaseSingleCrystalSimulation)]

        for base_simulation in base_simulations:
            try:
                sim_template_file = os.path.join(
                    self.template_directory,
                    self.template_file_lut[base_simulation.get_id()]
                )
                input_file = "lammps.in"
                base_simulation_instance = base_simulation(
                    sim_template_file,
                    self.lammps_model_file,
                    self.potential_file,
                    self.lammps_simulation_root_directory,
                    self.run_scripts['lammps'],
                    parameters=self.kwargs
                )
                self.setup_input_file(
                    os.path.join(base_simulation_instance.get_directory(), sim_template_file),
                    os.path.join(base_simulation_instance.get_directory(), input_file),
                    base_simulation_instance.get_parameters()
                )

                print(f'potential file: {self.potential_file}')
                print(f'base simulation locations: {base_simulation_instance.get_directory()}')
                print(f'lammps model file: {self.lammps_model_file}')
                print(f'base simulation params: {base_simulation_instance.get_parameters()}')

                result = base_simulation_instance.run(overwrite=overwrite, local=local)

                self.base_results[base_simulation.get_id()] = result
                self.base_simulation_objects[base_simulation.get_id()] = base_simulation_instance
            except Exception as e:
                print(f"Base simulation {base_simulation.__name__} failed: {e}")
                self.base_results[base_simulation.get_id()] = None

        for sim_id, result in self.base_results.items():
            if result is None:
                raise RuntimeError(f"Base simulation {sim_id} failed. Cannot proceed with validations.")

        self.validations_to_run.sort(key=lambda x: self.validation_lut[x].get_computational_weight())
        print("base simulations are complete")

        for validation in self.validations_to_run:
            try:
                print(f'Running validation: {validation}')
                modifier = LAMMPSModifier(self.lammps_model_file)
                lammps_file = modifier.generate_supercell(self.validation_lut[validation].get_required_supercell())

                validation_class = self.validation_lut[validation](
                    self.potential_file,
                    lammps_file,
                    self.dft_model_file,
                    self.element_list,
                    self.validation_folder_path,
                    self.run_scripts,
                    self.template_directory,
                    parameters=self.kwargs
                )

                required_base_simulations = validation_class.get_required_base_simulations()
                if required_base_simulations:
                    for base_simulation in required_base_simulations:
                        if issubclass(base_simulation, BaseSingleCrystalSimulation):
                            validation_class.set_single_crystal_data(self.single_crystal_data)
                        else:
                            validation_class.get_simulation().set_sub_simulation(
                                base_simulation.get_id(),
                                self.base_simulation_objects.get(base_simulation.get_id())
                            )

                print("dft validation folder: ", self.dft_validation_folder)
                dft_result_path = os.path.join(self.dft_validation_folder, self.dft_results_lut[validation])
                print(f"DFT result path: {dft_result_path}")
                validation_class.retrieve_dft_result(dft_result_path)
                print(f"DFT result: retrieved")

                result = validation_class.start_validation(overwrite=overwrite, local=local)
                print(f"Validation result: {result}")
                lammps_raw = validation_class.get_lammps_validation_results()
                dft_raw = validation_class.get_dft_validation_results()

                print(f"Validation: {validation} - Percent Error: {result}")

                self.validation_data.add_validation(validation, result)
                self.validation_data.add_raw_data(validation, lammps_raw, dft_raw)
            except Exception as e:
                print(f"Validation {validation} failed: {e}")

        print("validation manager finished")
        return self.validation_data

            
class SingleCrystalValidationManager(ValidationManager):
    def __init__(self, potential_version, structure_id, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, validations_to_run, kwargs=None):
        super().__init__(potential_version, structure_id, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, validations_to_run, kwargs)
        # Determine the base simulations needed
        print(f'validations to run: {validations_to_run}')
        for validation in validations_to_run:
            required_simulations = self.validation_lut[validation].get_required_single_crystal_simulations()
            print(f'required simulations: {required_simulations}')
            if required_simulations:
                self.base_simulations_required.extend(required_simulations)

        # Remove duplicates
        self.base_simulations_required = list(set(self.base_simulations_required))
        print(f'base simulations: {self.base_simulations_required}')
        self.validations_to_run = self.base_simulations_required
        
    
    def run_validation(self, overwrite=True, local=False):
        # Remove duplicates
        base_simulations = list(set(self.base_simulations_required))
        #find the element from the lammps model file
        modifier = LAMMPSModifier(self.lammps_model_file)
        elements = modifier.get_elements()
        self.kwargs.update({'uf3_elements': elements})
      
        # Run base simulations
        for base_simulation in base_simulations:
            sim_template_file = os.path.join(
                self.template_directory,
                self.template_file_lut[base_simulation.get_id()]
            )
            input_file = "lammps.in"
            base_simulation_instance = base_simulation(
                sim_template_file,
                self.lammps_model_file,
                self.potential_file,
                self.lammps_simulation_root_directory,
                self.run_scripts['lammps'],
                parameters=self.kwargs
            )
            self.setup_input_file(os.path.join(base_simulation_instance.get_directory(), sim_template_file), os.path.join(base_simulation_instance.get_directory(), input_file), base_simulation_instance.get_parameters())
           
            print(f'potential file: {self.potential_file}')
            print(f'base simulation locations: {base_simulation_instance.get_directory()}')
            print(f'lammps model file: {self.lammps_model_file}')
            print(f' base simulation params: {base_simulation_instance.get_parameters()}')
           

            # Run simulation
            result = base_simulation_instance.run(overwrite=overwrite, local=local)

            # Store results
            self.base_results[self.structure_id] = result
            self.base_simulation_objects[base_simulation.get_id()] = base_simulation_instance

        # Check if base simulation results are valid
        for result in self.base_results.values():
            if not result:
                raise ValueError("Base simulation results are empty. Please check your potential file and model file.")

        self.validation_data = self.get_base_results()
        print("SINGLE CRYSTAL VALIDATION DATA: ", self.validation_data)
        return self.validation_data
