from UF3Tools.uf3_handler import *
from UF3Tools.simulations import *
from UF3Tools.automation.ModelAutomater import *
import os
import shutil
from tqdm import tqdm
import math
from dask.distributed import Client
import optuna
from dask_jobqueue import SLURMCluster
from dask.distributed import get_worker, WorkerPlugin
from UF3Tools.common.MPAPIKey import MP_API_KEY
import uuid
import socket
import shutil
from datetime import datetime
import concurrent.futures
import time
import optuna
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import dask
from dask.distributed import WorkerPlugin

class AutomaterPlugin(WorkerPlugin):
    def __init__(self, automater_init_args, settings):
        self.automater_init_args = automater_init_args
        self.settings = settings

    def setup(self, worker):
        worker_hostname = socket.gethostname()
        worker_pid = os.getpid()
        self.worker_id = f"{worker_hostname}-{worker_pid}"

        trial_folder = os.path.join(self.settings['base_directory'], "optuna_trials", f"worker_{self.worker_id}")
        os.makedirs(trial_folder, exist_ok=True)
        # If we need to copy a "validation_base" folder or something:
        validation_src = os.path.join(self.settings['base_directory'], "validation_base")
        if os.path.exists(validation_src):
            validation_dst = os.path.join(trial_folder, "validation", "structures")
            if not os.path.exists(validation_dst):
                shutil.copytree(validation_src, validation_dst, dirs_exist_ok=True)

        automater = ModelAutomater(**self.automater_init_args)
        automater.set_ignore_list(self.settings['ignore_list'])
        automater.set_trainer(self.settings['trainer'])
        automater.load_existing_base_data()
        automater.load_existing_features()
        automater.load_filtered_data()
        automater.consolidate_data()

        if self.settings['testing_sets'] is not None:
            automater.select_testing_set(
                self.settings['testing_sets'],
                include_in_training=False,
                test_split_fraction=self.settings['train_test_split'],
                seed=42
            )
        else:
            automater.split_data(
                self.settings['train_test_split'],
                self.settings['unique_indices'],
                self.settings['unique_indices'],
                self.settings['excluded_testing_sets']
            )
        automater.set_dft_validation_structures(self.settings['testing_mp_structures'])
        automater.set_lammps_validation_structures(self.settings['testing_mp_structures'])
        os.chdir(trial_folder)
        automater.download_validation_structures()
        if 'formation_energy' in self.settings['validations_to_run']:
            if 'formation_energy_structures' not in self.settings:
                raise ValueError("formation_energy_structures not provided in settings")
            automater.download_structures(self.settings['formation_energy_structures'])
            automater.set_single_crystal_validation_structures(self.settings['formation_energy_structures'])

        automater.load_fitting_data(
            automater.get_training_set(),
            automater.get_testing_set(),
            batch_size=self.settings['batch_size'],
            num_cores=self.settings['cores_per_job']
        )

        # Store in worker attributes
        worker.plugins['uf3_automater'] = automater
        worker.plugins['uf3_worker_id'] = self.worker_id
def distributed_train_model_and_validate(train_model_fn, validate_model_fn, params, settings, trial_id=0):
    worker = get_worker()
    automater = worker.plugins['uf3_automater']

    if automater is None:
        raise RuntimeError("Automater not loaded on worker.")

    train_result = train_model_fn(params, settings, trial_id)

    # fail-safe handling
    if not isinstance(train_result, tuple) or len(train_result) != 5:
        print("Train model failed or returned unexpected output:", train_result)
        return None

    chem_sys, trainer_model, bspline_config, model_filename, potential_label = train_result

    validation_data, normalized_error = validate_model_fn(
       settings, chem_sys, trainer_model, bspline_config, model_filename, potential_label, trial_id
    )
    print("Validation data returned:", validation_data)
    print("Normalized error:", normalized_error)

    return validation_data, normalized_error

class UF3AbstractOptunaTrainer():
    def __init__(self, base_directory, automater, settings, job_params):
        self.base_directory = base_directory
        self.automater = automater
        self.settings = {**settings, **job_params}
        self.client = None
        self.cluster = None
    def init_workers(self):
        automater_args = {
            "directories": [],
            "run_data_path": "",
            "feature_filename": "",
            "element_list": self.automater.get_element_list(),
            "degree": self.automater.get_degree(),
            "chemical_system": self.automater.get_chemical_system(),
            "bspline_config": self.automater.get_bspline_config(),
            "num_cores": self.settings['cores_per_job']
        }
        self.settings['ignore_list'] = self.automater.get_ignore_list()
        if self.settings.get('trainer') is None:
            self.settings['trainer'] = self.automater.get_trainer()

        self.cluster = SLURMCluster(
            account=self.settings['account'],
            cores=self.settings['cores_per_job'],
            processes=1,
            memory=self.settings['memory'],
            walltime=self.settings['walltime'],
            job_extra=self.settings['extra'],
            log_directory=self.automater.get_base_directory(),
        )

        self.client = Client(self.cluster, heartbeat_interval=30000, timeout=9999999999)
        self.cluster.scale(self.settings['n_workers'])
        self.client.wait_for_workers(self.settings['n_workers'])

        plugin = AutomaterPlugin(automater_args, self.settings)
        self.client.register_worker_plugin(plugin, name="uf3_plugin")
    def objective(self):
        return None
    def cleanup(self, temp_search=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = "temp_search_results" if temp_search else "grid_search_results"
        backup_path = os.path.join(self.base_directory, folder, timestamp)

        if not self.settings.get('debug', False):
            os.makedirs(backup_path, exist_ok=True)
            trials_path = os.path.join(self.base_directory, "optuna_trials")

            for worker_folder in os.listdir(trials_path):
                src = os.path.join(trials_path, worker_folder)
                for root, dirs, _ in os.walk(src, topdown=True):
                    for d in dirs:
                        if "validation" in d:
                            shutil.rmtree(os.path.join(root, d))
                shutil.move(src, backup_path)

            for f in os.listdir(self.base_directory):
                if "dask-worker" in f:
                    try:
                        os.remove(os.path.join(self.base_directory, f))
                    except Exception:
                        pass

        print("Cleanup done.")
    def optimize(self, sampler):
        study = optuna.create_study(
            sampler=sampler,
            study_name="UF3_Distributed_Optuna",
            direction="minimize",
        )

        study.optimize(
            lambda trial: self.objective(trial, self.client, self.settings),
            n_trials=self.settings.get('n_trials', 2),
            n_jobs=self.settings.get('n_workers', 2),
        )

        print("Best value:", study.best_value)
        print("Best params:", study.best_params)

        return study
    def loss_function(self, avg_result, normalized_error, acceptable_errors, weights):
        print("avg result: ", avg_result)
        # Extract relevant values
        cohesive_energy = avg_result.get('cohesive_energy', 0.0)
        bulk_modulus = avg_result.get('bulk_modulus_voigt', 0.0)
        shear_modulus = (avg_result.get('shear_modulus_voigt', 0.0) + avg_result.get('shear_modulus_reuss', 0.0)) / 2
        lattice_constants = (avg_result.get('a', 0.0) + avg_result.get('b', 0.0) + avg_result.get('c', 0.0)) / 3
        
        indep_elastic = avg_result.get('independent_elastic_constants', 0.0)
        surface_energy = avg_result.get('surface_energy', 0.0)
        phonopy_dos = avg_result.get('total_dos_rmse',0.0)
        phonopy_band = avg_result.get('band_structure_rmse',0.0)
        phonopy_fe = avg_result.get('free_energy', 0.0)
        phonopy_entropy = avg_result.get('entropy', 0.0)
        phonopy_heat_capacity = avg_result.get('heat_capacity', 0.0)
        energy_per_atom = avg_result.get('energy_per_atom', 0.0)
        
        
        # Normalize errors by acceptable thresholds
        norm_cohesive = abs(cohesive_energy) / acceptable_errors.get('cohesive_energy', 1.0)
        norm_bulk = abs(bulk_modulus) / acceptable_errors.get('bulk_modulus', 1.0)
        norm_shear = abs(shear_modulus) / acceptable_errors.get('shear_modulus', 1.0)
        norm_lattice = abs(lattice_constants) / acceptable_errors.get('lattice_constants', 1.0)
        norm_error = abs(normalized_error) / acceptable_errors.get('normalized_error', 1.0)
        norm_surface_energy = abs(surface_energy) / acceptable_errors.get('surface_energy', 1.0)
        norm_indep_elastic = abs(indep_elastic) / acceptable_errors.get('elastic_tensor', 1.0)
        norm_phonopy_dos = abs(phonopy_dos) / acceptable_errors.get('phonopy_dos', 1.0)
        norm_phonopy_band = abs(phonopy_band) / acceptable_errors.get('phonopy_band', 1.0)
        norm_phonopy_fe = abs(phonopy_fe) / acceptable_errors.get('phonopy_free_energy', 1.0)
        norm_phonopy_entropy = abs(phonopy_entropy) / acceptable_errors.get('phonopy_entropy', 1.0)
        norm_phonopy_heat_capacity = abs(phonopy_heat_capacity) / acceptable_errors.get('phonopy_heat_capacity', 1.0)
        norm_energy_per_atom = abs(energy_per_atom) / acceptable_errors.get('energy_per_atom', 1.0)
        
        # Weights (adjustable for importance)
        weight_cohesive = weights.get('cohesive_energy', 0.0)
        weight_elastic = weights.get('elastic', 0.0)
        weight_bulk = weights.get('bulk', 0.0)
        weight_shear = weights.get('shear', 0.0)
        weight_lattice = weights.get('lattice', 0.0)
        weight_error = weights.get('error', 0.0)
        weight_surface_energy = weights.get('surface_energy', 0.0)
        weight_phonopy_dos = weights.get('phonopy_dos', 0.0)
        weight_phonopy_band = weights.get('phonopy_band', 0.0)
        weight_phonopy_fe = weights.get('phonopy_free_energy', 0.0)
        weight_phonopy_entropy = weights.get('phonopy_entropy', 0.0)
        weight_phonopy_heat_capacity = weights.get('phonopy_heat_capacity', 0.0)
        weight_energy_per_atom = weights.get('energy_per_atom', 0.0)
        # Calculate loss with normalized errors
        loss = (
            weight_cohesive * norm_cohesive +
            weight_elastic * norm_indep_elastic +
            weight_bulk * norm_bulk +
            weight_shear * norm_shear +
            weight_lattice * norm_lattice +
            weight_error * norm_error +
            weight_surface_energy * norm_surface_energy +
            weight_phonopy_dos * norm_phonopy_dos +
            weight_phonopy_band * norm_phonopy_band + 
            weight_phonopy_fe * norm_phonopy_fe +
            weight_phonopy_entropy * norm_phonopy_entropy +
            weight_phonopy_heat_capacity * norm_phonopy_heat_capacity + 
            weight_energy_per_atom * norm_energy_per_atom
        )
        return loss
    def parse_validation_data(self, validation_data, normalized_error, acceptable_errors, weights, loss_function):
        all_data = []
        all_results = []
        
        if isinstance(validation_data, list):
            for data in validation_data:
                all_data.append(data)
        else:
            all_data.append(validation_data)
            
        for data in all_data:
            result = {}
            if(isinstance(data, ValidationData)):
                data2 = data.get_validation_data()
                for validation_type, validation_results in data2.items():
                    # Check if validation_results is a list of dictionaries
                    if isinstance(validation_results, list):
                        for res in validation_results:
                            if isinstance(res, dict):
                                for key, value in res.items():
                                    if isinstance(value, (int, float)):  # Check if the value is numeric
                                        result[key] = result.get(key, 0) + value
                            else:
                                print("its not a dictionary bud")  # Handle non-dictionary list items
                    else:
                        # Handle case where validation_results is a single dictionary
                        for key, value in validation_results.items():
                            if isinstance(value, (int, float)):  # Check if the value is numeric
                                result[key] = result.get(key, 0) + value
            else: 
                print("validation data list is broken")
                print(data)
            
            all_results.append(result)
        
        # Compute the average for each metric
        avg_result = {}
        num_results = len(all_results)
        
        if num_results > 0:
            for res in all_results:
                for key, value in res.items():
                    avg_result[key] = avg_result.get(key, 0) + value / num_results
        else:
            print("no validation results")
            return None
        
        final_loss = self.loss_function(avg_result, normalized_error, acceptable_errors, weights)
        print("Final Loss:", final_loss)
        if final_loss is None:
            return None
        if(final_loss <= 0.00001):
            return None
        else:
            return final_loss
    def train_model( params, settings, trial_id=0):
        print("train model function")
        worker = get_worker()
        automater = worker.plugins['uf3_automater']
        worker_id = worker.plugins['uf3_worker_id']
        print("automater loaded")
        if automater is None:
            raise RuntimeError("No automater loaded on this worker! Did you run worker_init_once?")

        (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight) = params
        potential_label = (
            f"trial_{trial_id}_r1b_{ridge_1b}_r2b_{ridge_2b}_r3b_{ridge_3b}_"
            f"curv2b_{curvature_2b}_curv3b_{curvature_3b}_w_{weight}"
        )
        unique_id = str(uuid.uuid4())[:32]
        if settings.get('optimize_sample_weights', False):
            weight_str = "_".join([f"{k}_{v:.2f}" for k, v in settings['sample_weights'].items()])
            potential_label += f"_sw_{weight_str}"
        potential_label = f'{unique_id}_{potential_label}'
        model_filename = unique_id
        print("potential label: ", potential_label)
        # Prepare the regularizer
        regularizer = {
            "ridge_1b": ridge_1b,
            "ridge_2b": ridge_2b,
            "ridge_3b": ridge_3b,
            "curvature_2b": curvature_2b,
            "curvature_3b": curvature_3b
        }

        try:
            # Train the model in parallel with some # of cores (optional)
            chemical_system, trainer_model, bspline_config, _, _,_ = automater.train_model_from_df(
                model_filename=model_filename,
                training_data=automater.get_training_set(),
                testing_data=automater.get_testing_set(),
                regularizer=regularizer,
                weight=weight,
                batch_size=settings['batch_size'],
                num_cores=settings['cores_per_job'],
                sample_weights=settings['sample_weights'],
                exclude_elements=settings['excluded_testing_elements'],
            )
            return chemical_system, trainer_model, bspline_config, model_filename, potential_label
        except Exception as exc:
            return None, f"Training failed: {exc}"
    def validate_model( settings, chemical_system, trainer_model, bspline_config, model_filename, potential_label, trial_id=0):
        worker = get_worker()
        automater = worker.plugins['uf3_automater']
        worker_id = worker.plugins['uf3_worker_id']
        analyzer = ModelAnalyzer(
            chemical_system, 
            trainer_model, 
            bspline_config, 
            automater.get_training_set(),
            automater.get_testing_set(),
            automater.get_fitting_features(),
            dft_validation_folder = settings.get('dft_validation_folder', None),
        )

        
        analyzer.set_dft_validation_structures(automater.get_dft_validation_structures())
        analyzer.set_lammps_validation_structures(automater.get_lammps_validation_structures())
        analyzer.set_single_crystal_validation_structures(automater.get_single_crystal_validation_structures())
        analyzer.set_up_folder_structure()
        for pair in chemical_system.interactions_map[2]:
            analyzer.fix_repulsion_2b(pair=pair, r_target=1.0, min_curvature=50.0)
        normalized_error=0
        result = None
        if(settings['loss_weights']['error'] == 0):
            normalized_error = 0
        else:
            #result = analyzer.test_model_prediction(training=False, use_elements=chemical_system.element_list,)
            #normalized_error= 0.5*((result['rmse_e']/result['sigma_e']) + (result['rmse_f']/result['sigma_f']))
            #normalized_error = 0.5*(result['rmse_e'] + result['rmse_f'])
            normalized_error=0
        

        potential_file = os.path.join(os.getcwd(), "potentials", model_filename)
        analyzer.export_model_to_lammps(
            os.path.join(os.getcwd(), "potentials"), 
            model_filename, 
            True, 
            "ntaormina", 
            "metal", 
            "nk"
        )
        model_folder = os.path.join(os.getcwd(), "model_files")
        os.makedirs(model_folder, exist_ok=True)
        analyzer.export_model_to_json(model_folder, model_filename)
        print("export model to lammps")
        def _validate_potential_with_timeout(analyzer, potential_label, model_filename, element_list, rmse,validations_to_run, relative_order, timeout=3600):
            # assumes 999 loss if the simulation takes too long (it blew up)
            def run_validation():
                print("calling validate potential - optuna handler")
                return analyzer.validate_potential(potential_label, model_filename, element_list, rmse, validations_to_run, relative_order)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_validation)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    print("Validation took too long and was aborted.")
                    return None
        if settings['loss_weights']['error'] == 0:
            rmse = False
        else:
            rmse = False #debug 
        try:
            validation_data = _validate_potential_with_timeout(
                analyzer, potential_label, potential_file, 
                chemical_system.element_list, rmse, settings['validations_to_run'], settings['relative_orders'], timeout=7200
            )
            print("validation data: ", validation_data)
            if isinstance(validation_data, list):
                for valid in validation_data:
                    if valid is not None and result is not None:
                        valid.set_normalized_error(normalized_error)
                        valid.set_force_rmse(result['rmse_f'])
                        valid.set_energy_rmse(result['rmse_e'])
            else:
                if result is not None:
                    validation_data.set_normalized_error(normalized_error)
                    validation_data.set_force_rmse(result['rmse_f'])
                    validation_data.set_energy_rmse(result['rmse_e'])
            if(validation_data is None):
                print("validation data is none! timed out")
                return None, 999  # Assigning high loss
        except Exception as e:
            return {"status": "failed", "error": f"Validation error: {str(e)}"}

        # Save validation data
        automater.store_validation_data_alternate_dir(model_filename, validation_data, os.getcwd())
        print("validation data stored")
        return validation_data, normalized_error
    def train_model_and_validate( params, settings, trial_id=0, pure=False):
        worker = get_worker()
        automater = worker.plugins['uf3_automater']
        worker_id = worker.plugins['uf3_worker_id']

        if automater is None:
            raise RuntimeError("No automater loaded on this worker! Did you run worker_init_once?")

        # Train the model
        train_result = self.train_model(params, settings, trial_id)
        if train_result is None or isinstance(train_result, str):
            return None
        chemical_system, trainer_model, bspline_config, model_filename, potential_label = train_result
        validation_data, normalized_error = self.validate_model(
            settings, chemical_system, trainer_model, bspline_config, model_filename, potential_label, trial_id)
        return validation_data, normalized_error
    def parse_validation_data(self, validation_data, normalized_error, acceptable_errors, weights):
        all_data = []
        all_results = []
        
        if isinstance(validation_data, list):
            for data in validation_data:
                all_data.append(data)
        else:
            all_data.append(validation_data)
            
        for data in all_data:
            result = {}
            if(isinstance(data, ValidationData)):
                data2 = data.get_validation_data()
                for validation_type, validation_results in data2.items():
                    # Check if validation_results is a list of dictionaries
                    if isinstance(validation_results, list):
                        for res in validation_results:
                            if isinstance(res, dict):
                                for key, value in res.items():
                                    if isinstance(value, (int, float)):  # Check if the value is numeric
                                        result[key] = result.get(key, 0) + value
                            else:
                                print("its not a dictionary bud")  # Handle non-dictionary list items
                    else:
                        # Handle case where validation_results is a single dictionary
                        for key, value in validation_results.items():
                            if isinstance(value, (int, float)):  # Check if the value is numeric
                                result[key] = result.get(key, 0) + value
            else: 
                print("validation data list is broken")
                print(data)
            
            all_results.append(result)
        
        # Compute the average for each metric
        avg_result = {}
        num_results = len(all_results)
        
        if num_results > 0:
            for res in all_results:
                for key, value in res.items():
                    avg_result[key] = avg_result.get(key, 0) + value / num_results
        else:
            print("no validation results")
            return None
        
        final_loss = self.loss_function(avg_result, normalized_error, acceptable_errors, weights)
        print("Final Loss:", final_loss)
        if final_loss is None:
            return None
        if(final_loss <= 0.00001):
            return None
        else:
            return final_loss    
    def run(self, temp_search=False):
        self.init_workers()
        study = self.optimize()
        self.client.close()
        self.cluster.close()
        self.cleanup(temp_search)
        return study
    
class UF3OptunaTPETrainer(UF3AbstractOptunaTrainer):
    def __init__(self, base_directory, automater, settings, job_params):
        super().__init__(base_directory, automater, settings, job_params)

    def run(self, temp_search=False):
        self.init_workers()

        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            n_startup_trials=self.settings.get('n_startup_trials', 10)
        )

        study = optuna.create_study(
            sampler=sampler,
            study_name="UF3_Distributed_Optuna",
            direction="minimize"
        )

        study.optimize(
            lambda trial: self.objective(trial),
            n_trials=self.settings.get('n_trials', 2),
            n_jobs=self.settings.get('n_workers', 2)
        )

        print("Best value:", study.best_value)
        print("Best params:", study.best_params)

        self.client.close()
        self.cluster.close()
        self.cleanup(temp_search)

        return study
    
    def objective(self, trial):
        ridge_1b = trial.suggest_float("ridge_1b", 1e-9, 1e-1, log=True)
        ridge_2b = trial.suggest_float("ridge_2b", 1e-9, 1e-1, log=True)
        ridge_3b = trial.suggest_float("ridge_3b", 1e-9, 1e-1, log=True)
        curvature_2b = trial.suggest_float("curvature_2b", 1e-9, 1e-1, log=True)
        curvature_3b = trial.suggest_float("curvature_3b", 1e-9, 1e-1, log=True)
        weight = trial.suggest_float("weight", 0.01, 1.0, log=False)
        params = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight)

        if self.settings['optimize_sample_weights']:
            weights = [
                trial.suggest_float(f"sample_weights_{k}", 0.01, 1.0, log=False)
                for k in self.settings['sample_weights_to_optimize']
            ]
            normed = [w / sum(weights) for w in weights]
            self.settings['sample_weights'].update(dict(zip(
                self.settings['sample_weights_to_optimize'], normed)))

        future = self.client.submit(
            distributed_train_model_and_validate,
            UF3OptunaTPETrainer.train_model,  # bound to class method
            UF3OptunaTPETrainer.validate_model,  # bound to class method
            params,
            self.settings,
            trial_id=trial.number,
            pure=False
        )
        result = future.result()

        if isinstance(result, tuple) and len(result) == 2:
            validation_data, normalized_error = result
        else:
            return None

        if validation_data is None:
            return None

        loss = self.parse_validation_data(
            validation_data,
            normalized_error,
            self.settings['acceptable_errors'],
            self.settings['loss_weights']
        )
        return loss

class UF3OptunaTPEFineTuningTrainer(UF3AbstractOptunaTrainer):
    def __init__(self, base_directory, automater, settings, job_params, model_path):
        super().__init__(base_directory, automater, settings, job_params)
        self.model_path = model_path

    def run(self, temp_search=False):
        self.init_workers()

        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            n_startup_trials=self.settings.get('n_startup_trials', 10)
        )

        study = optuna.create_study(
            sampler=sampler,
            study_name="UF3_Distributed_Optuna",
            direction="minimize"
        )

        study.optimize(
            lambda trial: self.objective(trial),
            n_trials=self.settings.get('num_trials', 2),
            n_jobs=self.settings.get('n_workers', 2)
        )

        print("Best value:", study.best_value)
        print("Best params:", study.best_params)

        self.client.close()
        self.cluster.close()
        self.cleanup(temp_search)

        return study
    def train_model(params, settings, trial_id=0):
        worker = get_worker()
        automater = worker.plugins['uf3_automater']
        worker_id = worker.plugins['uf3_worker_id']

        if automater is None:
            raise RuntimeError("No automater loaded on this worker! Did you run worker_init_once?")

        (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight, alpha, model) = params
        potential_label = (
            f"trial_{trial_id}_r1b_{ridge_1b}_r2b_{ridge_2b}_r3b_{ridge_3b}_"
            f"curv2b_{curvature_2b}_curv3b_{curvature_3b}_w_{weight}"
        )
        unique_id = str(uuid.uuid4())[:32]
        if settings.get('optimize_sample_weights', False):
            weight_str = "_".join([f"{k}_{v:.2f}" for k, v in settings['sample_weights'].items()])
            potential_label += f"_sw_{weight_str}"
        potential_label = f'{unique_id}_{potential_label}'
        model_filename = unique_id
        print("potential label: ", potential_label)
        # Prepare the regularizer
        regularizer = {
            "ridge_1b": ridge_1b,
            "ridge_2b": ridge_2b,
            "ridge_3b": ridge_3b,
            "curvature_2b": curvature_2b,
            "curvature_3b": curvature_3b
        }

        try:
            chemical_system, trainer_model,bspline_config, features= automater.import_model(regularizer, "/blue/ypchen/ntaormina/uf3/AlN/test_auto/UF3-Tools/test_model")

            generator = UF3LammpsPotentialGenerator(trainer_model, chemical_system, bspline_config)
            generator.potential_to_model_json('/blue/ypchen/ntaormina/uf3/AlN/test_auto/UF3-Tools/grid_search_results/worker_c0713a-s21.ufhpc-4099472/db916af2-e2d1-4da2-a54e-bbb00227', '/blue/ypchen/ntaormina/uf3/AlN/test_auto/UF3-Tools/db916af2-e2d1-4da2-a54e-bbb00227.json', 'db916af2-e2d1-4da2-a54e-bbb00227_trial_445_r1b_1.3767099184507492e-09_r2b_9.354851029931691e-09_r3b_2.8391161554856034e-08_curv2b_1.2092727600321928e-06_curv3b_9.25158512840749e-08_w_0.9985110353270067_sw_structure_0.04_isif_pressure2_0.02_AlN_structures_0.24_solid_n2_structure_0.11_relaxed_polytypes_0.15_elastic_tensor_2x2x2_0.19_complete_structures_0.24')

            automater.import_model(regularizer, model)
            analyzer = ModelAnalyzer(chemical_system, trainer_model, bspline_config, automater.get_training_set(), automater.get_testing_set(), automater.get_fitting_features(), dft_validation_folder='/blue/ypchen/ntaormina/uf3/AlN/test_auto/UF3-Tools/validation/structures/')
            model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            materials = ['mp-661']
            analyzer.set_dft_validation_structures(materials)
            analyzer.set_lammps_validation_structures(materials)
            analyzer.set_up_folder_structure()
            analyzer.retrieve_model_files()
            for pair in chemical_system.interactions_map[2]:
                analyzer.fix_repulsion_2b(pair=pair, r_target=1.0, min_curvature=50.0)
            # results = analyzer.analyze_model(automater.data_coordinator)
            # energy_ids, force_ids, energy_bad, force_bad = analyzer.find_problematic_data(2.5, 2.5)
            # output_file = "energy_bad.xlsx"
            # energy_bad.to_excel(output_file)
            # output_file = "force_bad.xlsx"
            # force_bad.to_excel(output_file)

            analyzer.export_model_to_lammps("potentials/", model_filename, True, "ntaormina", "metal", "nk")
            
            # Train the model in parallel with some # of cores (optional)
            chemical_system, trainer_model, bspline_config, _, _,_ = automater.fine_tune_model_from_dataframe_blend(
                trainer=automater.get_trainer(),
                model_filename=model_filename,
                training_data=automater.get_training_set(),
                testing_data=automater.get_testing_set(),
                regularizer=regularizer,
                weight=weight,
                batch_size=settings['batch_size'],
                num_cores=settings['cores_per_job'],
                sample_weights=settings['sample_weights'],
                exclude_elements=settings['excluded_testing_elements'],
                alpha=alpha
            )
            return chemical_system, trainer_model, bspline_config, model_filename, potential_label
        except Exception as exc:
            return None, f"Training failed: {exc}"
    def objective(self, trial):
        ridge_1b = trial.suggest_float("ridge_1b", 1e-9, 1e-1, log=True)
        ridge_2b = trial.suggest_float("ridge_2b", 1e-9, 1e-1, log=True)
        ridge_3b = trial.suggest_float("ridge_3b", 1e-9, 1e-1, log=True)
        curvature_2b = trial.suggest_float("curvature_2b", 1e-9, 1e-1, log=True)
        curvature_3b = trial.suggest_float("curvature_3b", 1e-9, 1e-1, log=True)
        weight = trial.suggest_float("weight", 0.01, 1.0, log=False)
        alpha = trial.suggest_float("alpha", 0.01, 1.0, log=False)
        params = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight, alpha, self.model_path)

        if self.settings['optimize_sample_weights']:
            weights = [
                trial.suggest_float(f"sample_weights_{k}", 0.01, 1.0, log=False)
                for k in self.settings['sample_weights_to_optimize']
            ]
            normed = [w / sum(weights) for w in weights]
            self.settings['sample_weights'].update(dict(zip(
                self.settings['sample_weights_to_optimize'], normed)))

        future = self.client.submit(
            distributed_train_model_and_validate,
            UF3OptunaTPEFineTuningTrainer.train_model,  # bound to class method
            UF3OptunaTPEFineTuningTrainer.validate_model,  # bound to class method
            params,
            self.settings,
            trial_id=trial.number,
            pure=False
        )
        result = future.result()

        if isinstance(result, tuple) and len(result) == 2:
            validation_data, normalized_error = result
        else:
            return None

        if validation_data is None:
            return None

        loss = self.parse_validation_data(
            validation_data,
            normalized_error,
            self.settings['acceptable_errors'],
            self.settings['loss_weights']
        )
        return loss