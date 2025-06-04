from UF3Tools.uf3_handler import *
from UF3Tools.simulations import *
from UF3Tools.automation.ModelAutomater import *
import os
import shutil
from tqdm import tqdm
import os
import math
from dask.distributed import Client
import optuna
from dask_jobqueue import SLURMCluster
from dask.distributed import get_worker, WorkerPlugin
from UF3Tools.common.MPAPIKey import MP_API_KEY
import uuid
import os
import shutil
from datetime import datetime
import concurrent.futures
import time
import optuna
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# UF3 imports
from UF3Tools.uf3_handler import *
from UF3Tools.simulations import *
from UF3Tools.automation.ModelAutomater import *

###########################################
# -- GLOBAL VARIABLE ON EACH WORKER --
###########################################
GLOBAL_AUTOMATER = None
GLOBAL_WORKER_ID = None

def worker_init_once(
    base_directory,
    element_list,
    chemical_system,
    bspline_config,
    degree,
    ignore_list,
    settings
):
    """
    This function is called ONCE per Dask worker to load 
    the big data into a global variable. We do any heavy I/O here.
    """
    print("worker init once start")
    global GLOBAL_AUTOMATER, GLOBAL_WORKER_ID
    print("global vars init")
    import socket
    worker_hostname = socket.gethostname()  # Identify worker machine
    worker_pid = os.getpid()  # Get the process ID (unique per worker)

    # Assign worker ID based on PID (this is guaranteed unique for the node)
    GLOBAL_WORKER_ID = f"{worker_hostname}-{worker_pid}"
    print(f"[worker_init_once] Worker {GLOBAL_WORKER_ID} initializing data...")
    worker_id = GLOBAL_WORKER_ID or "unknown"
    print("global variables set ")
    print(f"[worker_init_once] Initializing data on worker: {worker_hostname}")
    base_dir = base_directory
    trial_folder = os.path.join(base_dir, "optuna_trials", f"worker_{worker_id}")
    os.makedirs(trial_folder, exist_ok=True)
    print("init once trial folder ")

    # If we need to copy a "validation_base" folder or something:
    validation_src = os.path.join(base_dir, "validation_base")
    if os.path.exists(validation_src):
        validation_dst = os.path.join(trial_folder, "validation", "structures")
        if not os.path.exists(validation_dst):
            shutil.copytree(validation_src, validation_dst, dirs_exist_ok=True)
    # Switch to the trial_folder so all output goes there
    print("init once validation folder")
    old_cwd = os.getcwd()
    

    automater = ModelAutomater(
        directories=[],
        run_data_path="",
        feature_filename="",
        element_list=element_list,
        degree=degree,
        chemical_system=chemical_system,
        bspline_config=bspline_config,
        num_cores=1
    )
    print("init once automater")

    automater.set_ignore_list(ignore_list)

    automater.load_existing_base_data()
    automater.load_existing_features()
    automater.load_filtered_data()
    automater.consolidate_data()
    if settings['testing_sets'] is not None:
        automater.select_testing_set(settings['testing_sets'], include_in_training=False, test_split_fraction=settings['train_test_split'], seed=42)
    else:
        automater.split_data(settings['train_test_split'], settings['unique_indices'], settings['unique_indices'], settings['excluded_testing_sets'])
    automater.set_dft_validation_structures(settings['testing_mp_structures'])
    automater.set_lammps_validation_structures(settings['testing_mp_structures'])
    
    print("init once data laoded")
    # Store in the global
    GLOBAL_AUTOMATER = automater
    os.chdir(trial_folder)
    automater.download_validation_structures()
    
    if 'formation_energy' in settings['validations_to_run']:
        if 'formation_energy_structures' in settings:
            automater.download_structures(settings['formation_energy_structures'])
            automater.set_single_crystal_validation_structures(settings['formation_energy_structures'])
        else:
            raise ValueError("formation_energy_structures not provided in settings")
    automater.load_fitting_data(automater.get_training_set(), automater.get_testing_set(), batch_size=settings['batch_size'], num_cores=settings['cores_per_job'])
    
    return f"Data loaded on {worker_hostname}"

def validate_potential_with_timeout(analyzer, potential_label, model_filename, element_list, rmse,validations_to_run, relative_order, timeout=3600):
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
# Define loss function with normalized errors
def loss_function(avg_result, normalized_error, acceptable_errors, weights):
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
def parse_validation_data(validation_data, normalized_error, acceptable_errors, weights):
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
    
    final_loss = loss_function(avg_result, normalized_error, acceptable_errors, weights)
    print("Final Loss:", final_loss)
    if final_loss is None:
        return None
    if(final_loss <= 0.00001):
        return None
    else:
        return final_loss
def train_with_preloaded_data(params, settings, trial_id=0):
    """
    Trains + validates the UF3 model using the data
    in GLOBAL_AUTOMATER (already loaded).
    
    Params is a tuple: (r1b, r2b, r3b, c2b, c3b, weight)
    settings is a dict with batch_size, etc.
    trial_id is just for naming the subfolder, logs, etc.
    """
    global GLOBAL_AUTOMATER, GLOBAL_WORKER_ID
    automater = GLOBAL_AUTOMATER
    worker_id = GLOBAL_WORKER_ID or "unknown"

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
        chemical_system, trainer_model, bspline_config, _, _ = automater.train_model_from_df(
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
    except Exception as exc:
        return None, f"Training failed: {exc}"
    
    # Validation
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
        result = analyzer.test_model_prediction(training=False, use_elements=settings['element_list'],)
        normalized_error= 0.5*((result['rmse_e']/result['sigma_e']) + (result['rmse_f']/result['sigma_f']))
    

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
    if settings['loss_weights']['error'] == 0:
        rmse = False
    else:
        rmse = False #debug 
    try:
        validation_data = validate_potential_with_timeout(
            analyzer, potential_label, potential_file, 
            settings['element_list'], rmse, settings['validations_to_run'], settings['relative_orders'], timeout=7200
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


def objective(trial, client, settings):
    """
    Optuna objective that:
      1) Samples hyperparams
      2) Submits them to a Dask worker
      3) Retrieves the result (RMSE)
    """
    # Pick hyperparams
    ridge_1b     = trial.suggest_float("ridge_1b", 1e-9, 1e-1, log=True)
    ridge_2b     = trial.suggest_float("ridge_2b", 1e-9, 1e-1, log=True)
    ridge_3b     = trial.suggest_float("ridge_3b", 1e-9, 1e-1, log=True)
    curvature_2b = trial.suggest_float("curvature_2b", 1e-9, 1e-1, log=True)
    curvature_3b = trial.suggest_float("curvature_3b", 1e-9, 1e-1, log=True)
    weight       = trial.suggest_float("weight",    0.01, 1.0,  log=False)

    params = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight)
    
    if settings['optimize_sample_weights']:
        sample_weights = {}
        raw_weights = []
        
        for key in settings["sample_weights_to_optimize"]:
            raw_weight = trial.suggest_float(f"sample_weights_{key}", 0.01, 1.0, log=False)
            raw_weights.append(raw_weight)

        # Normalize so they sum to 1
        weight_sum = sum(raw_weights)
        normalized_weights = [w / weight_sum for w in raw_weights]

        for key, norm_weight in zip(settings["sample_weights_to_optimize"], normalized_weights):
            sample_weights[key] = norm_weight

        settings['sample_weights'].update(sample_weights)

    # Submit the training/validation to a worker
    future = client.submit(
        train_with_preloaded_data, 
        params,
        settings,
        trial_id=trial.number,  # for labeling
        pure=False
    )
    
    result = future.result()

    if isinstance(result, tuple) and len(result) == 2:
        validation_data, normalized_error = result
    else:
        print("weird else case: ", result)
        return None


    if validation_data is None:
        # Return a large penalty
        print(f"[Trial {trial.number}] failed: {validation_data}")
        return None
    print(validation_data)
    # Parse RMSE from validation_data
    # Adjust to your actual data structure
    loss = parse_validation_data(validation_data, normalized_error, settings['acceptable_errors'], settings['loss_weights'])

    return loss

from dask import config
def run_optuna_tpe_with_dask_hpc(
    automater,
    settings,
    job_params,
    n_trials=2,
    n_workers=2,
    n_startup_trials=100,
    temp_search=False
):
    base_directory = os.getcwd()
    automater_params = {
        'ignore_list':automater.get_ignore_list(),
        'element_list': automater.get_element_list(),
        'chemical_system':automater.get_chemical_system(),
        'bspline_config':automater.get_bspline_config(),
        'degree':automater.get_degree()
        
    }
    settings.update(automater_params)
    settings.update(job_params)
    import dask

    dask.config.set({
        "distributed.scheduler.worker-ttl": "600 minutes",
        "distributed.client.heartbeat": "60s"
    })
    # 1) Create the SLURM cluster
    cluster = SLURMCluster(
        account=job_params['account'],
        cores=job_params['cores_per_job'],
        processes=1,
        memory=job_params['memory'],
        walltime=job_params['walltime'],
        job_extra=job_params['extra'],
        log_directory=automater.get_base_directory(),
    )
    
    

    # 2) Create the client
    client = Client(cluster, heartbeat_interval=30000, timeout=9999999999)
    def show_worker_config():
        
        return dask.config.get("distributed.scheduler.worker-ttl")

    worker_configs = client.run(show_worker_config)
    print(worker_configs)
    cluster.scale(n_workers)
    print(dask.config.config)

    client.wait_for_workers(n_workers)
    print("[INFO] Dask dashboard:", client.dashboard_link)
    # 3) Initialize data on each worker once
    init_future = client.run(
        worker_init_once,
        settings['base_directory'],
        automater.get_element_list(),
        automater.get_chemical_system(),
        automater.get_bspline_config(),
        automater.get_degree(),
        automater.get_ignore_list(),
        settings,
    )
    # .run() returns a dict {worker_address: return_value}
    if not init_future or any(val is None for val in init_future.values()):
        raise RuntimeError("Worker data initialization failed! Some workers may not have loaded data.")
    else:
        print("[INFO] Data successfully initialized on workers:", init_future)

    # 4) Create your Optuna study
    sampler = optuna.samplers.TPESampler(multivariate=True, 
                                         group=True,
                                         n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        sampler=sampler,
        study_name="UF3_Distributed_Optuna",
        direction="minimize",
        
        
    )

    # 5) Launch optimization
    study.optimize(
        lambda trial: objective(trial, client, settings),
        n_trials=n_trials,
        n_jobs=n_workers
    )

    # 6) Results
    print("Study best value =", study.best_value)
    print("Study best params=", study.best_params)

    # 7) Cleanup
    client.close()
    cluster.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if(temp_search):
        backup_folder = backup_folder = os.path.join(base_directory, "temp_search_results", timestamp)
    else: 
        backup_folder = os.path.join(base_directory, "grid_search_results", timestamp)
        
    if settings['debug'] == False:
        # copy results 
        os.makedirs(backup_folder, exist_ok=True)
        optuna_trials_path = os.path.join(base_directory, "optuna_trials")
        for worker_folder in os.listdir(optuna_trials_path):
            source_path = os.path.join(optuna_trials_path, worker_folder)
            # delete the validation folder
            for root, dirs, files in os.walk(source_path, topdown=True):
                for directory in dirs:
                    if "validation" in directory:
                        validation_path = os.path.join(root, directory)
                        shutil.rmtree(validation_path)  # Delete the validation folder
                        print(f"Deleted validation folder: {validation_path}")
            #move the worker folder to results folder
            shutil.move(source_path, backup_folder)
            print(f"Moved: {worker_folder} -> {backup_folder}")
            
        #clear dask worker logs 
        current_directory = os.getcwd()

        for filename in os.listdir(current_directory):
            if "dask-worker" in filename: 
                file_path = os.path.join(current_directory, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    print("Deletion process completed.")