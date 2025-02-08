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
    maps,
    degree,
    ignore_list,
    train_test_split=0.01
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
        maps=maps,
        num_cores=1
    )
    print("init once automater")

    automater.set_ignore_list(ignore_list)
    

    # Potentially big data loads happen here
    automater.load_existing_base_data()
    automater.load_existing_features()
    automater.load_filtered_data()
    automater.consolidate_data()
    automater.split_data(train_test_split) # e.g. 99% train, 1% test
    print("init once data laoded")
    # Store in the global
    GLOBAL_AUTOMATER = automater
    os.chdir(trial_folder)
    return f"Data loaded on {worker_hostname}"
# Define loss function with normalized errors
def loss_function(avg_result, acceptable_errors):
    # Extract relevant values
    cohesive_energy = avg_result.get('cohesive_energy', 0)
    bulk_modulus = avg_result.get('bulk_modulus_voigt', 0)
    shear_modulus = (avg_result.get('shear_modulus_voigt', 0) + avg_result.get('shear_modulus_reuss', 0)) / 2
    lattice_constants = (avg_result.get('a', 0) + avg_result.get('b', 0) + avg_result.get('c', 0)) / 3
    
    elastic_constants = [
        avg_result.get('C11all', 0), avg_result.get('C12all', 0), avg_result.get('C13all', 0),
        avg_result.get('C22all', 0), avg_result.get('C23all', 0), avg_result.get('C33all', 0),
        avg_result.get('C44all', 0), avg_result.get('C55all', 0), avg_result.get('C66all', 0)
    ]
    avg_elastic_constant = sum(elastic_constants) / len(elastic_constants)
    
    
    # Normalize errors by acceptable thresholds
    norm_cohesive = abs(cohesive_energy) / acceptable_errors['cohesive_energy']
    norm_elastic  = abs(avg_elastic_constant) / acceptable_errors['elastic_constants']
    norm_bulk     = abs(bulk_modulus) / acceptable_errors['bulk_modulus']
    norm_shear    = abs(shear_modulus) / acceptable_errors['shear_modulus']
    norm_lattice  = abs(lattice_constants) / acceptable_errors['lattice_constants']
    
    # Weights (adjustable for importance)
    weight_cohesive = 5.0
    weight_elastic  = 3.0
    weight_bulk     = 2.0
    weight_shear    = 2.0
    weight_lattice  = 1.0
    
    # Calculate loss with normalized errors
    loss = (
        weight_cohesive * norm_cohesive +
        weight_elastic  * norm_elastic  +
        weight_bulk     * norm_bulk     +
        weight_shear    * norm_shear    +
        weight_lattice  * norm_lattice
    )
    return loss
def parse_validation_data(validation_data, acceptable_errors):
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
                    # Single dictionary
                    for key, value in validation_results.items():
                        if isinstance(value, (int, float)): 
                            result[key] = result.get(key, 0) + value
        else: 
            print("validation data list is broken")
            print(data)
        
        all_results.append(result)
    
    avg_result = {}
    num_results = len(all_results)
    if num_results > 0:
        for res in all_results:
            for key, value in res.items():
                avg_result[key] = avg_result.get(key, 0) + value / num_results
    else:
        print("no validation results")
        return 999
    
    final_loss = loss_function(avg_result, acceptable_errors)
    print("Final Loss:", final_loss)
    if(final_loss <= 0.1):
        return 999
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
    print(f"[Worker {worker_id}] potential label: {potential_label}")

    regularizer = {
        "ridge_1b": ridge_1b,
        "ridge_2b": ridge_2b,
        "ridge_3b": ridge_3b,
        "curvature_2b": curvature_2b,
        "curvature_3b": curvature_3b
    }

    try:
        chemical_system, trainer_model, bspline_config, features, _ = automater.train_model_parallel(
            model_filename=model_filename,
            training_data=automater.get_training_set(),
            testing_data=automater.get_testing_set(),
            regularizer=regularizer,
            weight=weight,
            batch_size=settings['batch_size'],
            num_cores=settings['cores_per_job'],
            sample_weights=settings['sample_weights']
        )
    except Exception as exc:
        return None, f"Training failed: {exc}"
    
    # Validation
    frame_name, feature_name = settings['parity_data']
    automater.set_parity_data(frame_name, feature_name)
    # not strictly needed if you do it once, but we'll ensure correct
    frame, feature = automater.get_parity_data()

    analyzer = ModelAnalyzer(
        chemical_system,
        trainer_model,
        bspline_config,
        automater.get_training_set(),
        automater.get_testing_set(),
        features
    )
    analyzer.set_dft_validation_structures(automater.get_dft_validation_structures())
    analyzer.set_lammps_validation_structures(automater.get_lammps_validation_structures())
    analyzer.set_up_folder_structure()
    if 'al' in settings['chemical_formula'].lower():
        analyzer.fix_repulsion_2b(pair=('Al', 'Al'), r_target=0.25, min_curvature=50.0)

    print("analyzer")
    analyzer.export_model_to_lammps(
        os.getcwd(), 
        model_filename, 
        True, 
        "ntaormina", 
        "metal", 
        "nk"
    )
    print("export model to lammps")

    try:
        print("lick my balls")
        validation_data = analyzer.validate_potential(potential_label, model_filename, settings['element_list'])
        print("potential validated")
    except Exception as e:
        return {"status": "failed", "error": f"Validation error: {str(e)}"}

    # Save validation data
    automater.store_validation_data_alternate_dir(model_filename, validation_data, os.getcwd())
    print("validation data stored")
    return validation_data


def objective(trial, client, settings):
    # Sample hyperparams
    ridge_1b     = trial.suggest_float("ridge_1b", 1e-9, 1e-1, log=True)
    ridge_2b     = trial.suggest_float("ridge_2b", 1e-9, 1e-1, log=True)
    ridge_3b     = trial.suggest_float("ridge_3b", 1e-9, 1e-1, log=True)
    curvature_2b = trial.suggest_float("curvature_2b", 1e-9, 1e-1, log=True)
    curvature_3b = trial.suggest_float("curvature_3b", 1e-9, 1e-1, log=True)
    weight       = trial.suggest_float("weight", 0.01, 1.0, log=False)

    params = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight)
    if settings.get('optimize_sample_weights', False):
        sample_weights = {}
        for key in settings["sample_weights_to_optimize"]:
            sample_weights[key] = trial.suggest_float(f"sample_weights_{key}", 0.01, 5.0, log=True)
        settings['sample_weights'].update(sample_weights)

    # Submit the training/validation to a worker
    future = client.submit(
        train_with_preloaded_data,
        params,
        settings,
        trial_id=trial.number,
        pure=False
    )
    
    result = future.result()

    if isinstance(result, tuple) and len(result) == 2:
        error, validation_data = result
    else:
        error, validation_data = None, result  # Assume result is the validation_data


    if validation_data is None:
        # Return a large penalty
        print(f"[Trial {trial.number}] failed: {validation_data}")
        return 1e6
    print(validation_data)
    # Parse RMSE from validation_data
    # Adjust to your actual data structure
    loss = parse_validation_data(validation_data, settings['acceptable_errors'])

    return loss


def run_optuna_with_dask_hpc(
    automater,
    settings,
    job_params,
    n_trials=2,
    n_workers=2
):
    base_directory = os.getcwd()
    automater_params = {
        'ignore_list':automater.get_ignore_list(),
        'element_list': automater.get_element_list(),
        'chemical_system':automater.get_chemical_system(),
        'maps':automater.get_maps(),
        'degree':automater.get_degree()
        
    }
    settings.update(automater_params)
    settings.update(job_params)
    # 1) Create the SLURM cluster
    cluster = SLURMCluster(
        account=job_params['account'],
        cores=job_params['cores_per_job'],
        processes=1,
        memory=job_params['memory'],
        walltime=job_params['walltime'],
        job_extra=job_params['extra'],
        log_directory=automater.get_base_directory()
    )
    cluster.scale(n_workers)


    # 2) Create the client
    client = Client(cluster)
    
    client.wait_for_workers(n_workers)
    print("[INFO] Dask dashboard:", client.dashboard_link)
    # 3) Initialize data on each worker once
    init_future = client.run(
        worker_init_once,
        settings["base_directory"],
        automater.get_element_list(),
        automater.get_chemical_system(),
        automater.get_maps(),
        automater.get_degree(),
        automater.get_ignore_list(),
        train_test_split=0.01  # or your desired ratio
    )
    # .run() returns a dict {worker_address: return_value}
    if not init_future or any(val is None for val in init_future.values()):
        raise RuntimeError("Worker data initialization failed on at least one worker.")

    print("[INFO] Data successfully initialized on workers:", init_future)

    # 4) Set up Optuna
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=500
    )
    study = optuna.create_study(
        sampler=sampler,
        study_name="UF3_Distributed_Optuna",
        direction="minimize",
    )

    # 5) Optimize
    study.optimize(
        lambda trial: objective(trial, client, settings),
        n_trials=n_trials,
        n_jobs=n_workers
    )

    print("Study best value =", study.best_value)
    print("Study best params=", study.best_params)

    # 6) Cleanup
    client.close()
    cluster.close()

    # 7) Post-processing or archiving any results/logs...
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(base_directory, "grid_search_results", timestamp)
    os.makedirs(backup_folder, exist_ok=True)
    optuna_trials_path = os.path.join(base_directory, "optuna_trials")

    if os.path.exists(optuna_trials_path):
        for worker_folder in os.listdir(optuna_trials_path):
            source_path = os.path.join(optuna_trials_path, worker_folder)
            if not os.path.isdir(source_path):
                continue
            # Example: removing subfolder(s) named "validation"
            for root, dirs, files in os.walk(source_path, topdown=True):
                for directory in dirs:
                    if "validation" in directory:
                        validation_path = os.path.join(root, directory)
                        shutil.rmtree(validation_path, ignore_errors=True)
                        print(f"Deleted validation folder: {validation_path}")
            # Move the entire worker folder
            shutil.move(source_path, backup_folder)
            print(f"Moved: {worker_folder} -> {backup_folder}")

    # (Optional) Clear old logs, if desired:
    # for filename in os.listdir(base_directory):
    #     if "dask-worker" in filename: 
    #         ...
    print("All done.")
