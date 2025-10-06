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
import pyswarms as ps
from dask.distributed import Client
from functools import partial
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
    trial_folder = os.path.join(base_dir, "pso_trials", f"worker_{worker_id}")
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
    norm_elastic = abs(avg_elastic_constant) / acceptable_errors['elastic_constants']
    norm_bulk = abs(bulk_modulus) / acceptable_errors['bulk_modulus']
    norm_shear = abs(shear_modulus) / acceptable_errors['shear_modulus']
    norm_lattice = abs(lattice_constants) / acceptable_errors['lattice_constants']
    
    # Weights (adjustable for importance)
    weight_cohesive = 5.0
    weight_elastic = 3.0
    weight_bulk = 2.0
    weight_shear = 2.0
    weight_lattice = 1.0
    
    # Calculate loss with normalized errors
    loss = (
        weight_cohesive * norm_cohesive +
        weight_elastic * norm_elastic +
        weight_bulk * norm_bulk +
        weight_shear * norm_shear +
        weight_lattice * norm_lattice
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
        return 999999
    
    final_loss = loss_function(avg_result, acceptable_errors)
    print("Final Loss:", final_loss)
    if(final_loss <= 0.1):
        print("returning 999999 loss")
        return 999999
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

    downloader = StructureDownloader(MP_API_KEY)
    all_materials = downloader.get_structure_list(settings['chemical_formula'])
    filtered = [
        m for m in all_materials 
        if m.split(":")[0] in settings['testing_mp_structures']
    ]
    analyzer.set_dft_validation_structures(filtered)
    analyzer.set_lammps_validation_structures(filtered)
    analyzer.set_up_folder_structure()
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
        validations_to_run = {
            "lattice_constant": True,
            "cohesive_energy": True,
            "elastic_tensor": True,
            "surface_energy": False
        }
        validation_data = analyzer.validate_potential(potential_label, model_filename, settings['element_list'], validations_to_run)
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return {"status": "failed", "error": f"Validation error: {str(e)}"}

    # Save validation data
    automater.store_validation_data_alternate_dir(model_filename, validation_data, os.getcwd())
    print("validation data stored")
    return validation_data

def evaluate_hyperparams(particle, settings):
    """
    Prepares hyperparameters for model training.
    This function does NOT call client.submit().
    """
    # Extract hyperparameters
    ridge_1b, ridge_2b, ridge_3b = particle[0:3]
    curvature_2b, curvature_3b = particle[3:5]
    weight = particle[5]
    sample_weights = particle[6:]

    params = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight)
    settings['sample_weights'] = {f"sw_{i}": sw for i, sw in enumerate(sample_weights)}

    # Return parameters and settings (no Dask submission here)
    return params, settings


def objective_function(X, client, settings, batch_size=5):
    """
    Runs PSO particle evaluations in parallel using Dask.
    - Calls `evaluate_hyperparams()` to prepare inputs.
    - Uses `client.submit()` with explicit worker distribution.
    """
    num_particles = len(X)  # Total number of particles
    losses = np.zeros(num_particles)  # Store loss results

    scattered_settings = client.scatter(settings, broadcast=True)  # Reduce overhead

    for i in range(0, num_particles, batch_size):
        batch = X[i : i + batch_size]  # Select batch of particles
        
        # Prepare function calls for this batch
        tasks = [evaluate_hyperparams(particle, scattered_settings) for particle in batch]

        # Get the list of available workers
        worker_list = list(client.scheduler_info()['workers'].keys())
        print("worker list: ", worker_list)
        # Submit tasks ensuring different workers handle each call
        futures = []
        for idx, (params, setting) in enumerate(tasks):
            print("for i in task")
            worker = worker_list[idx % len(worker_list)]  # Round-robin workers
            future = client.submit(
                train_with_preloaded_data, params, setting,
                workers=worker,  # Ensure different workers execute
                allow_other_workers=True,  # Let Dask reassign if needed
                pure=False  # Avoid memoization issues
            )
            futures.append(future)

        # Gather results efficiently
        results = client.gather(futures)
        print("results obtained")
        # Compute loss values
        batch_losses = []
        for result in results:
            if result is None:
                batch_losses.append(9999)
            else:
                try:
                    loss = parse_validation_data(result, settings['acceptable_errors'])
                    batch_losses.append(loss)
                except:
                    batch_losses.append(9999)
        
        print("got batch losses")
        losses[i : i + batch_size] = batch_losses  # Store batch results
    print("returning losses: ", losses)
    return losses





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
    
    if(settings['optimize_sample_weights'] == True):
        sample_weights = {}
        for key in settings["sample_weights"]:
            sample_weights[key] = trial.suggest_float(f"sample_weights_{key}", 0.01, 5.0, log=True)
        settings['sample_weights'] = sample_weights

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


def create_slurm_cluster(num_workers, job_params):
    cluster = SLURMCluster(
        account=job_params['account'],
        cores=job_params['cores_per_job'],
        processes=1,
        memory=job_params['memory'],
        walltime=job_params['walltime'],
        job_extra=job_params['extra'],
        log_directory=os.getcwd()
    )
    cluster.scale(num_workers)


    client = Client(cluster)
    
    client.wait_for_workers(num_workers)
    print("[INFO] Dask dashboard:", client.dashboard_link)
    return client, cluster

def close_slurm_cluster(client, cluster):
    # 7) Cleanup
    client.close()
    cluster.close()

def run_pso_with_dask_hpc(
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
    client, cluster = create_slurm_cluster(n_workers, job_params)
    
    # 3) Initialize data on each worker once
    init_future = client.run(
        worker_init_once,
        settings['base_directory'],
        automater.get_element_list(),
        automater.get_chemical_system(),
        automater.get_maps(),
        automater.get_degree(),
        automater.get_ignore_list(),
        train_test_split=0.01  # or your desired ratio
    )
    # .run() returns a dict {worker_address: return_value}
    if not init_future or any(val is None for val in init_future.values()):
        raise RuntimeError("Worker data initialization failed! Some workers may not have loaded data.")
    else:
        print("[INFO] Data successfully initialized on workers:", init_future)

    regularizer_bounds = (np.full(6, 1e-9), np.full(6, 1e-1))  # 6 regularization terms
    sample_weight_bounds = (np.full(9, 0.01), np.full(9, 5.0))  # 9 sample weights

    # Combine into single bounds for PySwarms (stacking arrays)
    lower_bounds = np.concatenate([regularizer_bounds[0], sample_weight_bounds[0]])
    upper_bounds = np.concatenate([regularizer_bounds[1], sample_weight_bounds[1]])

    # Final bounds tuple for PySwarms
    bounds = (lower_bounds, upper_bounds)

    print("Lower bounds:", lower_bounds)
    print("Upper bounds:", upper_bounds)
    # 4) Create your Optuna study
    obj_function = partial(objective_function, client=client, settings=settings, batch_size=settings['num_workers'])

    # Initialize PySwarms
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_workers,
        dimensions=15,  
        options={'c1': 0.5, 'c2': 0.7, 'w': 0.9},
        bounds=bounds
    )

    # Run PSO with properly handled Dask submissions
    best_position, best_loss = optimizer.optimize(obj_function, iters=n_trials)
    print(f"Best Position: {best_position}, Best Loss: {best_loss}")

    close_slurm_cluster(client, cluster)
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(base_directory, "grid_search_results", timestamp)
    os.makedirs(backup_folder, exist_ok=True)
    for worker_folder in os.listdir(os.path.join(base_directory, "pso_trials")):
        shutil.move(os.path.join(base_directory, "pso_trials", worker_folder), backup_folder)