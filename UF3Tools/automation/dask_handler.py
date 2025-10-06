from UF3Tools.uf3_handler import *
from UF3Tools.simulations import *
from UF3Tools.automation.ModelAutomater import *
import os
import shutil
from tqdm import tqdm
import os
import math
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def train_param_combo(params, automater, settings, cores, worker_id):
    (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight) = params
    job_label = (
        f"r1b_{ridge_1b}_r2b_{ridge_2b}_r3b_{ridge_3b}_"
        f"curv2b_{curvature_2b}_curv3b_{curvature_3b}_w_{weight}"
    )
    regularizer = {
        "ridge_1b": ridge_1b,
        "ridge_2b": ridge_2b,
        "ridge_3b": ridge_3b,
        "curvature_2b": curvature_2b,
        "curvature_3b": curvature_3b
    }
    print(f"[Worker {worker_id}] Starting {job_label} ...")
    model_filename = job_label
    print("sample weight length: ", len(settings['sample_weights']))
    try:
        chemical_system, trainer_model, bspline_config, features, _ = automater.train_model_parallel(
            model_filename, 
            automater.get_training_set(), 
            automater.get_testing_set(),
            regularizer=regularizer, 
            weight=weight, 
            batch_size=settings['batch_size'],
            num_cores=cores,
            sample_weights=settings['sample_weights']
        )
    except Exception as e:
        return {
            "job_label": job_label,
            "status": "failed",
            "error": f"Training error: {str(e)}"
        }
    
    #------- now validate it ---------
    frame_name, feature_name = settings['parity_data']
    automater.set_parity_data(frame_name, feature_name)
    frame, feature = automater.get_parity_data()
    
    analyzer = ModelAnalyzer(chemical_system, trainer_model, bspline_config, automater.get_training_set(), automater.get_testing_set(), features[0])
    
    downloader = StructureDownloader(MP_API_KEY)
    all_materials = downloader.get_structure_list(settings['chemical_formula'])
    filtered = [m for m in all_materials if m.split(":")[0] in settings['testing_mp_structures']]
    print('materials filtered:', len(filtered))
    analyzer.set_dft_validation_structures(filtered)
    analyzer.set_lammps_validation_structures(filtered)
    print('structures set up')
    analyzer.set_up_folder_structure()
    #analyzer.retrieve_model_files()
    analyzer.fix_repulsion_2b(pair=('Al', 'Al'), r_target=0.25, min_curvature=50.0)
    print('analyzer set up')
    # Export model to LAMMPS
    analyzer.export_model_to_lammps(settings['job_folder'], model_filename, True, "ntaormina", "metal", "nk")
    print("about to validate")
    # Actually run the validation
    try:
        validation_data = analyzer.validate_potential(
            model_filename,    # train model label
            model_filename,    # test model label
            settings['element_list']       # elements
        )
    except Exception as e:
        return {
            "job_label": job_label,
            "status": "failed",
            "error": f"Validation error: {str(e)}",
            "validation_data": None
        }

    # 6) Save the validation data in the base_folder’s "potentials" or in job_folder
    #    Using your store method, or just do direct file writes:
    

    print(f"[Worker] Completed {job_label} with success.")
    return {
        "job_label": job_label,
        "status": "success",
        "validation_data": validation_data,
        "model_filename": model_filename
    }
    
def worker_subroutine(automater_params, worker_id, param_subset, cores):
        """
        Runs on a single Dask worker.  We:
          - Make a local subfolder 'grid_search/worker_<id>'
          - Write param_combos.txt
          - Possibly copy validation folder
          - For each param combo -> train, validate, store results
          - Return a DataFrame or list of dicts
        """
        print("worker subroutine")
        base_dir = automater_params['base_directory']
        gs_folder = os.path.join(base_dir, "grid_search")
        os.makedirs(gs_folder, exist_ok=True)
        
        worker_folder = os.path.join(gs_folder, f"worker_{worker_id}")
        os.makedirs(worker_folder, exist_ok=True)
        automater_params['job_folder'] = worker_folder

        # If you have a "validation" or "validation_base" folder to copy:
        validation_src = os.path.join(base_dir, "validation_base")  
        if os.path.exists(validation_src):
            validation_dst = os.path.join(worker_folder, "validation/structures/")
            if not os.path.exists(validation_dst):
                shutil.copytree(validation_src, validation_dst, dirs_exist_ok=True)

        # Write out the subset combos to a text file (so we can re-run manually if needed)
        combos_path = os.path.join(worker_folder, "param_combos.txt")
        with open(combos_path, "w") as f:
            for c in param_subset:
                # c is (r1b, r2b, r3b, c2b, c3b, w)
                line = (f"ridge_1b={c[0]} ridge_2b={c[1]} ridge_3b={c[2]} "
                        f"curvature_2b={c[3]} curvature_3b={c[4]} weight={c[5]}")
                f.write(line + "\n")
                
        automater = ModelAutomater(
            directories=[], 
            run_data_path="", 
            feature_filename="",
            element_list=automater_params['element_list'],
            degree=automater_params['degree'],
            chemical_system=automater_params['chemical_system'],
            maps=automater_params['maps'],
            num_cores=cores  # or whatever is appropriate
        )
        
        automater.base_directory = os.getcwd()
        automater.set_ignore_list(automater_params['ignore_list'])
        automater.load_existing_base_data()
        automater.load_existing_features()
        
        automater.load_filtered_data()
        automater.consolidate_data()
        training, testing = automater.split_data(0.01) 
        
        os.chdir(worker_folder)
        # Now run combos:
        results = []
        print("param subset:    ", param_subset)
        print("enumerate subset:", list(enumerate(param_subset)))
        for idx, combo in enumerate(param_subset):
            
            # combo = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight)
            # run_one_combination is your method that trains + validates
            print(f"[Worker {worker_id}] Starting combo {idx+1}/{len(param_subset)} ... {combo}")
            out_dict = train_param_combo(
                params=combo, 
                automater=automater,
                settings=automater_params,
                cores=cores, 
                worker_id=worker_id
            )
            out_dict["worker_id"] = worker_id
            print("got result:", out_dict)
            try:
                print("storign validation data")
                automater.store_validation_data_alternate_dir(out_dict['model_filename'], out_dict['validation_data'], automater_params['job_folder'])
                print("stored validation data")
            except Exception as e:
                print("exception e")
                return {
                    "job_label": combo,
                    "status": "failed",
                    "error": f"Storing results error: {str(e)}"
                }
            results.append(out_dict)
        print("combo loop done")
        # Convert results to a DataFrame, store in worker folder
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(worker_folder, "worker_validation_results.csv")
        df_results.to_csv(csv_path, index=False)
        print("returning results")
        return df_results

def run_grid_search_with_dask(automater,
                                regularizer_ranges,
                                weight_range,
                                num_workers=2,
                                sample_weights=None,
                                settings=None,
                                job_params=None):
        """
        This method uses Dask + SLURM to launch 'num_workers' Slurm jobs (i.e. dask workers),
        each of which runs a subset of the param combos.
        
        Parameters:
        -----------
        - regularizer_ranges : dict like {
            'ridge_1b': [1e-8, 1e-7],
            'ridge_2b': [1e-8, 1e-7],
            'ridge_3b': [1e-8, 1e-7],
            'curvature_2b': [1e-8, 1e-7],
            'curvature_3b': [1e-8, 1e-7]
          }
        - weight_range : list of floats, e.g. [0.3, 0.5]
        - num_workers : how many Slurm jobs (each job is one Dask worker).
        - queue       : Slurm partition/queue name
        - walltime    : HH:MM:SS
        - cores_per_job : # of CPU cores per worker
        - memory      : e.g. '4GB'
        """

        # 1) Build the parameter grid
        all_combos = []
        for r1b in regularizer_ranges['ridge_1b']:
            for r2b in regularizer_ranges['ridge_2b']:
                for r3b in regularizer_ranges['ridge_3b']:
                    for c2b in regularizer_ranges['curvature_2b']:
                        for c3b in regularizer_ranges['curvature_3b']:
                            for w in weight_range:
                                all_combos.append((r1b, r2b, r3b, c2b, c3b, w))
        
        if not all_combos:
            print("No parameter combos found. Check your ranges.")
            return
        
        chemical_system = automater.get_chemical_system()
        maps = automater.get_maps()
        element_list = automater.get_element_list()
        base_directory = os.getcwd()
        degree = automater.get_degree()
        automater_params = {
            'element_list': element_list,
            'chemical_system': chemical_system,
            'maps': maps,
            'base_directory': base_directory,
            'degree': degree,
            'batch_size': 1000,
            'ignore_list': automater.get_ignore_list(),
            'sample_weights': sample_weights
        }
        settings.update(automater_params)
        
        #2) Setup the Dask SLURM cluster
        cluster = SLURMCluster(
            account=job_params['account'],  # e.g. 'phillpot'
            cores=job_params['cores_per_job'],  # cores per job
            processes=1,           # 1 process that uses 'cores_per_job' threads, or just 1 thread if you like
            memory=job_params['memory'],  # e.g. '4GB'
            walltime=job_params['walltime'],  # e.g. '00:30:00'
            job_extra=job_params['extra'],  # e.g. ['--exclusive'] or other SBATCH extras
            log_directory=base_directory,  # where to write SLURM logs
        )
        print("Created SLURMCluster. Now scaling to", num_workers, "workers...")

        # Actually launch the jobs:
        cluster.scale(num_workers)
        
        # 3) Create the Dask client
        client = Client(cluster)
        client.wait_for_workers(num_workers)
        print("Dask dashboard URL:", client.dashboard_link)
        
        # 4) Split combos among the *num_workers* subsets
        combos_per_worker = math.ceil(len(all_combos) / num_workers)
        
        # 5) Submit tasks to each worker. 
        #    We'll do it so that each worker gets exactly one "big" task:
        #    That single task calls '_worker_subroutine' with the chunk of param combos.
        #    This ensures each worker has exactly one chunk. 
        futures = []
        print("all combos:", len(all_combos))
        print(all_combos)
        idx_start = 0
        #worker_subroutine(settings, 0, all_combos, job_params['cores_per_job'])
        for worker_id in range(num_workers):
            chunk = all_combos[idx_start: idx_start + combos_per_worker]
            idx_start += combos_per_worker
            if not chunk:
                break
            future = client.submit(worker_subroutine, settings, worker_id,  chunk,job_params['cores_per_job'], pure=False)
            futures.append(future)

        # 6) Gather results from all workers
        all_results = client.gather(futures)
        print("All results gathered.")
        

        client.close()
        cluster.close()
        
        #back up the worker folders. put into a timestamped folder in base_directory/grid_search_results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = os.path.join(base_directory, "grid_search_results", timestamp)
        os.makedirs(backup_folder, exist_ok=True)
        for worker_folder in os.listdir(os.path.join(base_directory, "grid_search")):
            shutil.move(os.path.join(base_directory, "grid_search", worker_folder), backup_folder)
            
        
        
