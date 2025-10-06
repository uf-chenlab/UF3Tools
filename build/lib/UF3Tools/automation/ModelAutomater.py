from UF3Tools.uf3_handler import *
from UF3Tools.simulations import *
import os
import shutil
from tqdm import tqdm
from threading import Lock
import os
import threading
from multiprocessing import Pool
import uuid
from datetime import datetime
import time
from UF3Tools.common.UF3Utility import daily_millis


def set_cpu_affinity(core_ids):
    """
    Set the CPU affinity for the current process to the specified core IDs.
    """
    try:
        os.sched_setaffinity(0, core_ids)  # Linux/Unix
    except AttributeError:
        print("CPU affinity setting is not supported on this platform.")
class CoreManager:
    def __init__(self, total_cores, cores_per_job):
        self.available_cores = list(range(total_cores))
        self.cores_per_job = cores_per_job
        self.lock = Lock()

    def allocate_cores(self):
        with self.lock:
            if len(self.available_cores) >= self.cores_per_job:
                allocated = self.available_cores[:self.cores_per_job]
                self.available_cores = self.available_cores[self.cores_per_job:]
                return allocated
            else:
                raise RuntimeError("Insufficient cores available!")

    def release_cores(self, cores):
        with self.lock:
            self.available_cores.extend(cores)
    def get_slurm_core_count(self):
        """
        Get the number of cores allocated by SLURM.
        :return: Number of cores assigned to the job.
        """
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


class ModelAutomater:
    def __init__(self, directories, run_data_path, feature_filename, element_list, degree, chemical_system, bspline_config, data_folder=None, num_cores=1, filtered_directory=None, filtered_history=[]):
         # Class-wide LUT for filter types to corresponding methods
        self.FILTER_FUNCTION_LUT = {
            'composition': 'filter_composition',
            'energy_above_hull': 'filter_energy_above_hull',
            'maximum_force': 'filter_force',
            'zero_force': 'remove_zero_force',
            'minimum_force': 'filter_force_minimum',
            'reduce_points': 'filter_reduce_data_points',
            'increase_points': 'filter_increase_data_points',
            'remove_by_id': 'filter_out_indices',
            'composition_range': 'filter_composition_range',
            'max_epa': 'filter_max_epa',
            'min_epa': 'filter_min_epa',
            'max_epa_by_run': 'filter_runs_by_epa_violation',
            'unreasonable_epas': 'filter_unreasonable_epas',
            'gas': 'filter_gas',
            'subsample':'filter_kmeans_subsample',
            'uf3_subsample':'filter_farthest_point_subsample',
            'cur_subsample': 'filter_cur_subsample',
            'min_pair_distance': 'filter_min_pair_distance',
            'min_pair_distance_by_element': 'filter_min_pair_distance_by_element',
            'isolated_pair_by_element': 'filter_isolated_pair_by_element',
            'isolated_atoms': 'filter_isolated_atoms',
            'invalid_gasp_runs': 'filter_invalid_gasp_runs',
            'duplicate_indices': 'filter_duplicate_indices',
            'unrelaxed_structures': 'filter_out_unrelaxed',
            'none': 'get_data'
            # Add new filters here as needed
        }
        
        self.directories = directories
        self.run_data_path = run_data_path
        self.feature_filename = feature_filename
        
        self.element_list = element_list
        self.degree = degree
        self.chemical_system = chemical_system
        self.bspline_config = bspline_config
        self.num_cores = num_cores
        self.base_directory = os.getcwd()
        self.data_folder = data_folder if data_folder is not None else self.base_directory
        print("data folder is: ", self.data_folder)
        self.pickle_directory = os.path.join(self.data_folder , 'data/pickle_files/')
        filtered_root = os.path.join(self.data_folder, 'data', 'pickle_files', 'filtered')
        os.makedirs(filtered_root, exist_ok=True)
        if filtered_directory is not None:
            self.filter_directory = filtered_directory
            os.makedirs(self.filter_directory, exist_ok=True)
        else:
            temp_name = uuid.uuid4().hex[:8]
            self.temp_filter_dir = os.path.join(filtered_root, temp_name)
            os.makedirs(self.temp_filter_dir, exist_ok=False)
            
            self.filter_directory = self.temp_filter_dir
        self.feature_directory = os.path.join(self.data_folder , 'data/featurized/')
        self.validations_to_run = {
                "lattice_constant": True,
                "cohesive_energy": True,
                "elastic_tensor": True,
                "surface_energy": True
            }
        self.config_directory = os.path.join(os.getcwd(), "config")
        #init variables
        self.feature_batch_size = 500
        self.data_coordinator = io.DataCoordinator()
        self.base_data_labels = []
        self.base_data_frames = {}
        self.filtered_data_frames = {}
        self.filtered_data_labels = []
        self.feature_paths = {}
        self.model_data = pd.DataFrame()
        self.data_ignore_list = []
        self.validation_datas = []
        self.validation_datas_all = []
        self.index_patterns = {}
        #self.initialize_directories()
        self.banned_name_list = ['cubic', 'hexagonal', 'fcc', 'bcc', 'hcp', 'diamond', 'graphite', 'amorphous', 'liquid', 'surface_energy',]
        self.trainer = None
        #self.structure_downloader = StructureDownloader(MP_API_KEY)
        # Validation Parameters
        self.validation_lammps_models = []
        self.cutoffs = {}
        self.parity_frame = None
        self.parity_feature = None
        self.dft_validation_structures = None
        self.lammps_validation_structures = None
        self.single_crystal_validation_structures = None
        self.fitting_df = None
        self.fitting_features = None
        self.trainer=None
        self.filter_history = filtered_history
    def set_filtered_directory(self, directory):
        self.filter_directory = directory
        os.makedirs(self.filter_directory, exist_ok=True)
    def get_filtered_directory(self):
        return self.filter_directory
    def set_fitting_df(self, fitting_df):
        self.fitting_df = fitting_df
    def get_fitting_df(self):
        return self.fitting_df
    def set_fitting_features(self, fitting_features):
        self.fitting_features = fitting_features
    def get_fitting_features(self):
        return self.fitting_features
    def set_dft_validation_structures(self, validation_structures):
        self.dft_validation_structures = validation_structures
    def get_dft_validation_structures(self):
        return self.dft_validation_structures
    def set_lammps_validation_structures(self, validation_structures):
        self.lammps_validation_structures = validation_structures
    def set_single_crystal_validation_structures(self, validation_structures):
        self.single_crystal_validation_structures = validation_structures
    def get_single_crystal_validation_structures(self):
        return self.single_crystal_validation_structures
    def download_structures(self, structures):
        if structures:
            for structure in structures:
                print("downloading structure: ", structure)
                mpid = structure
                directory = os.path.join("validation/structures/", structure)
                print("directory: ", directory)
                self.structure_downloader.download_single_structure(mpid, directory)
    def download_validation_structures(self):
        if self.dft_validation_structures:
            self.download_structures(self.dft_validation_structures)
    def get_lammps_validation_structures(self):
        return self.lammps_validation_structures
    def get_chemical_system(self):
        return self.chemical_system
    def get_maps(self):
        return self.bspline_config  
    def get_data_folder(self):
        return self.data_folder
    def get_bspline_config(self):
        return self.bspline_config
    def get_element_list(self):
        return self.element_list
    def get_degree(self):
        return self.degree
    def get_training_set(self):
        return self.training_set
    def get_testing_set(self):
        return self.testing_set
    def get_training_keys(self):
        return self.train_keys
    def get_testing_keys(self):
        return self.test_keys
    def get_trainer(self):
        return self.trainer
    def set_trainer(self, trainer):
        self.trainer=trainer
    def get_base_directory(self):
        return self.base_directory
    def initialize_directories(self, base_path="."):
        directories = [
            os.path.join(base_path, "data"),
            os.path.join(base_path, "data", "archive"),
            os.path.join(base_path, "data", "featurized"),
            os.path.join(base_path, "data", "pickle_files"),
            os.path.join(base_path, "data", "pickle_files", "filtered"),
            
            os.path.join(base_path, "generation"),
            
            os.path.join(base_path, "validation"),
            
            
            os.path.join(base_path, "validation", "lammps"),
            os.path.join(base_path, "validation", "lammps", "simulations"),
            os.path.join(base_path, "validation", "lammps", "simulations", "ElasticTensor"),
            os.path.join(base_path, "validation", "lammps", "simulations", "RelaxationSimulation"),
            os.path.join(base_path, "validation", "lammps", "simulations", "SurfaceEnergySimulation"),
            
            # We only create the top-level structures folder,
            # no sub-directories inside structures.
            os.path.join(base_path, "validation", "structures")
        ]
        dft_directories = [
            os.path.join(base_path, "validation", "dft"),
            os.path.join(base_path, "validation", "dft", "simulations"),
            os.path.join(base_path, "validation", "dft", "simulations", "ElasticTensor"),
            os.path.join(base_path, "validation", "dft", "simulations", "RelaxationSimulation"),
            os.path.join(base_path, "validation", "dft", "simulations", "SurfaceEnergySimulation"),
        ]
        if self.validation_folder is not None:
            directories.append(dft_directories)
    
        # Create each directory if it doesn't exist
        for directory in directories:
            os.makedirs(directory.lower(), exist_ok=True)
    def load_existing_data(self, directory):
        """
        Load existing data from the specified directory, ignoring files that match data types in data_ignore_list.
        """
        print("loading from directory: ", directory)
        data_parser = DataParser(directory, self.data_coordinator)
        data_files = {}
        local_data_labels = []
        for file in os.listdir(directory):
            if file.endswith('.pkl'):
                # Extract the data type from the file name and check against the ignore list
                data_type = file.replace('.pkl', '')
                data_type = data_type.replace('_filtered', '')
                if data_type in self.data_ignore_list:
                    continue
                data_parser.load_from_pickle(os.path.join(directory, file))
                data_files[file] = data_parser.get_data()
                local_data_labels.append(file)
        return data_files, local_data_labels
    def replace_existing_data(self, filename, data):
        filepath = os.path.join(self.pickle_directory, filename)
        parser = DataParser('', self.data_coordinator)
        parser.set_data(data)
        parser.export_to_pickle(filepath)
        
    def load_single_pickle_file(self, file_path):
        data_parser = DataParser('', self.data_coordinator)
        data_parser.load_from_pickle(file_path)
        return data_parser.get_data()
                
    def load_new_base_data(self, filter=True):
        for directory in self.directories:
            data_parser = DataParser([directory], io.DataCoordinator())
            data_parser.parse_directories(output=True, filter=filter)
            
            # Generate initial filename from the directory name
            filename = directory.split('/')[-1] or directory.split('/')[-2]

            # Check for banned names and duplicate files
            path_parts = directory.split('/')
            index = -1
            while filename in self.banned_name_list or \
                (self.pickle_directory + filename + '.pkl') in self._get_existing_pickle_files():
                index -= 1
                filename = path_parts[index] if -index <= len(path_parts) else f"unnamed_{abs(index)}"

            # Export to pickle with the resolved filename
            data_parser.export_to_pickle(self.pickle_directory + filename + '.pkl')
        
        self.load_existing_base_data()
    def load_new_base_data_with_subsampling(self, min_diff=1e-3, max_samples=1000):
        for directory in self.directories:
            data_parser = DataParser([directory], io.DataCoordinator())
            data_parser.parse_directories_with_subsampling(output=True, min_diff=min_diff, max_samples=max_samples)
            
            # Generate initial filename from the directory name
            filename = directory.split('/')[-1] or directory.split('/')[-2]

            # Check for banned names and duplicate files
            path_parts = directory.split('/')
            index = -1
            while filename in self.banned_name_list or \
                (self.pickle_directory + filename + '.pkl') in self._get_existing_pickle_files():
                index -= 1
                filename = path_parts[index] if -index <= len(path_parts) else f"unnamed_{abs(index)}"

            # Export to pickle with the resolved filename
            data_parser.export_to_pickle(self.pickle_directory + filename + '.pkl')
        
        self.load_existing_base_data()
    # Helper function to retrieve existing pickle files in the directory
    def _get_existing_pickle_files(self):
        import os
        return set(f for f in os.listdir(self.pickle_directory) if f.endswith('.pkl'))

    def load_new_base_data_with_names(self, names, filter=True):
        if len(names) != len(self.directories):
            raise ValueError("The number of names must match the number of directories.")
        dir = os.getcwd()
        for directory, name in zip(self.directories, names):
            data_parser = DataParser([directory], io.DataCoordinator())
            data_parser.parse_directories(output=True, filter=filter)
            frame = data_parser.get_data()
            print(f'Loaded data from {directory} with {len(frame)} entries.')
            # Use provided name as the filename
            filename = name

            # Check for banned names and duplicate files
            path_parts = directory.split('/')
            index = -1
            while filename in self.banned_name_list or \
                (self.pickle_directory + filename + '.pkl') in self._get_existing_pickle_files():
                index -= 1
                filename = path_parts[index] if -index <= len(path_parts) else f"unnamed_{abs(index)}"

            # Export to pickle with the resolved filename
            data_parser.export_to_pickle(self.pickle_directory + filename + '.pkl')
        os.chdir(dir)
        self.load_specific_existing_base_data( names)
        os.chdir(dir)
    def load_new_base_data_with_names_with_subsampling(self, names, min_diff=1e-3, max_samples=1000):
        if len(names) != len(self.directories):
            raise ValueError("The number of names must match the number of directories.")
        dir = os.getcwd()
        for directory, name in zip(self.directories, names):
            data_parser = DataParser([directory], io.DataCoordinator())
            data_parser.parse_directories_with_subsampling(output=True, min_energy=min_diff, max_samples=max_samples)
            frame = data_parser.get_data()
            print(f'Loaded data from {directory} with {len(frame)} entries.')
            # Use provided name as the filename
            filename = name

            # Check for banned names and duplicate files
            path_parts = directory.split('/')
            index = -1
            while filename in self.banned_name_list or \
                (self.pickle_directory + filename + '.pkl') in self._get_existing_pickle_files():
                index -= 1
                filename = path_parts[index] if -index <= len(path_parts) else f"unnamed_{abs(index)}"

            # Export to pickle with the resolved filename
            data_parser.export_to_pickle(self.pickle_directory + filename + '.pkl')
        os.chdir(dir)
        self.load_specific_existing_base_data( names)
        os.chdir(dir)
    def load_specific_existing_base_data(self, names):
        """
        Load specific existing data from the specified directory, ignoring files that match data types in data_ignore_list.
        And also set up the feature files, like in load_existing_base_data
        """
        data_parser = DataParser('', self.data_coordinator)
        data_files = {}
        local_data_labels = []
        for name in names:
            file = name + '.pkl'
            data_parser.load_from_pickle(self.pickle_directory + file)
            data_files[file] = data_parser.get_data()
            local_data_labels.append(file)
        self.base_data_labels = local_data_labels
        self.base_data_frames.update(data_files)
        for file in data_files:
            # Check and load corresponding feature files
            data_type = file.replace('.pkl', '')
            if data_type in self.data_ignore_list:
                continue
            feature_file = f'df_features_{data_type}.h5'
            feature_path = os.path.join(self.feature_directory, feature_file)
            if os.path.exists(feature_path):
                self.feature_paths[file] = feature_path
            else:
                print(f"Warning: No feature file found for {file}")
                
        
    def load_existing_base_data(self):
        """
        Load base data while ignoring data types in data_ignore_list.
        """
        self.base_data_labels = []
        self.base_data_frames = {}
        data_files, local_data_labels = self.load_existing_data(self.pickle_directory)
        self.base_data_labels += local_data_labels
        self.base_data_frames.update(data_files)
        for file in data_files:
            # Check and load corresponding feature files
            data_type = file.replace('.pkl', '')
            if data_type in self.data_ignore_list:
                continue
            feature_file = f'df_features_{data_type}.h5'
            feature_path = os.path.join(self.feature_directory, feature_file)
            if os.path.exists(feature_path):
                self.feature_paths[file] = feature_path
            else:
                print(f"Warning: No feature file found for {file}")

    def get_all_base_data(self):
        return self.base_data_frames
    def get_all_filtered_data(self):
        return self.filtered_data_frames
    def set_all_filtered_data(self, data):
        self.filtered_data_frames = data
        self.filtered_data_labels = data.keys()
    def get_all_filtered_labels(self):
        return self.filtered_data_labels
    def get_data(self, data_type):
        return self.filtered_data_frames[data_type]
    def get_base_data(self, data_type):
        return self.base_data_frames[data_type]
    def get_all_base_data(self):
        return self.base_data_frames, self.base_data_labels
    def get_feature_paths(self):
        return self.feature_paths
    def set_feature_paths(self, feature_paths):
        self.feature_paths = feature_paths
    
    def load_existing_features(self):
        """
        Load feature files, ignoring those corresponding to data types in data_ignore_list.
        """
        for file in os.listdir(self.feature_directory):
            if file.endswith('.h5'):
                data_type = file.replace('df_features_', '').replace('.h5', '')
                if data_type in self.data_ignore_list:
                    continue

                # Check for both the regular and filtered pickle file names
                pickle_file_name = f'{data_type}.pkl'
                filtered_pickle_file_name = f'{data_type}_filtered.pkl'

                if pickle_file_name in self.base_data_frames or filtered_pickle_file_name in self.base_data_frames:
                    self.feature_paths[pickle_file_name] = os.path.join(self.feature_directory, file)
                else:
                    print(f"Warning: Feature file {file} does not have a corresponding pickle file.")

    def generate_new_features(self):
        """
        Generate feature files for all data frames, excluding data types in data_ignore_list.
        """
        os.chdir(self.feature_directory)
        print("Generating Features for this base data: ")
        for label, data in self.base_data_frames.items():
            print(label)
        print("Starting feature generation...")
        for label, data in self.base_data_frames.items():
            data_type = label.replace('.pkl', '')
            if data_type in self.data_ignore_list:
                continue
            output_file = f'df_features_{data_type}.h5'
            print(f"Generating feature for {data_type}")
            featurizer = Featurizer(data, self.chemical_system, self.bspline_config)
            featurizer.featurize_data(self.num_cores, output_file, self.feature_batch_size)
        self.load_existing_features()
    def clear_filtered_data(self):
        UF3FileHandler(self.filter_directory).clear_directory()
        self.filtered_data_labels = []
        self.filtered_data_frames = {}
    def load_filtered_data(self):
        """
            Load filtered data from the filter directory. This data will actually be used for training/testing
        """
        self.filtered_data_labels = []
        self.filtered_data_frames = {}
        data_files, local_data_labels = self.load_existing_data(self.filter_directory)
        self.filtered_data_labels += local_data_labels
        self.filtered_data_frames.update(data_files)
    def check_structure_data(self, data):
        """
        Check for the presence of necessary data columns: eah and element fractions.
        
        Parameters:
        - data: DataFrame containing the structure data to check.
        
        Returns:
        - bool: True if all necessary data is present, False otherwise.
        """
        
        # Check for the existence of the 'eah' column
        if 'eah' not in data.columns:
            print('No eah data found')
            return False
        
        # Check for element fraction columns based on self.element_list
        missing_elements = []
        for element in self.element_list:
            if f'{element}_frac' not in data.columns:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"Missing fraction data for elements: {', '.join(missing_elements)}")
            return False
        
        return True
    def repair_structure_data(self, data):
        print("Repairing structure data...")
        sp = StructureParser(data)
        return sp.get_data()
    def generate_new_filtered_data(self, data_type, filter_type, check_existing=True, *args):
        """
        Generate new filtered data based on the data type and filter type provided.
        
        Parameters:
        - data_type: 'structures', 'temperature', 'pressure', etc.
        - filter_type: 'composition', 'energy_above_hull', 'maximum_force', etc.
        - args: Additional arguments for the specific filter.
        - check_existing: Boolean flag to check for existing filtered files before filtering.
        """
        filtered_filename = f'{data_type}_filtered.pkl'
        filtered_file_path = os.path.join(self.filter_directory,  filtered_filename)
        #check if on ignore list
        if data_type in self.data_ignore_list:
            return
        # Load the appropriate data
        if check_existing:
            #check if the filtered file is already in the filtered_data_frames
            if  filtered_filename in self.filtered_data_frames:
                data = self.filtered_data_frames[filtered_filename]
        else:
            data_filename = f'{data_type}.pkl'
            if data_filename not in self.base_data_frames:
                return
            data = self.base_data_frames[data_filename]
            # Check for the appropriate data columns
            if 'structure' in data_type:
                if not self.check_structure_data(data):
                    data = self.repair_structure_data(data)
                    self.replace_existing_data(data_filename, data)
                
        # Create a DataFilter instance
        data_filter = DataFilter(data)
        print(f"Filtering {data_type} data with {filter_type} filter.")
        print(f"Current data length: {len(data)}")
        
        # Find the appropriate filter method
        filter_method_name = self.FILTER_FUNCTION_LUT.get(filter_type)
        if not filter_method_name:
            print(f"Unsupported filter type: {filter_type}")
            return
        
        # Get the filter method from the DataFilter instance
        filter_method = getattr(data_filter, filter_method_name, None)
        if not filter_method:
            print(f"Filter method {filter_method_name} not found in DataFilter.")
            return
        
        # Apply the filter with the provided arguments
        filter_method(*args)
        
        filter_record = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'filter_type': filter_type,
            'params': args,
            'initial_length': len(data),
            'final_length': len(data_filter.get_data())
        }
        self.filter_history.append(filter_record)
        
        # Export the filtered data to a new pickle file in the filtered directory
        data_filter.export_to_pickle(filtered_file_path)
        
        print(f"Filtered data length: {len(data_filter.get_data())}")
        
        # Reload the filtered data
        self.load_filtered_data()
    def filter_all_data(self, filter_type, check_existing=True, *args):
        """
        Apply a filter to all data types we have.

        Parameters:
        - filter_type: The filter to apply (e.g., 'composition', 'energy_above_hull')
        - check_existing: If True, apply the filter to already-filtered data if it exists
        - args: Arguments to pass to the filter method
        """
        for data_filename, data in self.base_data_frames.items():
            data_type = data_filename.replace('.pkl', '')

            if data_type in self.data_ignore_list:
                continue

            filtered_filename = f'{data_type}_filtered.pkl'
            filtered_file_path = os.path.join(self.filter_directory, filtered_filename)

            if check_existing:
                if filtered_filename in self.filtered_data_frames:
                    data = self.filtered_data_frames[filtered_filename]

            else:
                if data_filename not in self.base_data_frames:
                    continue
                data = self.base_data_frames[data_filename]
                if data_type == 'structure':
                    if not self.check_structure_data(data):
                        data = self.repair_structure_data(data)
                        self.replace_existing_data(data_filename, data)

            data_filter = DataFilter(data)
            print(f"Filtering {data_type} data with {filter_type} filter.")
            print(f"Current data length: {len(data)}")

            filter_method_name = self.FILTER_FUNCTION_LUT.get(filter_type)
            if not filter_method_name:
                print(f"Unsupported filter type: {filter_type}")
                continue

            filter_method = getattr(data_filter, filter_method_name, None)
            if not filter_method:
                print(f"Filter method {filter_method_name} not found in DataFilter.")
                continue

            filter_method(*args)
            
            filter_record = {
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'filter_type': filter_type,
                'params': args,
                'initial_length': len(data),
                'final_length': len(data_filter.get_data())
            }
            self.filter_history.append(filter_record)

            data_filter.export_to_pickle(filtered_file_path)

            print(f"Filtered data length: {len(data_filter.get_data())}")

            self.load_filtered_data()

    def get_filter_history(self):
        return self.filter_history
    def clear_filter_history(self):
        self.filter_history = []
    def set_filter_history(self, history):
        self.filter_history = history
    def copy_unfiltered_data(self):
        """
        Copy unfiltered data to the filtered directory, excluding data types in data_ignore_list.
        """
        for label, data in self.base_data_frames.items():
            data_type = label.replace('.pkl', '')
            if data_type in self.data_ignore_list:
                continue
            filtered_label = label.replace('.pkl', '_filtered.pkl')
            if filtered_label not in self.filtered_data_labels:
                data_parser = DataParser('', self.data_coordinator)
                data_parser.set_data(data)
                data_parser.export_to_pickle(os.path.join(self.filter_directory, filtered_label))
    def clear_memory(self):
        """
        Clear all data from memory.
        """
        self.base_data_labels = []
        self.base_data_frames = {}
        self.filtered_data_frames = {}
        self.filtered_data_labels = []
    def clear_ignore_list(self):
        self.data_ignore_list = []
    def add_to_ignore_list(self, data_label):
        self.data_ignore_list.append(data_label)
    def get_ignore_list(self):
        return self.data_ignore_list
    def set_ignore_list(self, ignore_list):
        self.data_ignore_list = ignore_list
    def load_current_base_data_as_filtered(self):
        """fill the filtered data arrays with the current base data"""
        self.filtered_data_frames = {}
        self.filtered_data_labels = []
        for label, data in self.base_data_frames.items():
            data_type = label.replace('.pkl', '')
            if data_type in self.data_ignore_list:
                continue
            
            
            self.filtered_data_frames[label] = data
            self.filtered_data_labels.append(label)
    def consolidate_data_return(self):
        """
        Consolidate all data into a single DataFrame for training/testing,
        skipping types in data_ignore_list. Removes exact duplicates by index.
        Raises error if same index is used with conflicting content.
        """
        consolidated_data = pd.DataFrame()
        for label, data in self.filtered_data_frames.items():
            data_type = label.replace('.pkl', '').replace('_filtered', '').replace('_rebalanced', '')
            if data_type in self.data_ignore_list:
                continue
            data = data.copy()
            data["source_dataset"] = data_type
            print(f"adding {data_type} data with length {len(data)}")
            consolidated_data = pd.concat([consolidated_data, data], ignore_index=False)

        # check for duplicate indices
        #duplicated_indices = consolidated_data.index[consolidated_data.index.duplicated()].unique()
            # drop exact duplicates
        consolidated_data = consolidated_data[~consolidated_data.index.duplicated(keep='first')]

        print(f"Consolidated data length: {len(consolidated_data)}")
        return consolidated_data

    def consolidate_data(self):
        """
        Consolidate all data into a single DataFrame for training/testing, ignoring data types in data_ignore_list.
        """
        self.model_data = pd.DataFrame()
        self.model_data = self.consolidate_data_return()
    def get_unique_indices(self, data):
        """
        Get the unique indices in the DataFrame.

        :return: List of unique indices.
        """
        return data.index.to_series().apply(lambda x: x.lower().split('_run')[0]).unique()
    def save_split_indices(self, train_idx, test_idx, prefix=None):
        os.makedirs(self.filter_directory, exist_ok=True)
        tag = f"{prefix}_" if prefix else ""
        train_path = os.path.join(self.filter_directory, f"{tag}train_indices.txt")
        test_path = os.path.join(self.filter_directory, f"{tag}test_indices.txt")

        train_list = list(map(str, train_idx))
        test_list = list(map(str, test_idx))

        with open(train_path, "w") as f:
            f.write("\n".join(train_list))
        with open(test_path, "w") as f:
            f.write("\n".join(test_list))

    def split_data(self, test_percent=0.2, filter_index=True, unique_indices=True, exclude_list=None, random_seed=None, include_in_training=False):
        # bail out early if there's nothing to split
        if self.model_data.empty:
            raise ValueError("Model data is empty. Please consolidate data before splitting.")

        # carve out datasets we never want in test
        if exclude_list and "source_dataset" in self.model_data.columns:
            excluded_data = self.model_data[self.model_data["source_dataset"].isin(exclude_list)]
            included_data = self.model_data[~self.model_data["source_dataset"].isin(exclude_list)]
            print(f"Excluded from test split: {exclude_list} — {len(excluded_data)} rows")
        else:
            excluded_data = pd.DataFrame(columns=self.model_data.columns)
            included_data = self.model_data

        # unique index grouping by prefix before "_run"
        if unique_indices:
            index_prefixes = included_data.index.to_series().apply(lambda x: x.lower().split('_run')[0]).unique()
        else:
            index_prefixes = included_data.index.to_series()

        if random_seed is None:
            random_seed = daily_millis()
        np.random.seed(random_seed)
        num_indices_to_select = int(test_percent * len(index_prefixes))
        selected_prefixes = np.random.choice(index_prefixes, num_indices_to_select, replace=False)

        if filter_index:
            prefix_series = included_data.index.to_series().apply(lambda x: x.lower().split('_run')[0])
            test_mask = prefix_series.isin(selected_prefixes)
            test_split = included_data[test_mask]
            train_split = included_data[~test_mask]
        else:
            shuffled = included_data.sample(frac=1, random_state=42)
            split_point = int(len(shuffled) * (1 - test_percent))
            train_split = shuffled.iloc[:split_point]
            test_split = shuffled.iloc[split_point:]

        # assemble final sets
        if include_in_training:
            # training includes train + test + always-include bucket
            self.training_set = pd.concat([train_split, test_split, excluded_data], axis=0)
        else:
            # default: training excludes test, but includes the excluded bucket
            self.training_set = pd.concat([train_split, excluded_data], axis=0)

        self.testing_set = test_split

        self.train_keys = self.training_set.index
        self.test_keys = self.testing_set.index

        if include_in_training:
            print(f"Training set length: {len(self.training_set)} (includes test). Testing set length: {len(self.testing_set)}")
        else:
            print(f"Training set length: {len(self.training_set)}. Testing set length: {len(self.testing_set)}")
        self.save_split_indices(self.train_keys, self.test_keys)
        return self.training_set, self.testing_set


    def select_testing_set(self, test_dataset_names, include_in_training=False, test_split_fraction=None, seed=None):
        """
        Set specific datasets (by name) as the testing set. Optionally include them in training.

        Parameters:
        - test_dataset_names (list[str]): Dataset names like 'aln_polytypes'
        - include_in_training (bool): If True, also include testing data in the training set
        - test_split_fraction (float or None): If given, randomly split each dataset with this fraction for testing
        - seed (int): Random seed for reproducibility
        """
        if seed is None:
            seed = daily_millis()
        np.random.seed(seed)
        all_test_indices = set()
        all_train_indices = set()

        for name in test_dataset_names:
            label = f"{name}_filtered.pkl"
            if label not in self.filtered_data_frames:
                raise ValueError(f"Dataset {label} not found in filtered_data_frames.")

            df = self.filtered_data_frames[label]

            if test_split_fraction is not None:
                # Random split
                indices = df.index.to_numpy()
                np.random.shuffle(indices)
                split_idx = int(len(indices) * (1 - test_split_fraction))
                train_idx = indices[:split_idx]
                test_idx = indices[split_idx:]
                all_train_indices.update(train_idx)
                all_test_indices.update(test_idx)
            else:
                # Entire dataset goes to test set
                all_test_indices.update(df.index)

        self.testing_set = self.model_data[self.model_data.index.isin(all_test_indices)]
        self.test_keys = self.testing_set.index

        if include_in_training:
            self.training_set = self.model_data
        else:
            if test_split_fraction is not None:
                combined_train_indices = set(self.model_data.index) - (all_test_indices - all_train_indices)
            else:
                combined_train_indices = set(self.model_data.index) - all_test_indices
            self.training_set = self.model_data[self.model_data.index.isin(combined_train_indices)]

        self.train_keys = self.training_set.index
        print(f"Testing set size: {len(self.testing_set)}, Training set size: {len(self.training_set)}")
        self.save_split_indices(self.train_keys, self.test_keys)
        return self.training_set, self.testing_set
    def select_testing_sets(self, dataset_names, test_split_fraction=0.1, seed=None):
        if seed is None:
            seed = daily_millis()
        np.random.seed(seed)
        all_test_indices = set()
        individual_test_splits = {}

        for name in dataset_names:
            label = f"{name}_filtered.pkl"
            if label not in self.filtered_data_frames:
                raise ValueError(f"Dataset {label} not found in filtered_data_frames.")

            df = self.filtered_data_frames[label]
            indices = df.index.to_numpy()
            np.random.shuffle(indices)
            split_idx = int(len(indices) * (1 - test_split_fraction))
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]

            all_test_indices.update(test_idx)
            individual_test_splits[name] = df.loc[test_idx]

        # All data in model_data is training except what's in test set
        self.testing_set = self.model_data[self.model_data.index.isin(all_test_indices)]
        self.training_set = self.model_data[~self.model_data.index.isin(all_test_indices)]

        self.train_keys = self.training_set.index
        self.test_keys = self.testing_set.index

        print(f"Training set size: {len(self.training_set)}. Testing set size: {len(self.testing_set)}")
        self.save_split_indices(self.train_keys, self.test_keys)
        return self.training_set, self.testing_set, individual_test_splits
    def set_training_data(self, data):
        self.training_set = data
        self.train_keys = data.index
    def set_testing_data(self, data):
        self.testing_set = data 
        self.test_keys = data.index 
    def get_training_data(self):
        return self.training_set, self.train_keys 
    def get_testing_data(self):
        return self.testing_set, self.test_keys
    def fill_sample_weights(self, sample_weights):
        sample_weights_final = {}

        # Ensure sample_weights is a dictionary, or default to an empty dictionary
        if sample_weights is None:
            sample_weights = {}

        for key in self.filtered_data_frames.keys():  # Iterate based on existing filtered data
            key_original = key.replace('_filtered.pkl', '')  # Get original key name
            weight = sample_weights.get(key_original, 1.0)  # Default to 1.0 if missing
            
            try:
                # Check if the key exists in filtered_data_frames
                if key not in self.filtered_data_frames:
                    print(f"Warning: Key '{key}' not found in filtered_data_frames. Skipping...")
                    continue
                
                indices = self.filtered_data_frames[key].index  # Extract indices
                
                # Assign weight to each index
                for index in indices:
                    sample_weights_final[index] = weight

            except Exception as e:
                print(f"Error processing key '{key}': {e}")

        return sample_weights_final


       
    def train_model(self, model_filename, training_data, testing_data, regularizer, weight=0.3, sample_weights=None, batch_size=500):
        """
        Train a model using the given data.
        
        Parameters:
        - regularizer: Regularization parameter for the model.
        - weight: Class weight for the model.
        - batch_size: Batch size for training.
        """
        #get the actual data from self.feature_paths
        features = []
        sample_weights_final = self.fill_sample_weights(sample_weights)
        for key in self.feature_paths:
            #remove .pkl from the key
            data_type = key.replace('.pkl', '')
            if(data_type in self.data_ignore_list):
                continue
            features.append(self.feature_paths[key])

        trainer = ModelTrainer(self.model_data, self.chemical_system, self.bspline_config, regularizer)
        trainer.set_training_set(training_data)
        trainer.set_testing_set(testing_data)
        
        
        if(len(features) == 0):
            print("No features found")
            return
        if(len(features) == 1):
            trainer.train_model(features[0], weight, batch_size, sample_weights=sample_weights_final if sample_weights is not None else None)
        else:
            trainer.train_model_multiple_files(features, weight, batch_size, sample_weights=sample_weights_final if sample_weights is not None else None)
        trainer.export_model(model_filename + ".json")
        #analyzer = ModelAnalyzer(self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), training_data, structure, structure_feature)
        return self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), features, model_filename
    def train_model_parallel(self, model_filename, training_data, testing_data, regularizer, weight=0.3, batch_size=500, sample_weights=None, num_cores=1):
        """
        Train a model using the given data.
        
        Parameters:
        - regularizer: Regularization parameter for the model.
        - weight: Class weight for the model.
        - batch_size: Batch size for training.
        """
        #get the actual data from self.feature_paths
        features = []
        sample_weights_final = self.fill_sample_weights(sample_weights)
        for key in self.feature_paths:
            #remove .pkl from the key
            data_type = key.replace('.pkl', '')
            if(data_type in self.data_ignore_list):
                continue
            features.append(self.feature_paths[key])
        trainer = ModelTrainer(self.model_data, self.chemical_system, self.bspline_config, regularizer)
        trainer.set_training_set(training_data)
        trainer.set_testing_set(testing_data)
        if(len(features) == 1):
            trainer.train_model_parallel(features[0], weight=weight, batch_size=batch_size, num_cores=num_cores, sample_weights=sample_weights_final if sample_weights is not None else None)
        else:
            trainer.train_model_multiple_files_parallel(features, weight=weight, batch_size=batch_size, num_cores=num_cores,sample_weights=sample_weights_final if sample_weights is not None else None)
        trainer.export_model(model_filename)
        #analyzer = ModelAnalyzer(self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), training_data, structure, structure_feature)
        return self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), features, model_filename
    def get_elements_in_feature_data(self, feature_data):
        columns = feature_data.columns
        elements = set()
        for col in columns:
            if col.startswith('n_'):
                element = col.split('_')[1]
                elements.add(element)
        
    def load_fitting_data(self, training_data, testing_data, batch_size=500, num_cores=1, use_cur=False, exclude_elements=None):
        features = []
        for key in self.feature_paths:
            #remove .pkl from the key
            data_type = key.replace('.pkl', '')
            if(data_type in self.data_ignore_list):
                continue
            features.append(self.feature_paths[key])
        regularizer =  dict(ridge_1b=1e-6,
                ridge_2b=1e-4,
                ridge_3b=1e-4,
                curvature_2b=1e-4,
                curvature_3b=1e-4)
        trainer = ModelTrainer(self.model_data, self.chemical_system, self.bspline_config, regularizer)
        trainer.set_training_set(training_data)
        trainer.set_testing_set(testing_data)
        self.fitting_df = trainer.load_fitting_data(features, num_cores=num_cores, use_cur=use_cur)
        # drop_columns = []
        # if exclude_elements is not None:
        #     drop_columns = [
        #         col for col in self.fitting_df.columns
        #         if any(el.lower() in col.lower() for el in exclude_elements)
        #     ]
        # self.fitting_df.drop(columns=drop_columns, inplace=True)
        self.fitting_features = features
        return self.chemical_system, self.fitting_df, trainer.get_bspline_config(), features
    def train_model_from_df(self, model_filename, training_data, testing_data, regularizer, weight=0.3, batch_size=500, num_cores=1, sample_weights=None, exclude_elements=None, outlier_config=None):
        sample_weights_final = self.fill_sample_weights(sample_weights)
        trainer = ModelTrainer(self.model_data, self.chemical_system, self.bspline_config, regularizer)
        trainer.set_training_set(training_data)
        trainer.set_testing_set(testing_data)
        trainer.fit_from_dataframe(self.fitting_df, weight=weight, batch_size=batch_size, num_cores=num_cores,sample_weights=sample_weights_final if sample_weights is not None else None, exclude_elements=exclude_elements, outlier_config=outlier_config)
        trainer.export_model(model_filename)
        return self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), None, model_filename, trainer
    def fine_tune_model_from_dataframe_blend(self, trainer, model_filename, training_data, testing_data, regularizer, weight=0.3, batch_size=500, num_cores=1, sample_weights=None, exclude_elements=None, alpha=0.05):
        sample_weights_final = self.fill_sample_weights(sample_weights)
        trainer.set_training_set(training_data)
        trainer.set_testing_set(testing_data)
        trainer.fine_tune_from_dataframe_blend(self.fitting_df, alpha=alpha, weight=weight, batch_size=batch_size, num_cores=num_cores,sample_weights=sample_weights_final if sample_weights is not None else None, exclude_elements=exclude_elements)
        trainer.export_model(model_filename)
        return self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), None, model_filename, trainer
    def train_model_multistep(self, model_filename, training_data, testing_data, regularizer, weight=0.3, batch_size=500):
        """
        Train a model using the given data.
        
        Parameters:
        - regularizer: Regularization parameter for the model.
        - weight: Class weight for the model.
        - batch_size: Batch size for training.
        """
        #get the actual data from self.feature_paths
        features = []
        for key in self.feature_paths:
            #remove .pkl from the key
            data_type = key.replace('.pkl', '')
            if(data_type in self.data_ignore_list):
                continue
            features.append(self.feature_paths[key])
        #find the structures feature file
        for feature in features:
            if 'structure' in feature:
                structure_feature = feature
                break
        if(self.trainer is None):
            self.trainer = ModelTrainerMultipleSteps(self.model_data, self.chemical_system, self.bspline_config, regularizer)
        self.trainer.set_training_set(training_data)
        self.trainer.set_testing_set(testing_data)
        if(len(features) == 0):
            print("No features found")
            return
        if(len(features) == 1):
            self.trainer.train_model(features[0], weight, batch_size)
        else:
            print("----------------ERROR NO MULTIPLE FILFES YET")
        
        #analyzer = ModelAnalyzer(self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), training_data, structure, structure_feature)
        return self.chemical_system, self.trainer.get_model(), self.trainer.get_bspline_config(), features, model_filename
    def train_model_multistep_finalize(self, model_filename, training_data, testing_data, regularizer, weight=0.3, batch_size=500):
        """
        Train a model using the given data.
        
        Parameters:
        - regularizer: Regularization parameter for the model.
        - weight: Class weight for the model.
        - batch_size: Batch size for training.
        """
        self.trainer.finalize_fit(weight=weight)
        self.trainer.export_model(model_filename + ".json")
        #analyzer = ModelAnalyzer(self.chemical_system, trainer.get_model(), trainer.get_bspline_config(), training_data, structure, structure_feature)
        return self.chemical_system, self.trainer.get_model(), self.trainer.get_bspline_config(), None, model_filename
    def import_model(self, regularizer, model_filename):
        """
        Import a pre-trained model from a file.

        Parameters:
        - model_filename: Path to the model file to import.
        
        Returns:
        - Imported model and its associated chemical system.
        """
        # Ensure the model file path is provided
        if not model_filename:
            print("Model file path is required.")
            return
        features = []
        for key in self.feature_paths:
            #remove .pkl from the key
            data_type = key.replace('.pkl', '')
            if(data_type in self.data_ignore_list):
                continue
            features.append(self.feature_paths[key])
        #find the structures feature file
        for feature in features:
            if 'structure' in feature:
                structure_feature = feature
                break
        
        # Use the ModelTrainer to import the model
        trainer = ModelTrainer(self.model_data, self.chemical_system, self.bspline_config, regularizer)
        trainer.import_model(model_filename)
        self.set_trainer(trainer)
        # Retrieve the imported model
        imported_model = trainer.get_model()
        
        # Optionally, you could perform additional checks or setup here
        print(f"Model imported successfully from {model_filename}.")
        
        # Return the chemical system and the imported model
        return self.chemical_system, imported_model, trainer.get_bspline_config(), features
    def store_validation_data(self, model_filename, validation_data):
        print("writing validation data to file")

        potential_dir = os.path.abspath('potentials')
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)

        validation_file = os.path.join(potential_dir, f"{model_filename}_validation_data.txt")
        with open(validation_file, 'w') as f:
            if isinstance(validation_data, list):
                for data in validation_data:
                    f.write(str(data))
                    f.write('\n')
            else:
                f.write(str(validation_data))
                f.write('\n')

        if isinstance(validation_data, list):
            for data in validation_data:
                data.save_raw_data_to_file(potential_dir)
        else:
            validation_data.save_raw_data_to_file(potential_dir)

        self.validation_datas.append(validation_data)

        all_data_file = os.path.abspath('validation_datas.txt')
        with open(all_data_file, 'w') as f:
            for data_group in self.validation_datas:
                for data in data_group:
                    f.write(str(data))
                    f.write('\n')
    def store_validation_data_local(self, model_filename, validation_data):
        """
        Store the validation data for a given model.
        """
        #store the validation data in potentials folder
        with open(f'{model_filename}_validation_data.txt', 'w') as f:
                    f.write(str(validation_data) + '\n')
                    f.flush()
        
    def store_validation_data_all(self, model_filename, validation_data):
        self.validation_datas_all.append(validation_data)
        
        #write the entire validation datas to a file
        with open('validation_datas_all.txt', 'w') as f:
            for data2 in self.validation_datas_all:
                for data in data2:
                    f.write(str(data) + '\n')
                    f.flush()
    def store_validation_data_alternate_dir(self, model_filename, validation_data, base_folder):
        print("writing validation data to file")
        abs_base = os.path.abspath(base_folder)
        potential_dir = os.path.join(abs_base, 'validation_datas')
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)

        # Write the primary percent-error data
        validation_file = os.path.join(potential_dir, f"{model_filename}_validation_data.txt")
        with open(validation_file, 'w') as f:
            if isinstance(validation_data, list):
                for data in validation_data:
                    f.write(str(data))
                    f.write('\n')
            else:
                f.write(str(validation_data))
                f.write('\n')

        # Save raw data for each ValidationData object to a separate file
        if isinstance(validation_data, list):
            for data in validation_data:
                data.save_raw_data_to_file(potential_dir)
        else:
            validation_data.save_raw_data_to_file(potential_dir)

        # If you keep a master list, you can write them all below
        self.validation_datas.append(validation_data)
        all_data_file = os.path.join(abs_base, 'validation_datas.txt')
        with open(all_data_file, 'w') as f:
            for data_group in self.validation_datas:
                for data in data_group:
                    f.write(str(data))
                    f.write('\n')
    def store_validation_data_alternate_dir_standalone(model_filename, validation_data, base_folder):
        print("writing validation data to file")
        abs_base = os.path.abspath(base_folder)
        validation_datas = []
        potential_dir = os.path.join(abs_base, 'validation_datas')
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)

        # Write the primary percent-error data
        validation_file = os.path.join(potential_dir, f"{model_filename}_validation_data.txt")
        with open(validation_file, 'w') as f:
            if isinstance(validation_data, list):
                for data in validation_data:
                    f.write(str(data))
                    f.write('\n')
            else:
                f.write(str(validation_data))
                f.write('\n')

        # Save raw data for each ValidationData object to a separate file
        if isinstance(validation_data, list):
            for data in validation_data:
                data.save_raw_data_to_file(potential_dir)
        else:
            validation_data.save_raw_data_to_file(potential_dir)

        # If you keep a master list, you can write them all below
        validation_datas.append(validation_data)
        all_data_file = os.path.join(abs_base, 'validation_datas.txt')
        with open(all_data_file, 'w') as f:
            for data_group in validation_datas:
                f.write(str(data_group))
                f.write('\n')
            
    def get_all_dataframes(self):
        """
        Retrieve all the DataFrames used in training the model, including both base and filtered data frames.
        
        Returns:
            dict: A dictionary containing base data frames and filtered data frames.
        """
        # Ensure data is loaded
        if not self.base_data_frames:
            self.load_existing_base_data()
        if not self.filtered_data_frames:
            self.load_filtered_data()
        
        # Combine both base and filtered data into a single dictionary
        all_data_frames = {
            "base_data_frames": self.base_data_frames,
            "filtered_data_frames": self.filtered_data_frames
        }
        
        return all_data_frames
 
    def set_dataframe(self, dataframe):
        """
        Accept a dataframe directly and assign it to the model_data attribute.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # Assign the provided dataframe to the model_data attribute
        self.model_data = dataframe
        print(f"Dataframe with {len(dataframe)} rows and {len(dataframe.columns)} columns set successfully.")
        
    def extract_index_pattern(self, df, pattern_key):
        """
        Extract and store the index pattern for a given DataFrame.
        
        Parameters:
        - df: The DataFrame from which to extract the index pattern.
        - pattern_key: A unique key to associate with the extracted pattern (e.g., the dataset name).
        """
        if not df.index.empty:
            # Extract a naming pattern based on the first index value
            sample_index = str(df.index[0])
            
            # Here, we assume the pattern is the prefix before '_run' or any specific delimiter
            if '_run' in sample_index:
                pattern = sample_index.split('_run')[0] + '_run'
            else:
                pattern = sample_index
            
            # Store the extracted pattern
            self.index_patterns[pattern_key] = pattern
            print(f"Extracted index pattern for {pattern_key}: {pattern}")
        else:
            print(f"DataFrame {pattern_key} has an empty index. No pattern extracted.")

    def apply_index_pattern(self, df, pattern_key):
        """
        Apply the stored index pattern when reassigning indices to a DataFrame.
        
        Parameters:
        - df: The DataFrame to which the index pattern will be applied.
        - pattern_key: The key for the pattern stored in self.index_patterns.
        
        Returns:
        - DataFrame with reassigned indices based on the stored pattern.
        """
        if pattern_key in self.index_patterns:
            pattern = self.index_patterns[pattern_key]
            new_indices = [f"{pattern}_{i}" for i in range(len(df))]
            df.index = new_indices
            print(f"Applied index pattern {pattern_key} to DataFrame.")
        else:
            print(f"No pattern found for {pattern_key}. Index not changed.")
        
        return df
    def export_rebalanced_data(self, rebalanced_data_dict):
        """
        Save each rebalanced DataFrame in rebalanced_data_dict to a new .pkl file with '_rebalanced' suffix.
        
        Parameters:
        - rebalanced_data_dict: Dictionary of rebalanced dataframes (dataset name -> dataframe).
        """
        for original_filename, df in rebalanced_data_dict.items():
            # Generate the new filename with '_rebalanced' suffix
            rebalanced_filename = original_filename.replace('_filtered', '_rebalanced')
            
            # Use replace_existing_data to save the rebalanced DataFrame
            self.replace_existing_data(rebalanced_filename, df)
            
            print(f"Saved rebalanced data for {original_filename} as {rebalanced_filename}") 
    def set_parity_data(self, parity_frame, parity_feature):
        """
        Set the parity data for the model.
        
        Parameters:
        - parity_data: DataFrame containing the parity data.
        """
        self.parity_frame = parity_frame
        self.parity_feature = parity_feature
    def get_parity_data(self):
        #return the parity data FRAMES
        if self.parity_frame is None:
            print("ERROR: No parity data set")
            return None, None
        return self.get_data(self.parity_frame), self.parity_feature
    def set_lammps_validation_models(self, models):
        """
        Set the LAMMPS validation models for the current potential when doing grid search. otherwise, defaults to materialsproject
        
        Parameters:
        - models: List of LAMMPS validation models.
        """
        self.validation_lammps_models = models
    def set_grid_search_cutoffs(self, cutoffs):
        """
        Set the cutoff values for grid search.
        
        Parameters:
        - cutoffs: Dictionary of cutoff values for each validation result 
        """
        self.cutoffs = cutoffs
    def check_validation_cutoff(self, validation_data, cutoff_name):
        if(self.cutoffs is None):
            print("No cutoffs set")
            return False
        output = True
        for key in validation_data:
            if cutoff_name in self.cutoffs:
                tester = key.getValidation(cutoff_name)
                if(tester is not None):
                    if(isinstance(tester, list)):
                        tester = tester[0]
                    for cutoff in self.cutoffs[cutoff_name]:
                        result = tester[cutoff]
                        threshold = self.cutoffs[cutoff_name][cutoff]
                        if(result > threshold):
                            output = False
        return output
        
    def check_validation_cutoffs(self, validation_data):
        """
        Check the validation data against the cutoff values.
        
        Parameters:
        - validation_data: Dictionary of validation results.
        
        Returns:
        - bool: True if the validation data meets the cutoffs, False otherwise.
        """
        if(self.cutoffs is None):
            print("No cutoffs set")
            return False
        output = True
        for key in validation_data:
            for cutoff_type in self.cutoffs:
                tester = key.getValidation(cutoff_type)
                if(tester is not None):
                    if(isinstance(tester, list)):
                        tester = tester[0]
                    for cutoff in self.cutoffs[cutoff_type]:
                        result = tester[cutoff]
                        threshold = self.cutoffs[cutoff_type][cutoff]
                        if(result > threshold):
                            output = False
        return output
    def accept_model(self, model_filename):
        """
        Accept a model and move it to the valid_potentials folder.
        """
        #make sure model_filename exists
        if not os.path.exists(f"{model_filename}.json"):
            print(f"Model {model_filename} not found.")
            return
        os.rename(f"{model_filename}.json", f"{self.valid_potentials_folder}/{model_filename}.json")
        #take the validation data and store it in the valid_potentials folder
        os.rename(f"potentials/{model_filename}_validation_data.txt", f"{self.valid_potentials_folder}/{model_filename}_validation_data.txt")  
    def setup_valid_potential_folder(self):
        """
        Setup the valid_potentials folder for storing accepted models. Creates
        a subfolder in the valid_potentials folder with the current date and time.
        """
        os.makedirs("valid_potentials", exist_ok=True)
        folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"valid_potentials/{folder_name}", exist_ok=False)
        self.valid_potentials_folder = f"valid_potentials/{folder_name}"
    def run_cutoff_grid_search_quick_test(self, composition_element, composition_range, energy_range, force_range, min_force_range, regularizer_ranges):
        self.setup_valid_potential_folder()
        elastic = {"bulk_modulus_voigt":200.0, "shear_modulus_voigt":200.0}
        lattice_constant = {"a":1.0, "b":1.0, "c":1.0}
        surface_energy = {"bulk_energy":50.0}
        normalized_error = {"normalized_error":0.20}
        self.cutoffs = {"elastic_tensor":elastic, 
                        "lattice_constant":lattice_constant, 
                        "surface_energy":surface_energy}
        model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        #structure_downloader = StructureDownloader(MP_API_KEY)
        materials = structure_downloader.get_structure_list('AlN')
                                            
        # Set the current regularizer values
        regularizer =  dict(ridge_1b=1e-08,
                ridge_2b=1e-8,
                ridge_3b=1e-8,
                curvature_2b=1e-8,
                curvature_3b=1e-7)
        model_path = "test_model.json"
        chemical_system, trainer_model,bspline_config, features = self.import_model(regularizer, model_path)
        if(self.parity_frame is None):
            frame = self.get_data('structure_filtered.pkl')
            for feature in features:
                if 'structure' in feature:
                    structure_feature = feature
                    break
        else:
            frame = self.parity_frame
            structure_feature = self.parity_feature
        analyzer = ModelAnalyzer(chemical_system, trainer_model, bspline_config, frame, frame, structure_feature)                                      
        # Export model to LAMMPS format
        model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        #structure_downloader = StructureDownloader(MP_API_KEY)
        #only test mp-661
        materials = [materials[0]]
        analyzer.set_dft_validation_structures(materials)
        analyzer.set_lammps_validation_structures(materials)
        analyzer.export_model_to_lammps("potentials/", model_filename, "ntaormina", "metal", "nk")
        analyzer.export_model_to_lammps(os.getcwd(), model_filename, "ntaormina", "metal", "nk")
        # Validate the model
        validation_data = analyzer.validate_potential(model_filename, model_filename, ['Al', 'N'])
        print(f"Validation data: {validation_data}")
        # check that the validation data meets the cutoffs before saving
        if(self.check_validation_cutoffs(validation_data)):
            self.store_validation_data(model_filename, validation_data)
        
        # Remove potential file from current working directory
        os.remove(f"{model_filename}")
    def run_cutoff_grid_search(self, elements, chemical_formula, composition_element, composition_range, energy_range, force_range, min_force_range, regularizer_ranges, weight_range, num_cores=1):
        base_directory = os.getcwd()
        self.setup_valid_potential_folder()
        elastic = {"bulk_modulus_voigt":40.0, "shear_modulus_voigt":40.0}
        lattice_constant = {"a":4.0, "b":4.0, "c":4.0}
        surface_energy = {"bulk_energy":50.0}
        normalized_error = {"normalized_error":0.20}
        cohesive_energy = {"cohesive_energy":40.0}
        self.cutoffs = {"elastic_tensor":elastic, 
                        "lattice_constant":lattice_constant, 
                        "surface_energy":surface_energy}
        tests = ['lattice_constant', 'cohesive_energy', 'elastic_tensor', 'surface_energy']
        model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        #structure_downloader = StructureDownloader(MP_API_KEY)
        materials = structure_downloader.get_structure_list(chemical_formula)
       
        
        for composition in composition_range:
            for energy in energy_range:
                for force in force_range:
                    for min_force in min_force_range: #after this, regularizers will be tested
                        self.clear_memory()
                        self.load_existing_base_data()
                        self.load_existing_features()
                        
                        # Generate filtered data
                        self.generate_new_filtered_data('al_structures_combined', 'composition', False, 'N', 0.70)
                        self.generate_new_filtered_data('al_structures_combined', 'energy_above_hull', True, 2.0)
                        self.generate_new_filtered_data('al_structures_combined', 'maximum_force', True, 18, 18, 18)
                        self.generate_new_filtered_data('al_structures_combined', 'zero_force', True)
                        self.copy_unfiltered_data()
                        self.load_filtered_data()
                        self.consolidate_data()
                        training, testing = self.split_data(0.10)
                        for weight in weight_range:
                            for ridge_1b in regularizer_ranges['ridge_1b']: #start regularizer test (don't need to reload data)
                                for ridge_2b in regularizer_ranges['ridge_2b']:
                                    for ridge_3b in regularizer_ranges['ridge_3b']:
                                        for curvature_2b in regularizer_ranges['curvature_2b']:
                                            for curvature_3b in regularizer_ranges['curvature_3b']:
                                                print(f"Running grid search for"
                                                    f"weight={weight}, ridge_1b={ridge_1b}, ridge_2b={ridge_2b}, ridge_3b={ridge_3b}, "
                                                    f"curvature_2b={curvature_2b}, curvature_3b={curvature_3b}")
                                                
                                                model_filename = (f'weight_{weight}_ridge1b_{ridge_1b}_ridge2b_{ridge_2b}_ridge3b_{ridge_3b}_'
                                                                f'curv2b_{curvature_2b}_curv3b_{curvature_3b}')
                                                
                                                # Set the current regularizer values
                                                regularizer = dict(ridge_1b=ridge_1b,
                                                                ridge_2b=ridge_2b,
                                                                ridge_3b=ridge_3b,
                                                                curvature_2b=curvature_2b,
                                                                curvature_3b=curvature_3b)
                                                if (num_cores == 1):
                                                    chemical_system, trainer_model,bspline_config, features, NULL = self.train_model(model_filename + ".json", training, testing, regularizer=regularizer, weight=weight)
                                                else:
                                                    chemical_system, trainer_model,bspline_config, features, NULL = self.train_model_parallel(model_filename + ".json", training, testing, regularizer=regularizer, weight=weight, num_cores=num_cores)
                                                # if(self.parity_frame is None):
                                                #     if(self.filtered_data_frames.__contains__('structure_filtered.pkl')):
                                                #         frame = self.get_data('structure_filtered.pkl')
                                                #         for feature in features:
                                                #             if 'structure' in feature:
                                                #                 structure_feature = feature
                                                #                 break
                                                #     else:
                                                #         #get first frame in filtered data
                                                #         frame = self.filtered_data_frames[self.filtered_data_labels[0]]
                                                #         for feature in features:
                                                #             if self.filtered_data_labels[0].replace('_filtered.pkl', '') in feature:
                                                #                 structure_feature = feature
                                                #                 break
                                                # else:
                                                #     frame = self.parity_frame
                                                #     structure_feature = self.parity_feature
                                                frame = self.get_data('al_structures_combined_filtered.pkl')
                                                for feature in features:
                                                    if 'al_structures_combined' in feature:
                                                        structure_feature = feature
                                                        break
                                                analyzer = ModelAnalyzer(chemical_system, trainer_model, bspline_config, frame, frame, structure_feature)                                      
                                                # Export model to LAMMPS format
                                                materials = [materials[0]]
                                                analyzer.set_dft_validation_structures(materials)
                                                analyzer.set_lammps_validation_structures(materials)
                                                analyzer.export_model_to_lammps("potentials/", model_filename, True, "ntaormina", "metal", "nk")
                                                analyzer.export_model_to_lammps(os.getcwd(), model_filename, True, "ntaormina", "metal", "nk")
                                                # Validate the model
                                                all_tests_passed = True
                                                validation_datas = []
                                                validation_data = analyzer.validate_potential(model_filename, model_filename, elements, self.validations_to_run)
                                                print(f"Model {model_filename} validation data: {validation_data}")
                                                self.store_validation_data_local(model_filename, validation_data)
                                                if all_tests_passed:
                                                    self.store_validation_data(model_filename, validation_data)
                                                    self.accept_model(model_filename)
                                                else:
                                                    print(f"Model {model_filename} did not meet cutoffs")
                                                    os.remove(f"{model_filename}")

                                                os.chdir(base_directory)

        #all done, copy the validation data to the valid_potentials folder
        os.rename("validation_datas.txt", f"{self.valid_potentials_folder}/validation_datas.txt")
        
    
    def run_grid_search(self, elements, elements_list, testing_mp_structures, composition_element, composition_range, energy_range, force_range, min_force_range, regularizer_ranges, weight_ranges, validation_data_dir=None, num_cores=1):
        """
        Run a grid search to find the best hyperparameters for the model, including the regularizer values.
        """
        model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_valid_potential_folder()
        #structure_downloader = StructureDownloader(MP_API_KEY)
        materials = structure_downloader.get_structure_list(elements)
        desired_materials = testing_mp_structures
        filtered_materials = [material for material in materials if material.split(':')[0] in desired_materials]
        materials = filtered_materials
        base_directory = os.getcwd()
        for composition in composition_range:
            for energy in energy_range:
                for force in force_range:
                    for min_force in min_force_range: 
                        self.clear_memory()
                        self.load_existing_base_data()
                        self.load_existing_features()
                        
                        # Generate filtered data
                        self.generate_new_filtered_data('structure', 'energy_above_hull', False, energy)
                        self.generate_new_filtered_data('structure', 'maximum_force', True, force, force, force)
                        self.copy_unfiltered_data()
                        self.load_filtered_data()
                        self.consolidate_data()
                        training, testing = self.split_data(0.10)
                        for weight in weight_ranges:
                            for ridge_1b in regularizer_ranges['ridge_1b']:
                                for ridge_2b in regularizer_ranges['ridge_2b']:
                                    for ridge_3b in regularizer_ranges['ridge_3b']:
                                        for curvature_2b in regularizer_ranges['curvature_2b']:
                                            for curvature_3b in regularizer_ranges['curvature_3b']:
                                                print(f"Running grid search for weight={weight}, composition={composition}, energy={energy}, force={force}, "
                                                    f"ridge_1b={ridge_1b}, ridge_2b={ridge_2b}, ridge_3b={ridge_3b}, "
                                                    f"curvature_2b={curvature_2b}, curvature_3b={curvature_3b}")
                                                
                                                model_filename = (f'weight_{weight}_'
                                                                f'ridge1b_{ridge_1b}_ridge2b_{ridge_2b}_ridge3b_{ridge_3b}_'
                                                                f'curv2b_{curvature_2b}_curv3b_{curvature_3b}')
                                                
                                                # Set the current regularizer values
                                                regularizer = dict(ridge_1b=ridge_1b,
                                                                ridge_2b=ridge_2b,
                                                                ridge_3b=ridge_3b,
                                                                curvature_2b=curvature_2b,
                                                                curvature_3b=curvature_3b)
                                                
                                                chemical_system, trainer_model,bspline_config, features, NULL = self.train_model(model_filename + ".json", training, testing, regularizer, weight=weight)
                                                if(self.parity_frame is None):
                                                    frame = self.get_data('structure_filtered.pkl')
                                                    for feature in features:
                                                        if 'structure' in feature:
                                                            structure_feature = feature
                                                            break
                                                else:
                                                    frame = self.parity_frame
                                                    structure_feature = self.parity_feature
                                                testing_indices = set(testing.index)
                                                frame_indices = set(frame.index)
                                                overlap_indices = testing_indices & frame_indices
                                                print(f"Overlap between frame and testing set: {len(overlap_indices)} items")

                                                # Split frame data into training and testing subsets
                                                train_frame = frame[~frame.index.isin(testing_indices)]
                                                test_frame = frame[frame.index.isin(testing_indices)]
                                                analyzer = ModelAnalyzer(chemical_system, trainer_model, bspline_config, train_frame, test_frame, structure_feature)                                      
                                                # Export model to LAMMPS format
                                                #model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                analyzer.set_dft_validation_structures(materials)
                                                analyzer.set_lammps_validation_structures(materials)
                                                analyzer.export_model_to_lammps("potentials/", model_filename, True, "ntaormina", "metal", "nk")
                                                analyzer.export_model_to_lammps(os.getcwd(), model_filename, True, "ntaormina", "metal", "nk")
                                                # Validate the model
                                                validation_data = analyzer.validate_potential(model_filename, model_filename, elements_list, self.validations_to_run)
                                                print(f"Model {model_filename} validation data: {validation_data}")
                                                
                                                # Store validation data
                                                if validation_data_dir is not None:
                                                    self.store_validation_data_alternate_dir(model_filename, validation_data, validation_data_dir)
                                                else:
                                                    self.store_validation_data(model_filename, validation_data)
                                                
                                                os.chdir(base_directory)
                                                # Remove potential file from current working directory
                                                #os.remove(f"{model_filename}")
    def run_grid_search_data_loaded(self, elements, elements_list, testing_mp_structures, composition_element, composition_range, energy_range, force_range, min_force_range, regularizer_ranges, weight_ranges, validation_data_dir=None, num_cores=1):
        """
        Run a grid search to find the best hyperparameters for the model, including the regularizer values.
        """
        model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_valid_potential_folder()
        #structure_downloader = StructureDownloader(MP_API_KEY)
        materials = structure_downloader.get_structure_list(elements)
        desired_materials = testing_mp_structures
        filtered_materials = [material for material in materials if material.split(':')[0] in desired_materials]
        materials = filtered_materials
        base_directory = os.getcwd()
        training, testing = self.split_data(0.10)
        for weight in weight_ranges:
            for ridge_1b in regularizer_ranges['ridge_1b']:
                for ridge_2b in regularizer_ranges['ridge_2b']:
                    for ridge_3b in regularizer_ranges['ridge_3b']:
                        for curvature_2b in regularizer_ranges['curvature_2b']:
                            for curvature_3b in regularizer_ranges['curvature_3b']:
                                print(f"Running grid search for weight={weight},"
                                    f"ridge_1b={ridge_1b}, ridge_2b={ridge_2b}, ridge_3b={ridge_3b}, "
                                    f"curvature_2b={curvature_2b}, curvature_3b={curvature_3b}")
                                
                                model_filename = (f'weight_{weight}_'
                                                f'ridge1b_{ridge_1b}_ridge2b_{ridge_2b}_ridge3b_{ridge_3b}_'
                                                f'curv2b_{curvature_2b}_curv3b_{curvature_3b}')
                                
                                # Set the current regularizer values
                                regularizer = dict(ridge_1b=ridge_1b,
                                                ridge_2b=ridge_2b,
                                                ridge_3b=ridge_3b,
                                                curvature_2b=curvature_2b,
                                                curvature_3b=curvature_3b)
                                
                                chemical_system, trainer_model,bspline_config, features, NULL = self.train_model_parallel(model_filename + ".json", training, testing, regularizer, weight=weight, num_cores=num_cores)
                                if(self.parity_frame is None):
                                    frame = self.get_data('al_structures_combined_filtered.pkl')
                                    for feature in features:
                                        if 'al_structures_combined' in feature:
                                            structure_feature = feature
                                            break
                                else:
                                    frame = self.parity_frame
                                    structure_feature = self.parity_feature
                                testing_indices = set(testing.index)
                                frame_indices = set(frame.index)
                                overlap_indices = testing_indices & frame_indices
                                print(f"Overlap between frame and testing set: {len(overlap_indices)} items")

                                # Split frame data into training and testing subsets
                                train_frame = frame[~frame.index.isin(testing_indices)]
                                test_frame = frame[frame.index.isin(testing_indices)]
                                analyzer = ModelAnalyzer(chemical_system, trainer_model, bspline_config, train_frame, test_frame, structure_feature)                                      
                                # Export model to LAMMPS format
                                #model_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                                analyzer.set_dft_validation_structures(materials)
                                analyzer.set_lammps_validation_structures(materials)
                                analyzer.export_model_to_lammps("potentials/", model_filename, True, "ntaormina", "metal", "nk")
                                analyzer.export_model_to_lammps(os.getcwd(), model_filename, True, "ntaormina", "metal", "nk")
                                # Validate the model
                                validation_data = analyzer.validate_potential(model_filename, model_filename, elements_list, self.validations_to_run)
                                print(f"Model {model_filename} validation data: {validation_data}")
                                
                                # Store validation data
                                 # Store validation data
                                if validation_data_dir is not None:
                                    self.store_validation_data_alternate_dir(model_filename, validation_data, validation_data_dir)
                                else:
                                    self.store_validation_data(model_filename, validation_data)
                                os.chdir(base_directory)
                                # Remove potential file from current working directory
                                os.remove(f"{model_filename}")
    def _parallel_wrapper(self, args):
        """
        Wrapper function for passing arguments to run_one_combination in Pool.
        """
        params, materials, cores = args
        return self.run_one_combination(params, materials, cores)
    def run_one_combination(self, params, materials, cores, position=0):
        """
        Inner loop for optimizing regularizer values (single potential training and validation).
        """
        (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight) = params
        


        print(f"Task {position} running on thread {threading.get_ident()}")
        # Create a unique directory for this combination
        job_label = (
            f"r1b_{ridge_1b}_r2b_{ridge_2b}_r3b_{ridge_3b}_"
            f"curv2b_{curvature_2b}_curv3b_{curvature_3b}_w_{weight}"
        )
        job_folder = os.path.join(self.grid_search_dir, job_label)
        os.makedirs(job_folder, exist_ok=True)
        print("starting job: ", job_label)

        try:
            # Switch to job folder
            current_dir = os.getcwd()
            os.chdir(job_folder)
            
            # Copy the validation folder only if it doesn't already exist
            validation_source = os.path.join(self.base_directory, "validation_base")
            validation_destination = os.path.join(job_folder, "validation", "structures")
            if not os.path.exists(validation_destination):
                shutil.copytree(validation_source, validation_destination, dirs_exist_ok=True)

            # Train the model
            regularizer = dict(
                ridge_1b=ridge_1b,
                ridge_2b=ridge_2b,
                ridge_3b=ridge_3b,
                curvature_2b=curvature_2b,
                curvature_3b=curvature_3b
            )
            model_filename = job_label
            chemical_system, trainer_model, bspline_config, features, _ = self.train_model(
                model_filename, self.training_set, self.testing_set,
                regularizer=regularizer, weight=weight, batch_size=500
            )
            # Validate the model
            #print("getting data: ", os.path.join(self.filter_directory,'al_structures_combined_filtered.pkl'))
            frame = self.get_data('al_structures_combined_filtered.pkl')
            
            analyzer = ModelAnalyzer(chemical_system, trainer_model, bspline_config, frame, frame, features[0])
            analyzer.set_dft_validation_structures(materials)
            analyzer.set_lammps_validation_structures(materials)
            analyzer.set_up_folder_structure()
            analyzer.retrieve_model_files()
            analyzer.export_model_to_lammps(os.getcwd(), model_filename, True, "ntaormina", "metal", "nk")
            #analyzer.export_model_to_lammps(os.path.join(current_dir,'/potentials/'), model_filename, True, "ntaormina", "metal", "nk")
            validation_data = analyzer.validate_potential(model_filename, model_filename, ['Al', 'N'])

            # Store results
            self.store_validation_data_alternate_dir(model_filename, validation_data, self.base_directory)
            os.chdir(current_dir)

            return {"job_label": job_label, "status": "success", "validation": validation_data}

        except Exception as e:
            return {"job_label": job_label, "status": "failed", "error": str(e)}

        finally:
            # Switch back to the original directory
            os.chdir(current_dir)
            
    def run_grid_search_jobs(self, element_list, chemical_formula, testing_mp_structures, regularizer_ranges, weight_range, num_cores=1, cores_per_job=1):
        """
        Orchestrates the grid search over regularizer values in parallel.
        """
        max_workers = num_cores // cores_per_job
        print(f"Running with max {max_workers} parallel jobs, each using {cores_per_job} cores.")

        base_directory = os.getcwd()
        self.setup_valid_potential_folder()
        elastic = {"bulk_modulus_voigt":40.0, "shear_modulus_voigt":40.0}
        lattice_constant = {"a":4.0, "b":4.0, "c":4.0}
        surface_energy = {"bulk_energy":50.0}
        normalized_error = {"normalized_error":0.20}
        cohesive_energy = {"cohesive_energy":40.0}
        self.cutoffs = {"elastic_tensor":elastic, 
                        "lattice_constant":lattice_constant, 
                        "surface_energy":surface_energy}
        tests = ['lattice_constant', 'cohesive_energy', 'elastic_tensor', 'surface_energy']
        #structure_downloader = StructureDownloader(MP_API_KEY)
        materials = structure_downloader.get_structure_list(chemical_formula)
        desired_materials = testing_mp_structures
        filtered_materials = [material for material in materials if material.split(':')[0] in desired_materials]
        materials = filtered_materials
        #self.clear_memory()
        self.load_existing_base_data()
        self.load_existing_features()
        
        # Generate filtered data
        # self.generate_new_filtered_data('al_structures_combined', 'composition', False, 'N', 0.70)
        # self.generate_new_filtered_data('al_structures_combined', 'energy_above_hull', True, 2.0)
        # self.generate_new_filtered_data('al_structures_combined', 'maximum_force', True, 18, 18, 18)
        # self.generate_new_filtered_data('al_structures_combined', 'zero_force', True)
        #self.copy_unfiltered_data()
        self.load_filtered_data()
        self.consolidate_data()
        training, testing = self.split_data(0.10)
        # Generate all parameter combinations for regularizers and weight
        self.grid_search_dir = os.path.join(base_directory, "grid_search")
        param_combinations = []
        for ridge_1b in regularizer_ranges['ridge_1b']:
            for ridge_2b in regularizer_ranges['ridge_2b']:
                for ridge_3b in regularizer_ranges['ridge_3b']:
                    for curvature_2b in regularizer_ranges['curvature_2b']:
                        for curvature_3b in regularizer_ranges['curvature_3b']:
                            for weight in weight_range:
                                params = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight)
                                param_combinations.append(params)
        available_positions = list(range(1, max_workers + 1))  # Position 0 reserved for the main progress bar
        position_map = {}  # M
        results = []
        # Run in parallel using ProcessPoolExecutor
        # with tqdm(total=len(param_combinations), desc="Grid Search", unit="combination", position=0, leave=True) as overall_pbar:
        #     results = []
        with Pool(processes=max_workers) as pool:
            # Use tqdm to display progress
            for result in tqdm(
                pool.imap_unordered(
                    self._parallel_wrapper,
                    [(params, materials, 1) for params in param_combinations],
                ),
                total=len(param_combinations),
                desc="Grid Search",
            ):
                results.append(result)

        # Summarize results
        successes = [r for r in results if r["status"] == "success"]
        failures = [r for r in results if r["status"] == "failed"]

        print(f"Grid search completed. Successes: {len(successes)}, Failures: {len(failures)}")
        for failure in failures:
            print(f"[FAILED] {failure['job_label']}: {failure['error']}")

        # Optionally save results to a file
        pd.DataFrame(successes).to_csv("grid_search_results.csv", index=False)        
    def run_grid_search_parallel(self, element_list, chemical_formula, testing_mp_structures, regularizer_ranges, weight_range, num_cores=1, cores_per_job=1):
        """
        Orchestrates the grid search over regularizer values in parallel.
        """
        max_workers = num_cores // cores_per_job
        print(f"Running with max {max_workers} parallel jobs, each using {cores_per_job} cores.")

        base_directory = os.getcwd()
        self.setup_valid_potential_folder()
        elastic = {"bulk_modulus_voigt":40.0, "shear_modulus_voigt":40.0}
        lattice_constant = {"a":4.0, "b":4.0, "c":4.0}
        surface_energy = {"bulk_energy":50.0}
        normalized_error = {"normalized_error":0.20}
        cohesive_energy = {"cohesive_energy":40.0}
        self.cutoffs = {"elastic_tensor":elastic, 
                        "lattice_constant":lattice_constant, 
                        "surface_energy":surface_energy}
        tests = ['lattice_constant', 'cohesive_energy', 'elastic_tensor', 'surface_energy']
        #structure_downloader = StructureDownloader(MP_API_KEY)
        materials = structure_downloader.get_structure_list(chemical_formula)
        desired_materials = testing_mp_structures
        filtered_materials = [material for material in materials if material.split(':')[0] in desired_materials]
        materials = filtered_materials
        #self.clear_memory()
        self.load_existing_base_data()
        self.load_existing_features()
        
        # Generate filtered data
        # self.generate_new_filtered_data('al_structures_combined', 'composition', False, 'N', 0.70)
        # self.generate_new_filtered_data('al_structures_combined', 'energy_above_hull', True, 2.0)
        # self.generate_new_filtered_data('al_structures_combined', 'maximum_force', True, 18, 18, 18)
        # self.generate_new_filtered_data('al_structures_combined', 'zero_force', True)
        #self.copy_unfiltered_data()
        self.load_filtered_data()
        self.consolidate_data()
        training, testing = self.split_data(0.10)
        # Generate all parameter combinations for regularizers and weight
        self.grid_search_dir = os.path.join(base_directory, "grid_search")
        param_combinations = []
        for ridge_1b in regularizer_ranges['ridge_1b']:
            for ridge_2b in regularizer_ranges['ridge_2b']:
                for ridge_3b in regularizer_ranges['ridge_3b']:
                    for curvature_2b in regularizer_ranges['curvature_2b']:
                        for curvature_3b in regularizer_ranges['curvature_3b']:
                            for weight in weight_range:
                                params = (ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b, weight)
                                param_combinations.append(params)
        available_positions = list(range(1, max_workers + 1))  # Position 0 reserved for the main progress bar
        position_map = {}  # M
        core_manager= CoreManager(num_cores, cores_per_job)
        results = []
        # Run in parallel using ProcessPoolExecutor
        # with tqdm(total=len(param_combinations), desc="Grid Search", unit="combination", position=0, leave=True) as overall_pbar:
        #     results = []
        with Pool(processes=max_workers) as pool:
            # Use tqdm to display progress
            for result in tqdm(
                pool.imap_unordered(
                    self._parallel_wrapper,
                    [(params, materials, 1) for params in param_combinations],
                ),
                total=len(param_combinations),
                desc="Grid Search",
            ):
                results.append(result)

        # Summarize results
        successes = [r for r in results if r["status"] == "success"]
        failures = [r for r in results if r["status"] == "failed"]

        print(f"Grid search completed. Successes: {len(successes)}, Failures: {len(failures)}")
        for failure in failures:
            print(f"[FAILED] {failure['job_label']}: {failure['error']}")

        # Optionally save results to a file
        pd.DataFrame(successes).to_csv("grid_search_results.csv", index=False)
        
    def set_validations_to_run(self, validations_to_run):
        """
        Set the validation tests to run during grid search.
        
        Parameters:
        - validations_to_run: List of validation tests to run.
        """
        self.validations_to_run = validations_to_run
                                            
