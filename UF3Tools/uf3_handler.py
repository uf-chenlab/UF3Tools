import glob
import os
import pandas as pd
from UF3Tools.simulations.managers.ValidationManager import ValidationManager, SingleCrystalValidationManager
from sklearn.cluster import MiniBatchKMeans
from UF3Tools.common.UF3Utility import *
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt
from uf3.util.json_io import *
from uf3.data import io
from uf3.data import geometry
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares
from uf3.forcefield import calculator
from scipy.linalg import svd
from uf3.forcefield import lammps
from uf3.util import parallel
from uf3.util import plotting
from uf3.util import plotting3d
from uf3.util.subsample import *
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import *
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry
from pymatgen.io.ase import AseAtomsAdaptor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from uf3.data import analyze
import re
from UF3Tools.simulations import *
from UF3Tools.simulations.validations.PotentialValidation import PotentialValidation, ValidationData
from UF3Tools.simulations.validations.LatticeConstantValidation import LatticeConstantValidation
from UF3Tools.simulations.validations.ElasticTensorValidation import ElasticTensorValidation
from UF3Tools.simulations.validations.SurfaceEnergyValidation import SurfaceEnergyValidation
from UF3Tools.simulations.validations.CohesiveEnergyValidation import CohesiveEnergyValidation
from UF3Tools.common.MPAPIKey import MP_API_KEY
from UF3Tools.common.StructureDownloader import StructureDownloader
from UF3Tools.common.LAMMPSModifier import LAMMPSModifier
class UF3FileHandler:
    def __init__(self, path):
        self.path = path
    def clear_directory(self):
        for file in os.listdir(self.path):
            file_path = os.path.join(self.path, file)
            if os.path.isfile(file_path):  # Ensuring only files are deleted
                os.remove(file_path)
    def get_files(self):
        return os.listdir(self.path)
    def get_file(self, filename):
        return os.path.join(self.path, filename)
    def backup_directory(self, backup_path):
        os.makedirs(backup_path, exist_ok=True)
        shutil.copytree(self.path, backup_path)
    
class RunDataPhaseDiagram:
    def __init__(self, path):
        self.path = path
        self.data = self.create_df(path)
    def transform_data(self):
        pde_entries = self.get_pde(self.data)
        self.cpd = PhaseDiagram(pde_entries)
        self.pdp = PDPlotter(self.cpd, show_unstable=2)
    def create_comp(self):
        return Composition(self.data['composition'])
    def create_df(self, path):
        df = pd.read_csv(path,delimiter=f'\t',skiprows=[0,1])#, engine='python')
        df['composition']= df['Unnamed: 1']
        df['epa'] = df[' epa']
        df['total_e']= df[' total energy']
        df= df.drop([' epa','id','Unnamed: 4','Unnamed: 6','Unnamed: 7',' num calcs', ' best value', ' total energy',' composition','Unnamed: 1'],axis=1)
        df['composition'] = df.apply(self.create_comp,axis=1)
    #     [for c in df.composition]
        return df
    def get_pde(self, df):
        return [PDEntry(comp,epa,name=str(idx)) for epa, 
                comp, idx in zip(df.total_e,df.composition,df.index)]
    def get_data(self):
        return self.data
    def get_cpd(self):
        return self.cpd
    def get_pdp(self):
        return self.pdp
    def plot_phase_diagram(self):
        self.pdp.get_plot(label_stable=True, label_unstable=False, process_attributes=False)
   



class DataParser:
    def __init__(self, directories, data_coordinator):
        """
        Initialize the DataParser with a list of directories and a data coordinator instance.

        :param directories: List of directory paths to parse.
        :param data_coordinator: An instance of DataCoordinator for parsing files.
        """
        self.directories = directories
        self.data_coordinator = data_coordinator
        self.data = pd.DataFrame()

    def get_directories(self):
        """
        Get the list of directories currently set for parsing.

        :return: List of directory paths.
        """
        return self.directories

    def set_directories(self, directories):
        """
        Set a new list of directories for parsing.

        :param directories: List of directory paths to set.
        """
        self.directories = directories

    def parse_directories(self, output, filter=False):
        """
        Recursively parse all OUTCAR files in the specified directories and their subdirectories,
        and store the parsed data in a DataFrame. This continues indefinitely until manually stopped.
        """
        parsedFiles = 0
        visited_directories = set()
        
        while True:
            new_directories_found = False
            
            for directory in self.directories:
                # Walk through the directory tree recursively
                for root, dirs, files in os.walk(directory):
                    # Check if the directory has already been processed
                    if root in visited_directories:
                        continue
                    visited_directories.add(root)
                    print(f"Parsing directory: {root}")
                    # Split the directory path and concatenate the relevant parent directories
                    relative_path = os.path.relpath(root, directory)
                    path_parts = relative_path.split(os.sep)
                    folder_name = '_'.join([os.path.basename(directory)] + path_parts)
                    
                    # Search for OUTCAR files in the current directory
                    outcar_files = glob.glob(os.path.join(root, 'OUTCAR*'))
                    
                    for file in outcar_files:
                        try:
                            # Extract the run number and index from the filename
                            match = re.match(r'OUTCAR_(\d+)_(\d+)', os.path.basename(file))
                            if match:
                                run_number = match.group(1)
                                index_number = match.group(2)
                                prefix = f'ind_{folder_name}_run_{run_number}__{index_number}'
                            else:
                                print(f"Using default index naming: {os.path.basename(file)}")
                                # Default to use the entire file name if it doesn't match the expected pattern
                                prefix = f'ind_{root}_{folder_name}_run_{os.path.basename(file)}'
                            if len(prefix) > 200:
                                excess_length = len(prefix) - 200
                                prefix = prefix[excess_length:]  # Trim from the front
                            # Use the data_coordinator to parse the file and update the data.
                            # if prefix contains anything from the data type list, turn off the filter
                            print("debug data prefix: ", prefix)
                            try:
                                self.data_coordinator.dataframe_from_trajectory(file, prefix=prefix, filter=filter)
                            except Exception as e:
                                print(f"Error parsing file {file}: {e}")
                            parsedFiles += 1
                        except Exception as e:
                            print(f"Error processing {file}: {e}")

                    # Mark that a new directory has been found and processed
                    new_directories_found = True

            if not new_directories_found:
                break

        # Consolidate the data after processing all files.
        if parsedFiles > 0:
            self.data = self.data_coordinator.consolidate()
            self.check_and_convert_indices()
    def parse_directories_with_subsampling(self, output, max_samples=1000, min_energy=1e-3):
        parsed_files = 0
        visited_directories = set()
        data_paths = []
        prefixes = {}

        while True:
            new_directories_found = False

            for directory in self.directories:
                for root, dirs, files in os.walk(directory):
                    if root in visited_directories:
                        continue
                    visited_directories.add(root)
                    print(f"Parsing directory: {root}")

                    relative_path = os.path.relpath(root, directory)
                    path_parts = relative_path.split(os.sep)
                    folder_name = '_'.join([os.path.basename(directory)] + path_parts)

                    outcar_files = glob.glob(os.path.join(root, 'OUTCAR*'))

                    for file in outcar_files:
                        match = re.match(r'OUTCAR_(\d+)_(\d+)', os.path.basename(file))
                        if match:
                            run_number = match.group(1)
                            index_number = match.group(2)
                            prefix = f'ind_{folder_name}_run_{run_number}__{index_number}'
                        else:
                            print(f"Using default index naming: {os.path.basename(file)}")
                            prefix = f'ind_{root}_{folder_name}_run_{os.path.basename(file)}'

                        if len(prefix) > 200:
                            excess_length = len(prefix) - 200
                            prefix = prefix[excess_length:]

                        print("debug data prefix:", prefix)
                        data_paths.append(file)
                        prefixes[file] = prefix

                    new_directories_found = True

            if not new_directories_found:
                break

        if data_paths:
            io.parse_with_subsampling(
                data_paths=data_paths,
                data_coordinator=self.data_coordinator,
                max_samples=max_samples,
                min_diff=min_energy,
                prefixes=prefixes
            )
            parsed_files = len(data_paths)

        if parsed_files > 0:
            self.data = self.data_coordinator.consolidate()
            self.check_and_convert_indices()
                          
    def check_and_convert_indices(self):
        """
        Check if the DataFrame indices are in the correct format (ind_#_run_#__#).
        If not, convert them to the correct format.
        
        :return: DataFrame with corrected indices.
        """
        
        # Regular expression for the correct index format
        correct_format = re.compile(r'ind_\d+_run_\d+__\d+')

        # Function to convert an index to the correct format
        def convert_index(index):
            # Example of incorrect format: 'ind_291_run_OUTCAR_3_0'
            # Correct format should be 'ind_291_run_3__0'
            parts = index.split('_')
            if len(parts) >= 6 and parts[3] == 'OUTCAR':
                return f'{parts[0]}_{parts[1]}_{parts[2]}_{parts[5]}__{parts[4]}'
            return index

        # Check and convert indices
        new_indices = []
        for index in self.data.index:
            if not correct_format.match(index):
                new_index = convert_index(index)
                new_indices.append(new_index)
            else:
                new_indices.append(index)
        
        # Update the DataFrame with the new indices
        self.data.index = new_indices

        return self.data

    def set_data(self, data):
        """
        Set the DataFrame containing the parsed data.

        """
        self.data = data

    def export_to_excel(self, filename):
        """
        Export the DataFrame to an Excel file.

        :param filename: Name of the Excel file.
        """
        self.data.to_excel(filename, index=True, sheet_name='ConsolidatedData')
        print(f'DataFrame has been exported to {filename}')

    def export_to_pickle(self, filename):
        """
        Export the DataFrame to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.data.to_pickle(filename)
    
    def load_from_pickle(self, filename):
        self.data = pd.read_pickle(filename)
        #make sure it actually has any data in it
        if self.data.empty:
            raise ValueError(f"Pickle file {filename} is empty.")
        # Checking if 'energy_per_atom' is missing
        if 'energy_per_atom' not in self.data.columns:
            sp = StructureParser(self.data)
            sp.append_energy_per_atom()
            self.data = sp.get_data()
            # Updating the pickle with the newly calculated column
            self.export_to_pickle(filename)
    def get_data(self):
        
        """
        Get the DataFrame containing the parsed data.

        :return: DataFrame containing the parsed data.
        """
        return self.data    
    def get_data_length(self):
        """
        Get the number of rows in the DataFrame.

        :return: Number of rows in the DataFrame.
        """
        #determine the number of unique indices. It will be in format ind_#, whwere # is what needs to be unique
        index_prefixes = self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).unique()
        
        return len(index_prefixes)
    def get_unique_indices(self):
        """
        Get the unique indices in the DataFrame.

        :return: List of unique indices.
        """
        return self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).unique()

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from ase.neighborlist import neighbor_list    
class DataSetAnalyzer:
    # Class-wide variable to hold the data type prefixes
    DATA_TYPE_PREFIXES = {
        "pressure": "p_",
        "temperature": "t_",
        "temperature": "T_",
        "strain": "strain_",
        "defect": "defect_",
        "surface": "surface_",
        # Add more types as needed
    }

    def __init__(self, data):
        """
        Initialize the DataSetAnalyzer with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data

    def get_data(self):
        """
        Get the DataFrame containing the parsed data.

        :return: DataFrame containing the parsed data.
        """
        return self.data

    def set_data(self, data):
        """
        Set a new DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data

    def get_data_length(self):
        """
        Get the number of rows in the DataFrame.

        :return: Number of rows in the DataFrame.
        """
        return len(self.data)

    def get_unique_indices(self):
        """
        Get the unique indices in the DataFrame.

        :return: List of unique indices.
        """
        return self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).unique()

    
    def export_to_pickle(self, filename):
        """
        Export the DataFrame to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.data.to_pickle(filename)
    
    def plot_eah_histogram(self, bins=20, alpha=0.5, upper_bound=None, lower_bound=None):
        """
        Plot a histogram of the energy above hull values, with optional upper and lower bounds.
        """
        # Apply lower bound filter if specified
        if lower_bound is not None:
            filtered_data = self.data[self.data['eah'] >= lower_bound]
        else:
            filtered_data = self.data

        # Apply upper bound filter if specified
        if upper_bound is not None:
            filtered_data = filtered_data[filtered_data['eah'] <= upper_bound]

        # Plot the histogram with the filtered data
        filtered_data['eah'].plot.hist(bins=bins, alpha=alpha)
        plt.xlabel('Energy above hull (eV/atom)')
        plt.ylabel('Frequency')
        plt.title('Energy above hull histogram')
        plt.show()
        
    def plot_epa_histogram(self, bins=20, alpha=0.5, upper_bound=None, lower_bound=None):
        """
        Plot a histogram of the energy per atom values, with optional upper and lower bounds.
        """
        # Apply lower bound filter if specified
        if lower_bound is not None:
            filtered_data = self.data[self.data['energy_per_atom'] >= lower_bound]
        else:
            filtered_data = self.data

        # Apply upper bound filter if specified
        if upper_bound is not None:
            filtered_data = filtered_data[filtered_data['energy_per_atom'] <= upper_bound]

        # Plot the histogram with the filtered data
        filtered_data['energy_per_atom'].plot.hist(bins=bins, alpha=alpha)
        plt.xlabel('Energy per atom (eV/atom)')
        plt.ylabel('Frequency')
        plt.title('Energy per atom histogram')
        plt.show()

    def plot_composition_histogram(self, element, bins=20, alpha=0.5):
        """
        Plot a histogram of the element composition.
        """
        self.data[f'{element}_frac'].plot.hist(bins=bins, alpha=alpha)
        plt.xlabel(f'{element} fraction')
        plt.ylabel('Frequency')
        plt.title(f'{element} fraction histogram')
        plt.show()
        

    def plot_force_histogram(self, bins=20, alpha=0.5, bin_range=[-10, 10]):
        """
        Plot histograms for the forces fx, fy, and fz with an optional bin range.
        
        Parameters:
        - bins: Number of bins for the histogram.
        - alpha: Transparency of the bars.
        - bin_range: A tuple or list of two numbers (min, max) setting the limits of the bins.
        """

        # Define a helper function to flatten the lists and plot a histogram
        def plot_histogram(data, column, bins, alpha, bin_range):
            # Flatten the lists in the column
            flattened_data = [item for sublist in data[column] for item in sublist]
            
            # Convert the flattened list to a Pandas Series
            series_data = pd.Series(flattened_data)
            
            # Plot the histogram with the specified bin range
            series_data.plot.hist(bins=bins, alpha=alpha, range=bin_range)
            plt.xlabel(f'Force {column.upper()}')
            plt.ylabel('Frequency')
            plt.title(f'Force {column.upper()} Histogram')
            plt.show()

        # Plot histograms for fx, fy, and fz with the specified bin range
        plot_histogram(self.data, 'fx', bins, alpha, bin_range)
        plot_histogram(self.data, 'fy', bins, alpha, bin_range)
        plot_histogram(self.data, 'fz', bins, alpha, bin_range)
        
    def plot_2b_pair_distance_densities(self, chemical_system, r_max=7.0, bins=200):
        # Collect pairwise distances for each 2-body interaction
        pair_distances = defaultdict(list)
        for atoms in self.data['geometry']:
            # compute full neighbor list
            i_list, j_list, distances = neighbor_list('ijd', atoms, cutoff=r_max)
            symbols = atoms.get_chemical_symbols()
            for i, j, dist in zip(i_list, j_list, distances):
                if i >= j:
                    continue  # avoid double-counting
                pair = tuple(sorted((symbols[i], symbols[j])))
                if pair in chemical_system.interactions_map[2]:
                    pair_distances[pair].append(dist)
        
        # Plot normalized histograms
        for pair, dists in pair_distances.items():
            dists = np.array(dists)
            hist, edges = np.histogram(dists, bins=bins, range=(0, r_max), density=True)
            centers = 0.5 * (edges[1:] + edges[:-1])
            plt.plot(centers, hist, label=f'{pair[0]}-{pair[1]}')

        plt.xlabel('Pair distance (Å)')
        plt.ylabel('Relative density')
        plt.title('2-body pair distance densities')
        plt.legend()
        plt.grid(True)
        plt.show()

    


class StructureParser:
    def __init__(self, data):
        """
        Initialize the StructureParser with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        #self.data = self.transform_data(data)
    def create_comp(self, df):
        return Composition(df['composition'])
    def read_run_data(self, path):
        df = pd.read_csv(path,delimiter=f'\t',skiprows=[0,1])
        df['composition']= df['Unnamed: 1']
        df['epa'] = df[' epa']
        df['total_e']= df[' total energy']
        df= df.drop([' epa','id','Unnamed: 4','Unnamed: 6','Unnamed: 7',' num calcs', ' best value', ' total energy',' composition','Unnamed: 1'],axis=1)
        df['composition'] = df.apply(self.create_comp,axis=1)
        self.run_data = df
        return df
    def get_run_data_pde(self, df):
        return [PDEntry(comp,epa,name=str(idx)) for epa, 
                comp, idx in zip(df.total_e,df.composition,df.index)]
    def get_unique_indices(self):
        """
        Get the unique indices in the DataFrame.

        :return: List of unique indices.
        """
        return self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).unique()
    def transform_data(self, data, epa_filter = [-15, 0], phase_diagram=True):
        #self.data['composition'] = self.data.apply(self.create_comp,axis=1)
        self.append_energy_per_atom()
        self.filter_epa(epa_filter)
        if(phase_diagram == True):
            pde_entries = self.get_pde_entries(self.data)
            self.cpd = PhaseDiagram(pde_entries)
            self.append_phase_diagram_energy()
            self.append_energy_above_hull()
        
            element_list = self.get_element_list()
            print(f"Elements in phase diagram: {element_list}")
            for element in element_list:
                self.append_element_fraction(element)
            
        return self.data
    def filter_epa(self, epa_filter):
        self.data = self.data[(self.data['energy_per_atom'] >= epa_filter[0]) & (self.data['energy_per_atom'] <= epa_filter[1])]
    def get_data(self):
        """
        Get the DataFrame containing the parsed data.

        :return: DataFrame containing the parsed data.
        """
        return self.data

    def set_data(self, data):
        """
        Set a new DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
    def export_to_pickle(self, filename):
        """
        Export the DataFrame to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.data.to_pickle(filename)
        
    def get_element_list(self):
        """
        Get the list of elements in the phase diagram.

        :return: List of elements in the phase diagram.
        """
        return self.cpd.elements
    def get_pdp(self):
        return self.pdp
    def get_composition(self, df_row):
        """
        Returns the composition of the structure.
        :param df_row: Series containing structure data.
        :return: Composition of the structure.
        """
        comp = AseAtomsAdaptor().get_structure(df_row.geometry).composition
        #insert a space between element and number
        comp = Composition(" ".join(str(comp).split()))
        return comp
    def get_pde_entries(self, df):
        """
        Returns a list of PhaseDiagramEntry objects from the DataFrame.
        :param df: DataFrame containing structure data.
        :return: List of PhaseDiagramEntry (PDEntry) objects.
        """
        entries = []
        for i in range(len(df)):
            index = df.index[i]
            comp = self.get_composition(df.loc[index])
            entry = PDEntry(comp, df.loc[index].energy, name=str(index))
            entries.append(entry)

        return entries
    def get_pde_single(self, df_row):
        """
        Returns a PhaseDiagramEntry object from a DataFrame row.
        :param df_row: Series containing structure data.
        :return: PhaseDiagramEntry (PDEntry) object.
        """
        comp = self.get_composition(df_row)
        return PDEntry(comp, df_row.energy, name=str(df_row.name))

    def get_frac(self, df_row, element):
        """
        Returns the fraction of element in the structure.
        :param df_row: Series containing structure data.
        :param element: Element to calculate fraction for.
        :return: Fraction of element in the structure.
        """
        comp = AseAtomsAdaptor().get_structure(df_row.geometry).composition
        return comp.get_atomic_fraction(element)

    def get_energy_above_hull(self, df_row):
        """
        Returns the energy above hull of the structure.
        :param df_row: Series containing structure data.
        :return: Energy above hull of the structure.
        """
        entry = self.get_pde_single(df_row)
        return self.cpd.get_e_above_hull(entry, allow_negative=True)

    def append_energy_above_hull(self):
        """
        Appends the energy above hull to the DataFrame.
        """
        self.data['eah'] = self.data.apply(self.get_energy_above_hull, axis=1)

    def append_phase_diagram_energy(self):
        """
        Appends the phase diagram energy to the DataFrame.
        """
        self.data['pde'] = self.data.apply(self.get_pde_single, axis=1)

    def append_element_fraction(self, element):
        """
        Appends the fraction of element to the DataFrame.
        """
        columnName = f'{element}_frac'
        self.data[columnName] = self.data.apply(self.get_frac, args=(element,), axis=1)

    def plot_phase_diagram(self):
        """
        Plots the phase diagram.
        """
        pde_entries = self.get_run_data_pde(self.run_data)
        self.cpd = PhaseDiagram(pde_entries)
        self.pdp = PDPlotter(self.cpd, show_unstable=2)
        self.pdp.get_plot(label_stable=True, label_unstable=False, process_attributes=False)
        
    def plot_composition_histogram(self, element):
        """
        Plots a histogram of the element composition.
        """
        self.data[f'{element}_frac'].plot.hist(bins=20, alpha=0.5)
        plt.xlabel(f'{element} fraction')
        plt.ylabel('Frequency')
        plt.title(f'{element} fraction histogram')
        plt.show()
    def get_cpd(self):
        return self.cpd
    def get_index_composition(self, index):
        """
        Get the composition of the structure at the specified index.

        :param index: Index prefix of the structure.
        :return: Composition of the structure for the first matching index.
        """
        # Find the first matching index that starts with the given index prefix
        matching_index = index
        
        if matching_index is not None:
            # Return the composition for the first matching index
            return self.get_composition(self.data.loc[matching_index])
        else:
            # If no matching index is found, return None or raise an exception
            return None  # or raise an exception if that's more appropriate 
    def get_energy_per_atom(self, df_row):
        """
        Returns the energy per atom of the structure.
        :param df_row: Series containing structure data.
        :return: Energy per atom of the structure.
        """
        comp = AseAtomsAdaptor().get_structure(df_row.geometry).composition
        return df_row.energy / comp.num_atoms
    def append_energy_per_atom(self):
        """
        Appends the energy per atom to the DataFrame.
        """
        self.data['energy_per_atom'] = self.data.apply(self.get_energy_per_atom, axis=1)
    


class DataFilter:
    def __init__(self, data):
        """
        Initialize the DataFilter with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        
    def get_data(self):
        """
        Get the DataFrame containing the parsed data.

        :return: DataFrame containing the parsed data.
        """
        return self.data
    
    def set_data(self, data):
        """
        Set a new DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
    def filter_reduce_data_points(self, num_points):
        print(f"Reducing data from {len(self.data)} points to {num_points} points.")
        if num_points >= len(self.data):
            print("Requested number of points is greater than or equal to the current data size. No reduction applied.")
            return
        self.data = self.data.sample(n=num_points)
    def filter_increase_data_points(self, num_points):
        additional_rows_needed = num_points - len(self.data)
        print(f"Increasing data from {len(self.data)} points to {num_points} points.")
        if additional_rows_needed > 0:
            additional_data = self.data.sample(n=additional_rows_needed, replace=True)
            self.data = pd.concat([self.data, additional_data], ignore_index=False)
        
    def filter_composition(self, element, percent):
        print(f"Filtering out data with over {percent*100}% {element} composition.")
        #check if data has a fraction column for the element
        if f'{element}_frac' in self.data.columns:
            self.data = self.data[self.data[f'{element}_frac'] <= percent]
        else:
            print(f"Element {element} fraction not found in data. Please add columns for each element fraction using StructureParser.")
    def filter_composition_range(self, element, el_range):
        print(f'Filterout out data with {element} composition outside of {el_range[0]} and {el_range[1]}.')
        #check if data has a fraction column for the element
        if f'{element}_frac' in self.data.columns:
            self.data = self.data[(self.data[f'{element}_frac'] >= el_range[0]) & (self.data[f'{element}_frac'] <= el_range[1])]
        else:
            print(f"Element {element} fraction not found in data. Please add columns for each element fraction using StructureParser.")
    def filter_gas(self, element):
        """
        Filters out rows from the dataframe containing the specified gas in the composition.

        Args:
            element (str): Element symbol to check for gas (e.g., 'H' for 'H2').
        """
        print(f'Filtering out data with {element}2 gas in the composition.')
        
        gas = f'{element}2'

        def has_gas(row):
            try:
                # Convert geometry to ASE Atoms and check for the gas in the symbols
                #atoms = AseAtomsAdaptor().get_atoms(AseAtomsAdaptor().get_structure(str(row['geometry'])))
                return gas in str(row['geometry'])
            except Exception as e:
                print(f"Error processing row: {e}")
                return False

        # Filter rows efficiently using DataFrame's .apply method
        self.data = self.data[~self.data.apply(has_gas, axis=1)]
        
        print(f"Removed rows containing {gas}.")
    def filter_min_epa(self, threshold):
        print(f"Filtering data for energy per atom greater than {threshold}.")
        self.data = self.data[self.data['energy_per_atom'] >= threshold]
    def filter_max_epa(self, threshold):
        print(f"Filtering data for energy per atom less than {threshold}.")
        self.data = self.data[self.data['energy_per_atom'] <= threshold]

        
    def filter_energy_above_hull(self, threshold):
        print(f"Filtering data for energies less than {threshold}.")
        #check if data has an energy above hull column
        if 'eah' in self.data.columns:
            self.data = self.data[self.data['eah'] <= threshold]
        else:
            print(f"Energy above hull column not found in data. Please add column for energy above hull using StructureParser.")
        
    def filter_force(self, thresholdX, thresholdY, thresholdZ):
        print(f"Filtering data for forces less than {thresholdX}, {thresholdY}, {thresholdZ}.")
        #check if data has a force column or individual force columns
        if 'force' in self.data.columns:
            threshold = np.sqrt(thresholdX**2 + thresholdY**2 + thresholdZ**2)
            self.data = self.data[self.data['force'] <= threshold]
        elif 'forceX' in self.data.columns and 'forceY' in self.data.columns and 'forceZ' in self.data.columns:
            self.data = self.data[self.data['forceX'] <= thresholdX]
            self.data = self.data[self.data['forceY'] <= thresholdY]
            self.data = self.data[self.data['forceZ'] <= thresholdZ]
        elif 'fx' in self.data.columns and 'fy' in self.data.columns and 'fz' in self.data.columns:
            self.data = self.data.loc[~self.data['fx'].apply(lambda x: any(abs(x) > thresholdX))]
            self.data = self.data.loc[~self.data['fy'].apply(lambda x: any(abs(x) > thresholdY))]
            self.data = self.data.loc[~self.data['fz'].apply(lambda x: any(abs(x) > thresholdZ))]

        else:
            print(f"Force columns not found in data. Please add columns for force. These should have come from the DFT OUTCAR.")
    def filter_force_minimum(self, thresholdX, thresholdY, thresholdZ):
        print(f"Filtering data for forces less than {thresholdX}, {thresholdY}, {thresholdZ}.")
        #check if data has a force column or individual force columns
        if 'force' in self.data.columns:
            threshold = np.sqrt(thresholdX**2 + thresholdY**2 + thresholdZ**2)
            self.data = self.data[self.data['force'] >= threshold]
        elif 'forceX' in self.data.columns and 'forceY' in self.data.columns and 'forceZ' in self.data.columns:
            self.data = self.data[self.data['forceX'] >= thresholdX]
            self.data = self.data[self.data['forceY'] >= thresholdY]
            self.data = self.data[self.data['forceZ'] >= thresholdZ]
        elif 'fx' in self.data.columns and 'fy' in self.data.columns and 'fz' in self.data.columns:
            self.data = self.data.loc[~self.data['fx'].apply(lambda x: any(abs(x) < thresholdX))]
            self.data = self.data.loc[~self.data['fy'].apply(lambda x: any(abs(x) < thresholdY))]
            self.data = self.data.loc[~self.data['fz'].apply(lambda x: any(abs(x) < thresholdZ))]

        else:
            print(f"Force columns not found in data. Please add columns for force. These should have come from the DFT OUTCAR.")
            
    def remove_zero_force(self):
        """
        Remove data points with zero forces in all 3 directions.
        """
        if 'fx' in self.data.columns and 'fy' in self.data.columns and 'fz' in self.data.columns:
            # Apply the lambda function to each force component
            self.data = self.data.loc[
                ~self.data['fx'].apply(lambda x: all(x == 0)) &
                ~self.data['fy'].apply(lambda x: all(x == 0)) &
                ~self.data['fz'].apply(lambda x: all(x == 0))
            ]
            
    def filter_out_indices(self, indices):
        """
        Filter out data points with the specified indices.

        :param indices: List of indices to filter out.
        """
        print(f"Filtering out data points with indices: {indices}")
        self.data = self.data[~self.data.index.isin(indices)]

    def export_to_pickle(self, filename):
        """
        Export the DataFrame to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.data.to_pickle(filename)
    def filter_unreasonable_epas(self):
        # Drop anything with a positive EPA (assuming negative EPAs are physically valid)
        negative_df = self.data[self.data['energy_per_atom'] <= 1]
        
        # Compute Q1, Q3, and IQR on the negative subset
        q1 = negative_df['energy_per_atom'].quantile(0.25)
        q3 = negative_df['energy_per_atom'].quantile(0.75)
        iqr = q3 - q1
        
        # Define lower and upper bounds
        lower_fence = q1 - 2.0 * iqr
        upper_fence = q3 + 2.0 * iqr
        
        # Keep only the points within these bounds
        filtered_df = negative_df[
            (negative_df['energy_per_atom'] >= lower_fence) & 
            (negative_df['energy_per_atom'] <= upper_fence)
        ]
        
        # Optionally report how many rows were removed
        removed_count = len(self.data) - len(filtered_df)
        print(f"Removed {removed_count} outlier rows based on IQR filtering of energy_per_atom.")
        
        self.data = filtered_df
    def filter_kmeans_subsample(self, n_samples, column='energy_per_atom'):
        """
        Subsample data using MiniBatch KMeans on a single column or descriptor vector.
        Picks one representative per cluster.
        """
        print(f"Current data length: {len(self.data)}")

        if column not in self.data.columns:
            print(f"Missing column: '{column}'")
            return

        X = self.data[column].values

        # handle case where each entry is a float
        if not hasattr(X[0], '__len__'):
            X = np.array(X).reshape(-1, 1)
        else:
            X = np.stack(X).astype(float)

        if n_samples >= len(X):
            print("Requested sample count >= dataset size. Skipping.")
            return

        kmeans = MiniBatchKMeans(n_clusters=n_samples, batch_size=2048, max_iter=100)
        kmeans.fit(X)

        closest_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        self.data = self.data.iloc[np.sort(closest_idx)]

        print(f"Filtered data length: {len(self.data)}")
    import numpy as np

    def filter_farthest_point_subsample(self, max_samples=100, min_diff=0.0, column='energy_per_atom'):
        """
        Efficient farthest point sampling with optional early stopping based on min_diff.
        """
        print(f"Running FPS on '{column}' with max_samples={max_samples}, min_diff={min_diff}")

        if column not in self.data.columns:
            print(f"Missing column: {column}")
            return

        values = self.data[column].values.astype(float)
        if values.ndim == 1:
            values = values[:, np.newaxis]

        n_points = len(values)
        if (max_samples >= n_points and min_diff == 0):
            print("Requested samples >= dataset size. Returning full dataset.")
            return

        selected = [np.argmin(values[:, 0])]
        distances = np.full(n_points, np.inf)

        for _ in range(1, max_samples):
            last_point = values[selected[-1]]
            new_distances = np.linalg.norm(values - last_point, axis=1)
            distances = np.minimum(distances, new_distances)

            # Select the point with the farthest minimum distance
            max_dist = distances.max()
            if min_diff > 0 and max_dist < min_diff:
                print(f"Early stopping at {_} samples due to min_diff = {min_diff}")
                break

            next_idx = distances.argmax()
            selected.append(next_idx)

        self.data = self.data.iloc[np.sort(selected)]
        print(f"Subsampled to {len(self.data)} points using efficient FPS.")
    def filter_cur_subsample(self, n_samples, columns=None, rank=None):
        """
        Subsample data using CUR decomposition by selecting rows with high statistical leverage scores.

        Parameters:
        - n_samples: The number of samples to select.
        - columns: A list of column names to consider for the CUR decomposition.
                   If None, all numerical columns will be used.
        - rank: The rank 'k' to use for computing leverage scores. If None, it will be
                set to n_samples or a reasonable fraction of the data dimension.
        """
        print(f"Current data length: {len(self.data)}")
        print(self.data.columns)
        print(self.data)

        if n_samples >= len(self.data):
            print("Requested sample count >= dataset size. Skipping CUR subsampling.")
            return

        # Prepare the data matrix for SVD
        print(set(columns))
        if columns is None:
            temp_df = self.data.select_dtypes(include=np.number).copy()
        
        else:
            if set(columns) <= {'fx', 'fy', 'fz'}:
                mask = self.data[columns].applymap(lambda x: isinstance(x, (list, np.ndarray)) and x is not None)
                valid_rows = mask.all(axis=1)
                filtered_df = self.data.loc[valid_rows]
                
                if filtered_df.empty:
                    print("No valid rows with force arrays for CUR subsampling.")
                    return

                def flatten_row(row):
                    return np.concatenate([np.array(row[col], dtype=np.float64) for col in columns])
                
                X = np.vstack(filtered_df.apply(flatten_row, axis=1).values)
                current_indices = filtered_df.index

            else:
                # Regular numeric columns
                print(self.data[columns])
                temp_df = self.data[columns].apply(pd.to_numeric, errors='coerce')
                print(temp_df)
                if temp_df.empty:
                    print("No data remaining after dropping NaNs for CUR subsampling.")
                    return
                X = temp_df.to_numpy(dtype=np.float64)
                current_indices = temp_df.index
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            print("No valid numerical data for CUR subsampling.")
            return

        # Handle potential NaNs by filling with 0 or a mean, or dropping rows/columns
        # For simplicity, let's drop rows with NaNs for now if not already handled
    
        # Perform SVD
        # U contains the left singular vectors (for row leverage scores)
        U, s, Vt = svd(X, full_matrices=False)

        # Determine the rank k for leverage scores
        if rank is None:
            # Common heuristics: min(n_samples, small_constant * min(rows, cols))
            # Let's use min(n_samples, min(U.shape)) to ensure it's within bounds
            k = min(n_samples, 10 * min(U.shape[0], U.shape[1]))
            if k == 0:
                print("Effective rank for SVD is zero. Cannot perform CUR subsampling.")
                return
        else:
            k = min(rank, U.shape[1])
            if k == 0:
                print("Specified rank for SVD is zero. Cannot perform CUR subsampling.")
                return

        # Calculate row leverage scores
        # Leverage scores are the squared Euclidean norms of the rows of U up to rank k
        leverage_scores = np.sum(U[:, :k]**2, axis=1)

        # Select indices corresponding to the top n_samples leverage scores
        # We use argsort and then slice the last n_samples (highest scores)
        if 'current_indices' in locals(): # if NaNs were handled
            selected_indices_in_temp_df = np.argsort(leverage_scores)[-n_samples:]
            selected_original_indices = current_indices[selected_indices_in_temp_df]
            self.data = self.data.loc[selected_original_indices]
        else:
            selected_indices = np.argsort(leverage_scores)[-n_samples:]
            self.data = self.data.iloc[selected_indices]

        print(f"Filtered data length after CUR subsampling: {len(self.data)}")


class Featurizer:
    def __init__(self, data, chemical_system, bspline_config):
        """
        Initialize the Featurizer with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        self.chemical_system = chemical_system
        self.bspline_config = bspline_config
        self.representation = process.BasisFeaturizer(self.bspline_config)
        
    def get_data(self):
        """
        Get the DataFrame containing the parsed data.

        :return: DataFrame containing the parsed data.
        """
        return self.data
    
    def set_data(self, data):
        """
        Set a new DataFrame containing parsed data.
        """
        self.data = data
    
    def get_bspline_config(self):
        """
        Get the bspline configuration.
        """
        return self.bspline_config
    
    def set_bspline_config(self, bspline_config):
        """
        Set the bspline configuration.
        """
        self.bspline_config = bspline_config
    
    def get_maps(self):
        """
        Get the maps.
        """
        return self.bspline_config
    
    def set_maps(self, maps):
        """
        Set the maps.
        """
        self.bspline_config = maps
    
    def get_chemical_system(self):
        """
        Get the chemical system.
        """
        return self.chemical_system
    
    def set_chemical_system(self, chemical_system):
        """
        Set the chemical system.
        """
        self.chemical_system = chemical_system
    
    def get_element_list(self):
        """
        Get the list of elements in the chemical system.
        """
        return self.chemical_system.element_list
    
    def featurize_data(self, num_cores, output_filename, batch_size=500):
        """
        Featurize the data.
        """
        # Check if the file already exists
        if os.path.exists(output_filename):
            print(f"Feature file already exists: {output_filename}")
            return

        # Adjust batch size if data size is smaller
        data_size = len(self.data)
        if data_size == 0:
            raise ValueError("No data available for featurization.")
        adjusted_batch_size = min(batch_size, data_size)

        table_template = "features_{}"
        client = ProcessPoolExecutor(max_workers=num_cores)

        # Log adjustment
        if adjusted_batch_size != batch_size:
            adjusted_batch_size = adjusted_batch_size/2
            adjusted_batch_size = round(adjusted_batch_size)
            print(f"Adjusted batch size to {adjusted_batch_size} due to smaller dataset.")
            
        if(adjusted_batch_size <= 1):
            adjusted_batch_size = adjusted_batch_size + 1
            print(f"Adjusted batch size to {adjusted_batch_size} due to smaller dataset.")
            
        # Featurization
        self.representation.batched_to_hdf(
            output_filename,
            self.data,
            client,
            n_jobs=num_cores,
            batch_size=adjusted_batch_size,
            progress="bar",
            table_template=table_template
        )
        
class ModelTrainer:
    def __init__(self, data, chemical_system, bspline_config, regularizer):
        """
        Initialize the ModelTrainer with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        self.chemical_system = chemical_system
        self.bspline_config = bspline_config
        self.representation = process.BasisFeaturizer(self.bspline_config)
        self.regularizer = self.bspline_config.get_regularization_matrix(ridge_1b=regularizer['ridge_1b'],
                                                                    ridge_2b=regularizer['ridge_2b'],
                                                                    ridge_3b=regularizer['ridge_3b'],
                                                                    curvature_2b=regularizer['curvature_2b'],
                                                                    curvature_3b=regularizer['curvature_3b'])
        self.model = least_squares.WeightedLinearModel(self.bspline_config, self.regularizer)
    def get_data(self):
        """
        Get the DataFrame containing the parsed data.

        :return: DataFrame containing the parsed data.
        """
        return self.data
    def set_data(self, data):
        """
        Set a new DataFrame containing parsed data.
        """
        self.data = data
    def get_bspline_config(self):
        """
        Get the bspline configuration.
        """
        return self.bspline_config
    def set_bspline_config(self, bspline_config):
        """
        Set the bspline configuration.
        """
        self.bspline_config = bspline_config
    def get_maps(self):
        """
        Get the maps.
        """
        return self.bspline_config
    def set_maps(self, maps):
        """
        Set the maps.
        """
        self.bspline_config = maps
    def get_chemical_system(self):
        """
        Get the chemical system.
        """
        return self.chemical_system
    def set_chemical_system(self, chemical_system):
        """
        Set the chemical system.
        """
        self.chemical_system = chemical_system
    def get_element_list(self):
        """
        Get the list of elements in the chemical system.
        """
        return self.chemical_system.element_list
    def get_unique_indices(self):
        """
        Get the unique indices in the DataFrame.

        :return: List of unique indices.
        """
        return self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).unique()
    def split_data(self, test_percent, filter_index=True):
        """
        Split the data into training and testing sets.

        Parameters:
        - test_percent: Fraction of data to use for testing.
        - filter_index: Boolean flag to determine whether or not to require indices
        to be fully in either train or test.
        """
        # Get unique index prefixes
        index_prefixes = self.get_unique_indices()
        
        np.random.seed(42)  # for reproducibility
        num_indices_to_select = int(test_percent * len(index_prefixes))
        selected_prefixes = np.random.choice(index_prefixes, num_indices_to_select, replace=False)

        if filter_index:
            # Filter data for testing set based on the selected prefixes
            self.testing_set = self.data[self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).isin(selected_prefixes)]

            # Filter data for training set
            self.training_set = self.data[~self.data.index.isin(self.testing_set.index)]
        else:
            # Shuffle and split data without filtering based on prefixes
            shuffled_data = self.data.sample(frac=1, random_state=42)
            split_point = int(len(shuffled_data) * (1 - test_percent))
            self.training_set = shuffled_data.iloc[:split_point]
            self.testing_set = shuffled_data.iloc[split_point:]

        # Store the indices for training and testing sets
        self.train_keys = self.training_set.index
        self.test_keys = self.testing_set.index

        print(f"Training set length: {len(self.training_set)}. Testing set length: {len(self.testing_set)}")
        return self.training_set, self.testing_set

    def export_train_to_pickle(self, filename):
        """
        Export the training set to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.training_set.to_pickle(filename)
        print(f'Training set has been saved to {filename}')
    def export_test_to_pickle(self, filename):
        """
        Export the testing set to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.testing_set.to_pickle(filename)
        print(f'Testing set has been saved to {filename}')
    def get_training_set(self):
        """
        Get the training set.

        :return: DataFrame containing the training set.
        """
        return self.training_set
    def get_testing_set(self):
        """
        Get the testing set.

        :return: DataFrame containing the testing set.
        """
        return self.testing_set
    def set_training_set(self, training_set):
        """
        Set the training set.

        :param training_set: DataFrame containing the training set.
        """
        self.training_set = training_set
        self.train_keys= self.training_set.index
    def set_testing_set(self, testing_set):
        """
        Set the testing set.

        :param testing_set: DataFrame containing the testing set.
        """
        self.testing_set = testing_set
    def train_model(self, filename, weight=0.3, batch_size=500, sample_weights=None):
        """
        Train the model.
        """
        self.model.fit_from_file(filename, self.train_keys, weight=weight, batch_size=batch_size, energy_key="energy", progress='bar',sample_weights=sample_weights)
    def train_model_parallel(self, filename, weight=0.3, batch_size=500,num_cores=1,callback=None, position=0, sample_weights=None):
        """
        Train the model.
        """
        print("training parallel model with cores: ", num_cores)
        self.model.fit_from_file_parallel(filename, self.train_keys, weight=weight, batch_size=batch_size, energy_key="energy", num_cores=num_cores, progress='bar', sample_weights=sample_weights)
    def train_model_multiple_files(self, filenames, weight=0.3, batch_size=500, sample_weights=None):
        """
        Train the model using multiple files.
        """
        self.model.fit_from_files(filenames, self.train_keys, weight=weight, batch_size=batch_size, energy_key='energy', progress='bar',sample_weights=sample_weights)
    def train_model_multiple_files_parallel(self, filenames, weight=0.3, batch_size=500, num_cores=1,callback=None,position=0, sample_weights=None):
        """
        Train the model using multiple files.
        """
        self.model.fit_from_files_parallel(filenames, self.train_keys, weight=weight, batch_size=batch_size, energy_key='energy', num_cores=num_cores, progress='bar', sample_weights=sample_weights)
    def load_fitting_data(self, filenames, num_cores=1, use_cur=False):
        return self.model.load_tables_parallel(filenames=filenames, subset=self.train_keys, num_cores=num_cores, progress='bar', use_cur=use_cur)
    def fit_from_dataframe(self, df, weight=0.3, batch_size=500, num_cores=1, sample_weights=None, exclude_elements=None):
        self.model.fit_from_dataframe_parallel(df, self.train_keys, weight=weight, batch_size=batch_size, energy_key='energy', num_cores=num_cores, progress='bar', sample_weights=sample_weights, exclude_elements=exclude_elements)
    def fit_by_interaction(self, interaction, filenames, weight=0.3, batch_size=500, num_cores=1, sample_weights=None):
        self.model.fit_by_interaction(filenames, self.train_keys, target_interactions=interaction, weight=weight, batch_size=batch_size, energy_key='energy', num_cores=num_cores, progress='bar', sample_weights=sample_weights)
    def fine_tune_from_dataframe_blend(self, df, alpha=0.05, weight=0.3, batch_size=500, num_cores=1, sample_weights=None, exclude_elements=None):
        """
        Fine-tune the model using a DataFrame with blended data.
        """
        self.model.fine_tune_from_dataframe_blend(df, self.train_keys, alpha=alpha, weight=weight, batch_size=batch_size, energy_key='energy', num_cores=num_cores, progress='bar', sample_weights=sample_weights, exclude_elements=exclude_elements)
    def export_model(self, filename):
        """
        Export the model to a file.

        :param filename: Name of the file.
        """
        self.model.to_json(filename)
        print(f'Model has been exported to {filename}')
    def import_model(self, filename, bspline_config=None):
        """
        Import the model from a file.

        :param filename: Name of the file.
        """
        self.model = least_squares.WeightedLinearModel.from_json(filename, bspline_config=bspline_config)
        print(f'Model has been imported from {filename}')
    def get_model(self):
        """
        Get the model.

        :return: Model object.
        """
        return self.model
class ModelTrainerMultipleSteps:
    def __init__(self, data, chemical_system, bspline_config, regularizer):
        """
        Initialize the ModelTrainer with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        self.chemical_system = chemical_system
        self.bspline_config = bspline_config
        self.representation = process.BasisFeaturizer(self.bspline_config)
        self.regularizer = self.bspline_config.get_regularization_matrix(ridge_1b=regularizer['ridge_1b'],
                                                                    ridge_2b=regularizer['ridge_2b'],
                                                                    ridge_3b=regularizer['ridge_3b'],
                                                                    curvature_2b=regularizer['curvature_2b'],
                                                                    curvature_3b=regularizer['curvature_3b'])
        self.model = least_squares.WeightedLinearModel(self.bspline_config, self.regularizer)
        self.model.initialize_accumulators()
    def get_data(self):
        """
        Get the DataFrame containing the parsed data.

        :return: DataFrame containing the parsed data.
        """
        return self.data
    def set_data(self, data):
        """
        Set a new DataFrame containing parsed data.
        """
        self.data = data
    def get_bspline_config(self):
        """
        Get the bspline configuration.
        """
        return self.bspline_config
    def set_bspline_config(self, bspline_config):
        """
        Set the bspline configuration.
        """
        self.bspline_config = bspline_config
    def get_maps(self):
        """
        Get the maps.
        """
        return self.bspline_config
    def set_maps(self, maps):
        """
        Set the maps.
        """
        self.bspline_config = maps
    def get_chemical_system(self):
        """
        Get the chemical system.
        """
        return self.chemical_system
    def set_chemical_system(self, chemical_system):
        """
        Set the chemical system.
        """
        self.chemical_system = chemical_system
    def get_element_list(self):
        """
        Get the list of elements in the chemical system.
        """
        return self.chemical_system.elements
    def get_unique_indices(self):
        """
        Get the unique indices in the DataFrame.

        :return: List of unique indices.
        """
        return self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).unique()
    def split_data(self, test_percent, filter_index=True):
        """
        Split the data into training and testing sets.

        Parameters:
        - test_percent: Fraction of data to use for testing.
        - filter_index: Boolean flag to determine whether or not to require indices
        to be fully in either train or test.
        """
        # Get unique index prefixes
        index_prefixes = self.get_unique_indices()
        
        np.random.seed(42)  # for reproducibility
        num_indices_to_select = int(test_percent * len(index_prefixes))
        selected_prefixes = np.random.choice(index_prefixes, num_indices_to_select, replace=False)

        if filter_index:
            # Filter data for testing set based on the selected prefixes
            self.testing_set = self.data[self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).isin(selected_prefixes)]

            # Filter data for training set
            self.training_set = self.data[~self.data.index.isin(self.testing_set.index)]
        else:
            # Shuffle and split data without filtering based on prefixes
            shuffled_data = self.data.sample(frac=1, random_state=42)
            split_point = int(len(shuffled_data) * (1 - test_percent))
            self.training_set = shuffled_data.iloc[:split_point]
            self.testing_set = shuffled_data.iloc[split_point:]

        # Store the indices for training and testing sets
        self.train_keys = self.training_set.index
        self.test_keys = self.testing_set.index

        print(f"Training set length: {len(self.training_set)}. Testing set length: {len(self.testing_set)}")
        return self.training_set, self.testing_set

    def export_train_to_pickle(self, filename):
        """
        Export the training set to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.training_set.to_pickle(filename)
        print(f'Training set has been saved to {filename}')
    def export_test_to_pickle(self, filename):
        """
        Export the testing set to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.testing_set.to_pickle(filename)
        print(f'Testing set has been saved to {filename}')
    def get_training_set(self):
        """
        Get the training set.

        :return: DataFrame containing the training set.
        """
        return self.training_set
    def get_testing_set(self):
        """
        Get the testing set.

        :return: DataFrame containing the testing set.
        """
        return self.testing_set
    def set_training_set(self, training_set):
        """
        Set the training set.

        :param training_set: DataFrame containing the training set.
        """
        self.training_set = training_set
        self.train_keys= self.training_set.index
    def set_testing_set(self, testing_set):
        """
        Set the testing set.

        :param testing_set: DataFrame containing the testing set.
        """
        self.testing_set = testing_set
    def train_model(self, filename, weight=0.3, batch_size=2500):
        """
        Train the model.
        """
        sample_weights = {}
        for key in self.train_keys:
            sample_weights[key] = weight
        self.model.model.accumulate_gram_from_file(filename, self.train_keys, sample_weights=sample_weights, batch_size=batch_size, energy_key="energy", progress='bar')
    def train_model_multiple_files(self, filenames, weight=0.3, batch_size=2500):
        """
        Train the model using multiple files.
        """
        self.model.fit_from_files(filenames, self.train_keys, weight=weight, batch_size=batch_size, energy_key='energy', progress='bar')
    def finalize_fit(self, weight=0.3):
        self.model.finalize_fit_from_accumulator(weight=weight)
    def export_model(self, filename):
        """
        Export the model to a file.

        :param filename: Name of the file.
        """
        self.model.to_json(filename)
        print(f'Model has been exported to {filename}')
    def import_model(self, filename):
        """
        Import the model from a file.

        :param filename: Name of the file.
        """
        self.model = least_squares.WeightedLinearModel.from_json(filename)
        print(f'Model has been imported from {filename}')
    def get_model(self):
        """
        Get the model.

        :return: Model object.
        """
        return self.model        
class ModelAnalyzer:
    def __init__(self, chemical_system, model, bspline_config, training_data, testing_data, feature_filenames, dft_validation_folder = None, r_cut = 10, bins = 0.01):
        """
        Initialize the ModelAnalyzer with a chemical system, model, and data.

        :param chemical_system: ChemicalSystem object.
        :param model: Model object.
        :param data: DataFrame containing parsed data.
        """
        self.chemical_system = chemical_system
        self.model = model
        self.training_data = training_data
        self.training_keys = training_data.index
        self.testing_data = testing_data
        self.testing_keys = testing_data.index
        self.feature_filenames = feature_filenames
        self.dft_validation_folder = dft_validation_folder
        self.dft_validation_structures = []
        self.lammps_validation_structures = []
        self.single_crystal_validation_structures = []
        self.r_cut = r_cut
        self.bins = bins
        self.bspline_config = bspline_config
    def set_dft_validation_structures(self, dft_validation_structures):
        """
        Set the DFT validation structures.

        :param dft_validation_structures: List of DFT validation structures.
        """
        self.dft_validation_structures = dft_validation_structures
    def set_lammps_validation_structures(self, lammps_validation_structures):
        """
        Set the LAMMPS validation structures.

        :param lammps_validation_structures: List of LAMMPS validation structures.
        """
        self.lammps_validation_structures = lammps_validation_structures
    def set_single_crystal_validation_structures(self, single_crystal_validation_structures):
        """
        Set the single crystal validation structures.

        :param single_crystal_validation_structures: List of single crystal validation structures.
        """
        self.single_crystal_validation_structures = single_crystal_validation_structures
    def analyze_model(self, data_coordinator, r_cut = 10.0, bins = 0.01, training = False):
        """
        Analyze the model.
        """
        self.analyzer = analyze.DataAnalyzer(self.chemical_system, r_cut, bins)
        self.atoms_key = data_coordinator.atoms_key
        histogram_slice = np.random.choice(np.arange(len(self.training_data)),
                                   min(1000, len(self.training_data)),
                                   replace=False)
        df_slice = self.training_data[self.atoms_key].iloc[histogram_slice]
        self.analyzer.load_entries(df_slice)
        analysis = self.analyzer.analyze()
        for element_list in self.chemical_system.interactions_map[2]:
            self.analyze_2b(element_list, analysis)
        for element_list in self.chemical_system.interactions_map[3]:
            self.analyze_3b(element_list, self.model)
        self.generate_density_plot(analysis)
        results= self.test_model_prediction(training = training, use_elements= self.chemical_system.element_list)
        #results = self.export_prediction_data_to_dataframe()
        self.generate_parity_plots()
        return results
    def generate_density_plot(self, analysis):
        canvases = plotting.plot_pair_distributions(analysis, show_cutoffs=False)
        for fig, ax in canvases:
            fig.set_figwidth(6)
            fig.set_figheight(6)
            fig.set_dpi(120)
            fig.tight_layout()
            ax.set_ylim(0,5) # Set the y-axis limit to 5 
            ax.set_xlim(0,5) # Set the x-axis limit to 5 
    def fix_repulsion_2b(self, pair, r_target, min_curvature=0.0):
        """
        Fix the repulsion for a 2-body interaction.
        """
        self.model.fix_repulsion_2b(pair, r_target, min_curvature=min_curvature)
    def analyze_2b(self, element_list, analysis):
        """
        Analyze the 2-body interactions.
        """
        print(f"Analyzing 2-body interactions for {element_list}")
        pair = element_list
        r_target = analysis["lower_bounds"][pair]
        self.model.fix_repulsion_2b(pair, r_target, min_curvature=0.0)
        solutions = least_squares.arrange_coefficients(self.model.coefficients, self.bspline_config)
        coefficients = solutions[pair]
        knot_sequence = self.bspline_config.knots_map[pair]
        fig, ax = plotting.visualize_splines(coefficients, knot_sequence)
        plt.vlines([0.0], -100, 100, color="orange", linewidth=2)
        ax.set_ylim(-0.8, 0.4)
        ax.set_xlim(0,6)
        plt.show()
        return solutions
    def analyze_3b(self, element_list, model):
        #get all combinations of 3 body terms in element list 
        print(f"Analyzing 3-body interactions for {element_list}")
        tbp = plotting3d.ThreeBodyPlotter(model, element_list)
        tbp.sample_uniformly(20)
        plot = tbp.plot_uniform(val_limit=0.25)
        plot.show()
            
        
    def test_model_prediction(self, training=False, drop_columns=None, use_elements = None):
        """
        Test the model prediction.
        """
        keys = self.testing_keys if training is False else self.training_keys
        
        if(len(self.feature_filenames) > 1):
            y_e, p_e, y_f, p_f, rmse_e, rmse_f, ids, force_ids = self.model.batched_predict_multiple_files(self.feature_filenames, keys=keys, drop_columns=drop_columns, use_elements = use_elements)
        else:
            y_e, p_e, y_f, p_f, rmse_e, rmse_f, ids, force_ids = self.model.batched_predict(self.feature_filenames[0], keys=keys, drop_columns=drop_columns, use_elements=use_elements)

        sigma_e = np.std(y_e, ddof=1)  # Standard deviation of energy
        sigma_f = np.std(y_f, ddof=1)  # Standard deviation of force
        # Store results in a dictionary
        self.prediction_data = {
            "y_e": y_e,
            "p_e": p_e,
            "y_f": y_f,
            "p_f": p_f,
            "rmse_e": rmse_e,
            "rmse_f": rmse_f,
            "sigma_e": sigma_e,
            "sigma_f": sigma_f,
            "ids": ids,
            "force_ids": force_ids
        }
        # print("test model prediction ids:", ids)
        # print("test model prediction force ids:", force_ids)
        

        return self.prediction_data
    def export_prediction_data_to_dataframe(self):
        """
        Converts the prediction data dictionary into Pandas DataFrames and returns them.

        Parameters:
            prediction_data (dict): Dictionary containing model prediction data with keys 
                                ['y_e', 'p_e', 'y_f', 'p_f', 'rmse_e', 'rmse_f'].

        Returns:
            dict: A dictionary with keys as DataFrame names and values as DataFrame objects.
        """
        # Create DataFrames for each relevant key
        results = {}

        # Energies Data
        if "y_e" in self.prediction_data and "p_e" in self.prediction_data:
            df_energies = pd.DataFrame({
                "True Energies (y_e)": self.prediction_data["y_e"],
                "Predicted Energies (p_e)": self.prediction_data["p_e"]
            })
            results["energies"] = df_energies

        # Forces Data
        if "y_f" in self.prediction_data and "p_f" in self.prediction_data:
            df_forces = pd.DataFrame({
                "True Forces (y_f)": self.prediction_data["y_f"],
                "Predicted Forces (p_f)": self.prediction_data["p_f"]
            })
            results["forces"] = df_forces

        # RMSE Values
        if "rmse_e" in self.prediction_data and "rmse_f" in self.prediction_data:
            df_rmse = pd.DataFrame({
                "RMSE Energies (rmse_e)": [self.prediction_data["rmse_e"]],
                "RMSE Forces (rmse_f)": [self.prediction_data["rmse_f"]]
            })
            results["rmse"] = df_rmse

        return results
    def find_problematic_data(self, energy_cutoff, force_cutoff):
        """
        Returns the indices of the problematic data points.
        """
        #Energy #normalized error
        energy_error = np.abs(self.prediction_data["y_e"] - self.prediction_data["p_e"]) / np.std(self.prediction_data["y_e"])
        #force normalized error
        force_error = np.abs(self.prediction_data["y_f"] - self.prediction_data["p_f"]) / np.std(self.prediction_data["y_f"])
        # Find the indices of the problematic data points
        problematic_energy = np.where(energy_error > energy_cutoff)[0]
        problematic_force = np.where(force_error > force_cutoff)[0]
        #get the ids of the problematic data points
        problematic_energy_ids = [self.prediction_data["ids"][i] for i in problematic_energy]
        problematic_force_ids = [self.prediction_data["force_ids"][i] for i in problematic_force]
        #make a combined data frame of id and error for each data type
        problematic_energy_data = pd.DataFrame({
            "ID": problematic_energy_ids,
            "Error": energy_error[problematic_energy]
        })
        problematic_force_data = pd.DataFrame({
            "ID": problematic_force_ids,
            "Error": force_error[problematic_force]
        })
        #get unique ids
        problematic_energy_ids = list(set(problematic_energy_ids))
        problematic_force_ids = list(set(problematic_force_ids))
        return problematic_energy_ids, problematic_force_ids, problematic_energy_data, problematic_force_data
    def generate_parity_plots(self):
        """
        Generate parity plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Assuming `density_scatter` is a standalone function you have defined
        plotting.density_scatter(self.prediction_data["y_e"], self.prediction_data["p_e"], ax=ax[0], alpha=0.5)
        ax[0].set_xlabel("DFT Energy (eV)")
        ax[0].set_ylabel("Predicted Energy (eV)")
        ax[0].set_title("Energy Parity Plot")

        plotting.density_scatter(self.prediction_data["y_f"], self.prediction_data["p_f"], ax=ax[1], alpha=0.5)
        ax[1].set_xlabel("DFT Force (eV/A)")
        ax[1].set_ylabel("Predicted Force (eV/A)")
        ax[1].set_title("Force Parity Plot")
        ax[1].set_xlim(-8, 8)
        ax[1].set_ylim(-8, 8)

        plt.tight_layout()
        plt.show()
    def export_model_to_lammps(self, directory, filename=None, overwrite=True, author="author", units="metal", knots_spacing_type="nk"):
        """
        Export the model to a LAMMPS potential file using the UF3LammpsPotentialGenerator.

        :param directory: Directory where the LAMMPS potential file will be saved.
        :param author: Author name for the LAMMPS potential file.
        :param units: Units for the LAMMPS potential file.
        :param knots_spacing_type: Knot spacing type, 'uk' (uniform) or 'nk' (non-uniform).
        """
        generator = UF3LammpsPotentialGenerator(self.model, self.chemical_system, self.bspline_config)
        generator.generate_potential_file(directory=directory, filename=filename, overwrite=overwrite, author=author, units=units, knots_spacing_type=knots_spacing_type)
    def export_model_to_json(self, directory, filename):
        """
        Export the model to a JSON file.

        :param directory: Directory where the JSON file will be saved.
        """
        self.model.to_json(os.path.join(directory, filename))
        print(f'Model has been exported to {os.path.join(directory, filename)}')
    def set_up_folder_structure(self):
        current_directory = os.getcwd()
        validation_folder_path = os.path.join(current_directory, "validation")
        print("Validation folder path setup:", validation_folder_path)
        if not os.path.exists(validation_folder_path):
            os.makedirs(validation_folder_path)
        structures_folder_path = os.path.join(validation_folder_path, "structures")
        print("Structures folder path setup:", structures_folder_path)
        if not os.path.exists(structures_folder_path):
            os.makedirs(structures_folder_path)
        else:
            #make dft and lammps folders within every structure folder
            for structure in self.dft_validation_structures:
                structure_folder_path = os.path.join(structures_folder_path, structure)
                if os.path.isdir(structure_folder_path):
                    dft_folder_path = os.path.join(structure_folder_path, "dft")
                    if not os.path.exists(dft_folder_path):
                        os.makedirs(dft_folder_path)
                    lammps_folder_path = os.path.join(structure_folder_path, "lammps")
                    if not os.path.exists(lammps_folder_path):
                        os.makedirs(lammps_folder_path)
        print("returning from folder structure setup")
        return validation_folder_path
    def retrieve_model_files(self):
        print("retrieve model files")
        if self.dft_validation_structures:
            for structure in self.dft_validation_structures:
                #get the mp id from the structure name (its in format POSCAR_mpid)
                print("downloading structure: ", structure)
                mpid = structure
                directory = os.path.join("validation/structures/", structure)
                print("directory: ", directory)
                structure_downloader = StructureDownloader(MP_API_KEY)
                print("structure downloader object")
                structure_downloader.download_single_structure(mpid, directory)
        print("returning from retrieve model files")
                
    def validate_potential_by_name(self, potential_version, potential_file, element_list, test_name):
        current_directory = os.getcwd()
        datas = []
        print("Printing validation structures")
        
        for structure in self.dft_validation_structures:
            print(structure)
        print("Done printing the structures")
        
        for structure in self.dft_validation_structures:
            if "POSCAR" in structure:
                print("Contains POSCAR")
                continue
            if "LAMMPS" in structure:
                print("Contains LAMMPS")
                continue
            
            # Extract the Materials Project ID from the structure name
            print("Processing structure:", structure)
            structure_id = structure.replace("mp-", "")
            
            # Define paths for LAMMPS and DFT files
            validation_folder_path = os.path.join(current_directory, "validation/structures", structure)
            print("Validation folder path:", validation_folder_path)
            
            lammps_file = os.path.join(validation_folder_path, f"LAMMPS_{structure_id}.data")
            dft_file = os.path.join(validation_folder_path, f"POSCAR_{structure_id}")
            
            # Validate that the files exist
            if not os.path.isfile(lammps_file):
                print(f"LAMMPS file not found for structure {structure_id}: {lammps_file}")
                continue
            if not os.path.isfile(dft_file):
                print(f"DFT file not found for structure {structure_id}: {dft_file}")
                continue
            
            # Create the element list as a space-separated string
            element_list_str = " ".join(element_list)
            print("about to initialize validationData")
            # Initialize ValidationData
            data = ValidationData(potential_version, potential_file, structure_id)
            print("initialized validation data")
            # Dictionary to map test names to validation classes
            test_mapping = {
                "lattice_constant": LatticeConstantValidation,
                "cohesive_energy": CohesiveEnergyValidation,
                "elastic_tensor": ElasticTensorValidation,
                "surface_energy": SurfaceEnergyValidation
            }
            print('test mapping: ', test_name)
            print(test_mapping)
            if test_name not in test_mapping:
                print(f"Invalid test name: {test_name}")
                return None
            # Execute the specified test
            print(f"Running test: {test_mapping[test_name]}")
            validation_class = test_mapping[test_name]
            validation_instance = validation_class(potential_file, lammps_file, dft_file, element_list_str, validation_folder_path)
            validation_instance.initialize_simulation_files()
            # Map test names to their respective DFT result paths
            dft_paths = {
                "lattice_constant": os.path.join(validation_folder_path, "dft/simulations/RelaxationSimulation/"),
                "cohesive_energy": os.path.join(validation_folder_path, "dft/simulations/CohesiveEnergySimulation/results.txt"),
                "elastic_tensor": os.path.join(validation_folder_path, "dft/simulations/ElasticTensor/results.txt"),
                "surface_energy": os.path.join(validation_folder_path, "dft/simulations/SurfaceEnergySimulation/")
            }
            print('dft paths')
            
            validation_instance.retrieve_dft_result(test_name, test_name, dft_paths[test_name])
            print('dft result retrieved')
            percent_error = validation_instance.start_validation(overwrite=True, local=False)
            print('validation run')
            data.add_validation(test_name, percent_error)
            datas.append(data)
            
            print(f"Validation data for {test_name}: ", data)
            
            # Return to the initial directory after processing each structure
            os.chdir(current_directory)

        return datas
    def calculate_normalized_error(self, prediction_data):
        rmse_e = prediction_data["rmse_e"]
        rmse_f = prediction_data["rmse_f"]
        std_e = np.std(self.prediction_data["y_e"])
        std_f = np.std(self.prediction_data["y_f"])
        # Calculate normalized RMSE
        normalized_rmse = (rmse_e / std_e + rmse_f / std_f) / 2
        return normalized_rmse
    
    def validate_potential(self, potential_version, potential_file, element_list, rmse=False, validations_to_run=None, relative_orders=None):
        """
        Validate the potential dynamically using LAMMPS and DFT files 
        identified from the validation folder structure and file names.

        Args:
            potential_version: Version of the potential to validate.
            potential_file: Path to the potential file.
            element_list: List of elements involved in the validation.
            validations_to_run: Dictionary indicating which validations to run (True/False).
                                e.g., {"lattice_constant": True, "cohesive_energy": False, ...}
        """
        print("validate potential entry1")
        print('validate potential element list:', element_list)
        print("validate potential dft validation structures: ", self.dft_validation_structures)
        current_directory = os.getcwd()
        datas = []
        if validations_to_run is None:
            validations_to_run = {
                "lattice_constant": True,
                "cohesive_energy": True,
                "elastic_tensor": True,
                "surface_energy": True
            }
            
            
        def process_structure(structure, single_crystal_data=None):
            print("process structure started AnalyzerValidatePotential")
            if "POSCAR" in structure or "LAMMPS" in structure:
                print(f"Skipping structure containing POSCAR or LAMMPS: {structure}")
                return None

            structure_id = structure.replace("mp-", "")
            print("process structure AnalyzerValidatePotential")
            validation_folder_path = os.path.join(current_directory, "validation/structures", structure)
            if self.dft_validation_structures:
                dft_validation_folder = os.path.join(self.dft_validation_folder, structure)
            else:
                dft_validation_folder = validation_folder_path
            lammps_file = os.path.join(validation_folder_path, f"LAMMPS_{structure_id}.data")
            dft_file = os.path.join(validation_folder_path, f"POSCAR_{structure_id}")

            if not os.path.isfile(lammps_file) or not os.path.isfile(dft_file):
                print(f"Files missing for structure {structure_id}: {lammps_file}, {dft_file}")
                return None

            element_list_str = " ".join(element_list)
            print("abotu to make validationmanager AnalyzerValidatePotential")
            manager = ValidationManager(
                potential_version,
                structure_id,
                potential_file,
                lammps_file,
                dft_file,
                element_list_str,
                validation_folder_path,
                dft_validation_folder,
                validations_to_run
            )
            if(single_crystal_data is not None):
                manager.set_single_crystal_data(single_crystal_data)
            print("createdf validation manager AnalyzerValidatePotential")
            data = manager.run_validation()
            print("validation manager finishesd")
            if rmse == True:
                print("calculating normalized error")
                prediction_data = self.test_model_prediction(use_elements=element_list)
                data.set_force_rmse(prediction_data["rmse_f"])
                data.set_energy_rmse(prediction_data["rmse_e"])
                normalized_error = self.calculate_normalized_error(prediction_data)
            else:
                normalized_error = None
            data.set_normalized_error(normalized_error)
            print("process structure finished")
            return data
        single_crystal_data = None
        if 'formation_energy' in validations_to_run:
            for structure in self.single_crystal_validation_structures:
                structure_id = structure.replace("mp-", "")
                validation_folder_path = os.path.join(current_directory, "validation/structures", structure)
                lammps_file = os.path.join(validation_folder_path, f"LAMMPS_{structure_id}.data")
                dft_file = os.path.join(validation_folder_path, f"POSCAR_{structure_id}")

                if not os.path.isfile(lammps_file) or not os.path.isfile(dft_file):
                    print(f"Files missing for structure {structure_id}: {lammps_file}, {dft_file}")
                    return None

                element_list_str = " ".join(element_list)
                
                manager = SingleCrystalValidationManager(
                    potential_version,
                    structure_id,
                    potential_file,
                    lammps_file,
                    dft_file,
                    element_list_str,
                    validation_folder_path,
                    validations_to_run
                )

                single_crystal_data = manager.run_validation()
                print("Single crystal validation data:", single_crystal_data)

        print("about to give workers processstructure - analyzerValidatePotential")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_structure = {
                executor.submit(process_structure, s, single_crystal_data): s for s in self.dft_validation_structures
            }
            for future in as_completed(future_to_structure):
                result = future.result()
                if result:
                    datas.append(result)
                    print("structure processed")

        relative_order_result = {}

        if relative_orders is not None and len(datas) > 1:
            for validation, expected_order in relative_orders.items():
                print("Testing validation:", validation)
                lammps_results = {}
                print("relative order datas:", datas)
                for data in datas:
                    print("relative order data")
                    print(data)
                    validation_results = data.get_lammps_validation_results()
                    print("relative order validation_results:")
                    print(validation_results)
                    if validation in validation_results:
                        lammps_results[data.get_structure_id()] = validation_results[validation][validation]
                        print("relative order Validation data:", validation_results[validation][validation])

                # Sort structure IDs based on LAMMPS energy values (ascending order)
                sorted_lammps = sorted(lammps_results, key=lammps_results.get)
                print("Sorted LAMMPS:", sorted_lammps)

                # Compare sorted order with expected relative order
                if expected_order != sorted_lammps:
                    print(f"Relative order does not match for {validation}")
                    relative_order_result[validation] = False
                else:
                    print(f"Relative order matches for {validation}")
                    relative_order_result[validation] = True

            # Store the relative order results back into each data object
            for data in datas:
                data.set_relative_order_results(relative_order_result)
        print("ModelAnalyzer returning validate_potential")
        return datas


from datetime import datetime     
class UF3LammpsPotentialGenerator:
    def __init__(self, model, chemical_system, bspline_config):
        self.model = model
        self.chemical_system = chemical_system
        self.bspline_config = bspline_config

    def export_model_to_json(self, filename):
        self.model.to_json(filename)

    def generate_potential_file(self, directory, filename=None, overwrite=True,
                                author="author", units="metal", knots_spacing_type="nk"):
        if not os.path.exists(directory):
            os.mkdir(directory)

        if filename is None:
            filename = "".join(self.chemical_system.element_list) + ".uf3"

        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path) and not overwrite:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename += f"_{now}"
            full_path = os.path.join(directory, filename)

        self._write_uf3_lammps_pot_files(
            pot_dir=directory,
            uf3_lammps_pot_name=filename,
            author=author,
            lammps_units=units,
            knots_spacing_type=knots_spacing_type
        )

        # Example lines for LAMMPS
        pair_style_line = f"pair_style uf3 {self.bspline_config.degree} {len(self.chemical_system.element_list)}"
        pair_coeff_line = f"pair_coeff * * {full_path} " + " ".join(self.chemical_system.element_list)

    import re

    def potential_to_model_json(self, uf3_file_path, output_json_path, potential_version):
        with open(uf3_file_path, "r") as f:
            lines = f.readlines()

        # Parse regularizer values from the potential_version string
        reg_fields = {
            'ridge_1b': r'_r1b_([-+e0-9.]+)',
            'ridge_2b': r'_r2b_([-+e0-9.]+)',
            'ridge_3b': r'_r3b_([-+e0-9.]+)',
            'curvature_2b': r'_curv2b_([-+e0-9.]+)',
            'curvature_3b': r'_curv3b_([-+e0-9.]+)'
        }
        
        print("updated bspline config: ", self.bspline_config)
        regularizer_values = {}
        for key, pattern in reg_fields.items():
            match = re.search(pattern, potential_version)
            if not match:
                raise ValueError(f"Could not find {key} in potential_version string")
            regularizer_values[key] = float(match.group(1))

        config = {
        "r_max_map": self.bspline_config.r_max_map.copy(),
        "knots_map": self.bspline_config.knots_map.copy(),
        "leading_trim": self.bspline_config.leading_trim.copy(),
        "trailing_trim": self.bspline_config.trailing_trim.copy(),
        #"regularizer": self.bspline_config.get_regularization_matrix(**regularizer_values),
        "data_coverage": None
    }
        print("config bspline: ", self.bspline_config)
        coeffs = []
        i = 0
        self.expected_shapes = {}
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            print(line)
            tokens = line.split()
            if tokens[0] == "2B":
                key = tuple(tokens[1:3])
                config["leading_trim"][2] = int(tokens[3])
                config["trailing_trim"][2] = int(tokens[4])
                i += 1
                print(lines[i])
                rmax, _ = map(float, lines[i].strip().split())
                config["r_max_map"][key] = rmax
                i += 1
                print(lines[i])
                knots = list(map(float, lines[i].strip().split()))
                config["knots_map"][key] = np.array(knots)
                i += 1
                print(lines[i])
                nbasis = int(lines[i].strip())
                i += 1
                coeff_line = lines[i].strip().split()
                self.expected_shapes[key] = nbasis
                if len(coeff_line) != nbasis:
                    raise ValueError(f"2B: expected {nbasis} coeffs, got {len(coeff_line)} at line {i}")
                coeffs.extend(map(float, coeff_line))
                i += 2
                print(lines[i])

            elif tokens[0] == "3B":
                print(lines[i])
                key = tuple(tokens[1:4])
                config["leading_trim"][3] = int(tokens[4])
                config["trailing_trim"][3] = int(tokens[5])
                i += 1
                print(lines[i])
                rvals = list(map(float, lines[i].strip().split()))
                config["r_max_map"][key] = [ rvals[0], rvals[1], rvals[2]]
                i += 1
                print(lines[i])
                k2 = list(map(float, lines[i].strip().split())); i += 1
                k1 = list(map(float, lines[i].strip().split())); i += 1
                k0 = list(map(float, lines[i].strip().split())); i += 1
                #sort by order
                if len(k2) > len(k0):
                    k2, k1, k0 = k0, k1, k2
                config["knots_map"][key] = [ np.array(k2), np.array(k1), np.array(k0)]
                print(lines[i])
                shape = tuple(map(int, lines[i].strip().split()))
                i += 1

                expected_rows = shape[0] * shape[1]
                expected_cols = shape[2]
                self.expected_shapes[key] = shape
                print(key)
                print(f"3B shape: {shape}, expected_rows: {expected_rows}, expected_cols: {expected_cols}")
                
                for row_idx in range(expected_rows):
                    row = list(map(float, lines[i].strip().split()))
                    if len(row) != expected_cols:
                        raise ValueError(f"3B: line {i} has {len(row)} coeffs, expected {expected_cols}")
                    coeffs.extend(row)
                    print(f"3B coeffs row {row_idx}: {row}")
                    i += 1

                i += 1  # skip "#"
            else:
                i += 1
        print("coeffs shape: ", len(coeffs))
        config["solution"] = {
            "coefficients": np.array(coeffs)
        }
        # Force symmetry = 1 for all 3B interactions
        old_symmetry = {}
        for trio in self.bspline_config.interactions_map.get(3, []):
            old_symmetry[trio] = self.bspline_config.symmetry[trio]
            self.bspline_config.symmetry[trio] = 1
            
        self.bspline_config.update_basis_functions()

        # Expand 3B knot arrays so symmetry won't be inferred on reload
        # unsym_knots_map = {}
        # for k, v in config["knots_map"].items():
        #     if isinstance(k, tuple) and len(k) == 3:
        #         if isinstance(v, np.ndarray):
        #             unsym_knots_map[k] = [v.copy(), v.copy(), v.copy()]
        #         elif isinstance(v, list) and len(v) == 2:
        #             unsym_knots_map[k] = [v[0].copy(), v[1].copy(), v[1].copy()]
        #         else:
        #             unsym_knots_map[k] = [v[0].copy(), v[1].copy(), v[2].copy()]
        #     else:
        #         unsym_knots_map[k] = v.copy()
        # config["knots_map"] = unsym_knots_map
        partitions = self.bspline_config.get_interaction_partitions()
        print(partitions)
        print("partitions 0: ", partitions[0])
        expected_total = sum(partitions[0].values())
        coeff_map = {}
        coef_array = np.array(coeffs)
        for trio in self.bspline_config.interactions_map.get(3, []):
            
            test = self.bspline_config.compress_3B(coef_array, trio)
            print(f"3B test for {trio}: {len(test)}")
        component_sizes, component_offsets = self.bspline_config.get_interaction_partitions(uncompressed=True)
        #prepend coef_array with two ones
        for key,count in partitions[0].items():
            if isinstance(key, str):
                coef_array = np.concatenate(([1.0], coef_array))
        

        arranged = least_squares.arrange_coefficients(coef_array, self.bspline_config)
        for trio in self.bspline_config.interactions_map.get(3, []):
            print(f'len pre-compress: {len(arranged[trio])} ')
            # arranged[trio] = self.bspline_config.compress_3B(arranged[trio], trio)
            # print(f"3B test 2 for {trio}: {len(arranged[trio])}")
        start_index = 0
        for key, count in partitions[0].items():
            start = partitions[1][key] - start_index
            stop = start + count
            sub_coeffs = coef_array[start:stop]
            
            if isinstance(key, str):
                coeff_map[key] = 1
            elif len(key) == 2:  # 2B
                coeff_map["-".join(key)] = sub_coeffs.tolist()
            elif len(key) == 3:
                shape = self.expected_shapes[key]
                coeff_block = np.array(arranged[key])
                if coeff_block.size != np.prod(shape):
                    raise ValueError(f"Unexpected coeff size for {key}: {coeff_block.size}, expected {np.prod(shape)}")
                
                reshaped = coeff_block.reshape(shape)
                coeff_map["-".join(key)] = [reshaped[i].tolist() for i in range(shape[0])]
        component_sizes, component_offsets = self.bspline_config.get_interaction_partitions(uncompressed=False)
        print(encode_interaction_map(config["knots_map"]))
        print(encode_interaction_map(coeff_map))
        print(len(encode_interaction_map(coeff_map)))
        model_json = {
            "chemical_system": self.chemical_system.as_dict(),
            "element_list": self.chemical_system.element_list,
            "degree": 3,
            "r_max_map": encode_interaction_map(config["r_max_map"]),
            "knots_map": encode_interaction_map(config["knots_map"]),
            "knots": encode_interaction_map(config['knots_map']),
            "leading_trim": {str(k): v for k, v in config["leading_trim"].items()},
            "trailing_trim": {str(k): v for k, v in config["trailing_trim"].items()},
            "regularizer": None,
            "data_coverage": None,
            "coefficients": encode_interaction_map(coeff_map)
        }

        with open(output_json_path, "w") as f:
            json.dump(model_json, f, indent=2, cls=CompactJSONEncoder)


    def _write_uf3_lammps_pot_files(self, pot_dir, uf3_lammps_pot_name, author, lammps_units, knots_spacing_type):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        files = {}

        # Grab integer trims for 2B and 3B
        leading_trim_2B = self.bspline_config.leading_trim.get(2, 0)
        trailing_trim_2B = self.bspline_config.trailing_trim.get(2, 0)
        leading_trim_3B = self.bspline_config.leading_trim.get(3, 0)
        trailing_trim_3B = self.bspline_config.trailing_trim.get(3, 0)
        #sum up the interactions
        interaction_sum = 0
        for interaction in self.bspline_config.get_interaction_partitions()[0]:
            interaction_sum += self.bspline_config.get_interaction_partitions()[0][interaction]
        print("interaction sum: ", interaction_sum)
        if interaction_sum > len(self.model.coefficients):
            for trio in self.bspline_config.interactions_map.get(3, []):
                r_min = self.bspline_config.r_min_map[trio]
                r_max = self.bspline_config.r_max_map[trio]
                res = self.bspline_config.resolution_map[trio]
                self.bspline_config.symmetry[trio] = bspline.find_symmetry_3B(trio, r_min, r_max, res)
                print(f'Symmetry for {trio} set to {self.bspline_config.symmetry[trio]}')
        self.bspline_config.update_basis_functions()
        interaction_sum = 0
        for interaction in self.bspline_config.get_interaction_partitions()[0]:
            interaction_sum += self.bspline_config.get_interaction_partitions()[0][interaction]
        print("interaction sum: ", interaction_sum)
        if interaction_sum != len(self.model.coefficients):
            raise ValueError(f"Interaction sum {interaction_sum} does not match model coefficients length {len(self.model.coefficients)}")

        # 2B
        for interaction in self.chemical_system.interactions_map[2]:
            key = "_".join(interaction)
            files[key] = (
                f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n"
            )
            files[key] += f"2B {interaction[0]} {interaction[1]} {leading_trim_2B} {trailing_trim_2B} "
            if knots_spacing_type not in ["uk", "nk"]:
                raise ValueError(f"Invalid knot spacing type: {knots_spacing_type}")
            files[key] += knots_spacing_type + "\n"

            rmax = self.bspline_config.r_max_map[interaction]
            nknots = len(self.bspline_config.knots_map[interaction])
            files[key] += f"{rmax} {nknots}\n"

            knots = self.bspline_config.knots_map[interaction]
            files[key] += " ".join(['{:.17g}'.format(v) for v in knots]) + "\n"

            partitions = self.bspline_config.get_interaction_partitions()
            num_basis = partitions[0][interaction]
            start = partitions[1][interaction]
            coeff_slice = self.model.coefficients[start:start + num_basis]
            files[key] += str(num_basis) + "\n"
            files[key] += " ".join(['{:.17g}'.format(v) for v in coeff_slice]) + "\n"
            files[key] += "#\n"

        # 3B
        reduced_coefficients = []

        if 3 in self.chemical_system.interactions_map:
            for interaction in self.chemical_system.interactions_map[3]:
                key = "_".join(interaction)
                files[key] = (
                    f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n"
                )
                files[key] += f"3B {interaction[0]} {interaction[1]} {interaction[2]} {leading_trim_3B} {trailing_trim_3B} "
                if knots_spacing_type not in ["uk", "nk"]:
                    raise ValueError(f"Invalid knot spacing type: {knots_spacing_type}")
                files[key] += knots_spacing_type + "\n"

                rmax2 = self.bspline_config.r_max_map[interaction][2]
                rmax1 = self.bspline_config.r_max_map[interaction][1]
                rmax0 = self.bspline_config.r_max_map[interaction][0]
                nknots2 = len(self.bspline_config.knots_map[interaction][2])
                nknots1 = len(self.bspline_config.knots_map[interaction][1])
                nknots0 = len(self.bspline_config.knots_map[interaction][0])
                files[key] += f"{rmax2} {rmax1} {rmax0} {nknots2} {nknots1} {nknots0}\n"

                knots2 = self.bspline_config.knots_map[interaction][2]
                knots1 = self.bspline_config.knots_map[interaction][1]
                knots0 = self.bspline_config.knots_map[interaction][0]
                files[key] += " ".join(['{:.17g}'.format(v) for v in knots2]) + "\n"
                files[key] += " ".join(['{:.17g}'.format(v) for v in knots1]) + "\n"
                files[key] += " ".join(['{:.17g}'.format(v) for v in knots0]) + "\n"
                print("coefficient len: ", len(self.model.coefficients))
                print("bspline interaction partitions: ", self.bspline_config.get_interaction_partitions())
                arranged = least_squares.arrange_coefficients(self.model.coefficients, self.bspline_config)
                decompressed = self.bspline_config.decompress_3B(arranged[interaction], interaction)
                files[key] += f"{decompressed.shape[0]} {decompressed.shape[1]} {decompressed.shape[2]}\n"
                for i in range(decompressed.shape[0]):
                    for j in range(decompressed.shape[1]):
                        files[key] += ' '.join(map(str, decompressed[i, j])) + "\n"
                files[key] += "#\n"

        full_path = os.path.join(pot_dir, uf3_lammps_pot_name)
        with open(full_path, "w") as fp:
            for content in files.values():
                fp.write(content)