import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uf3.data import io
from uf3.data import geometry
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares
from uf3.forcefield import calculator
from uf3.forcefield import lammps
from uf3.util import parallel
from uf3.util import plotting
from uf3.util import plotting3d
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import *
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry
from pymatgen.io.ase import AseAtomsAdaptor
from concurrent.futures import ProcessPoolExecutor
from uf3.data import analyze
import re

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
        Parse all OUTCAR files in the specified directories and their subdirectories,
        and store the parsed data in a DataFrame.
        """
        parsedFiles = 0
        for directory in self.directories:

            # Walk through the directory tree
            for root, dirs, files in os.walk(directory):
                
                # Get the folder name; handle root directory
                if root == directory:
                    folder_name = os.path.basename(root.rstrip('/\\')) or os.path.basename(os.path.abspath(root))
                else:
                    folder_name = os.path.basename(root)

                # Search for OUTCAR files in the current directory
                outcar_files = glob.glob(os.path.join(root, 'OUTCAR*'))
                
                for file in outcar_files:
                    if output:
                        print(f"Processing file: {file}")
                    try:
                        # Extract the run number and index from the filename
                        match = re.match(r'OUTCAR_(\d+)_(\d+)', os.path.basename(file))
                        if match:
                            run_number = match.group(1)
                            index_number = match.group(2)
                            prefix = f'ind_{folder_name}_run_{run_number}__{index_number}'
                        else:
                            # Default to use the entire file name if it doesn't match the expected pattern
                            prefix = f'ind_{folder_name}_run_{os.path.basename(file)}'
                        
                        # Use the data_coordinator to parse the file and update the data.
                        self.data_coordinator.dataframe_from_trajectory(file, prefix=prefix, filter=filter)
                        parsedFiles += 1
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        # Consolidate the data after processing all files.
        if parsedFiles > 0:
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
        print(f'DataFrame has been saved to {filename}')
    
    def load_from_pickle(self, filename):
        """
        Load the DataFrame from a pickle file.

        :param filename: Name of the pickle file.
        """
        self.data = pd.read_pickle(filename)
        print(f'DataFrame has been loaded from {filename}')
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

    
class DataSetAnalyzer:
    # Class-wide variable to hold the data type prefixes
    DATA_TYPE_PREFIXES = {
        "pressure": "p_",
        "temperature": "t_",
        "strain": "strain_",
        "defect": "defect_",
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

    def get_structure_data(self):
        """
        Get the structure data from the DataFrame.

        :return: DataFrame containing structure data.
        """
        # Exclude rows with any of the data type prefixes in the index
        pattern = '|'.join(DataSetAnalyzer.DATA_TYPE_PREFIXES.values())
        return self.data[~self.data.index.str.contains(pattern)]

    def get_data_by_type(self, data_type):
        """
        Get the data from the DataFrame based on the specified data type.

        :param data_type: Type of data to retrieve (e.g., 'pressure', 'temperature').
        :return: DataFrame containing the specified type of data.
        """
        if data_type in DataSetAnalyzer.DATA_TYPE_PREFIXES:
            prefix = DataSetAnalyzer.DATA_TYPE_PREFIXES[data_type]
            return self.data[self.data.index.str.contains(prefix)]
        else:
            raise ValueError(f"Data type '{data_type}' is not recognized. Available types: {list(DataSetAnalyzer.DATA_TYPE_PREFIXES.keys())}")

    def get_pressure_data(self):
        """
        Get the pressure data from the DataFrame.

        :return: DataFrame containing pressure data.
        """
        return self.get_data_by_type("pressure")

    def get_temperature_data(self):
        """
        Get the temperature data from the DataFrame.

        :return: DataFrame containing temperature data.
        """
        return self.get_data_by_type("temperature")

    def get_strain_data(self):
        """
        Get the strain data from the DataFrame.

        :return: DataFrame containing strain data.
        """
        return self.get_data_by_type("strain")

    def get_defect_data(self):
        """
        Get the defect data from the DataFrame.

        :return: DataFrame containing defect data.
        """
        return self.get_data_by_type("defect")
     
    def get_data_type_counts(self):
        """
        Get the counts of data types in the DataFrame.

        :return: Dictionary containing the count of each data type.
        """
        data_type_counts = {}
        for data_type in DataSetAnalyzer.DATA_TYPE_PREFIXES:
            data_type_counts[data_type] = len(self.get_data_by_type(data_type))
        return data_type_counts

    def get_data_type_proportions(self):
        """
        Get the proportions of data types in the DataFrame.

        :return: Dictionary containing the proportion of each data type.
        """
        data_type_proportions = {}
        total_data = self.get_data_length()
        data_type_counts = self.get_data_type_counts()
        for data_type, count in data_type_counts.items():
            data_type_proportions[data_type] = count / total_data
        return data_type_proportions

    def adjust_data_proportions(self, target_proportions):
        """
        Adjust the proportions of each data type in the DataFrame to match the target proportions.

        :param target_proportions: Dictionary specifying the target proportions for each data type.
                                   The values should sum to 1.0 or 100%.
        """
        # Ensure the target proportions are valid
        if abs(sum(target_proportions.values()) - 1.0) > 1e-6:
            raise ValueError("The target proportions must sum to 1.0 (or 100%).")

        # Calculate the total number of data points after adjustment
        total_data_points = self.get_data_length()

        adjusted_data = self.get_structure_data()  # Start with structure data

        for data_type, target_prop in target_proportions.items():
            target_count = int(target_prop * total_data_points)
            current_data = self.get_data_by_type(data_type)
            current_count = len(current_data)

            if current_count < target_count:
                # Duplicate data to reach the target proportion
                num_repeats = (target_count + current_count - 1) // current_count  # Ceiling division
                replicated_data = pd.concat([current_data] * num_repeats)
                adjusted_data = pd.concat([adjusted_data, replicated_data.iloc[:target_count]])
            else:
                # If the current count meets or exceeds the target, just add the data
                adjusted_data = pd.concat([adjusted_data, current_data])

        # Update the data with the adjusted DataFrame
        self.set_data(adjusted_data)
    
    def export_to_pickle(self, filename):
        """
        Export the DataFrame to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.data.to_pickle(filename)
        print(f'DataFrame has been saved to {filename}')
    
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

    def plot_composition_histogram(self, element, bins=20, alpha=0.5):
        """
        Plot a histogram of the element composition.
        """
        self.data[f'{element}_frac'].plot.hist(bins=bins, alpha=alpha)
        plt.xlabel(f'{element} fraction')
        plt.ylabel('Frequency')
        plt.title(f'{element} fraction histogram')
        plt.show()
        

    def plot_force_histogram(self, bins=20, alpha=0.5, bin_range=None):
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

    


class StructureParser:
    def __init__(self, data):
        """
        Initialize the StructureParser with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        self.data = self.transform_data(data)
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
    def transform_data(self, data):
        pde_entries = self.get_pde_entries(self.data)
        self.cpd = PhaseDiagram(pde_entries)
        self.append_phase_diagram_energy()
        self.append_energy_above_hull()
        element_list = self.get_element_list()
        print(f"Elements in phase diagram: {element_list}")
        for element in element_list:
            self.append_element_fraction(element)
            
        return self.data

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
        print(f'DataFrame has been saved to {filename}')
        
    def get_element_list(self):
        """
        Get the list of elements in the phase diagram.

        :return: List of elements in the phase diagram.
        """
        return self.cpd.elements
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
        unique_data = self.get_unique_indices()
        
        
        
        entries = []
        for i in range(len(unique_data)):
            index = unique_data[i]
             # Find the first matching index that starts with the given index prefix
            matching_index = next((x for x in self.data.index if x.startswith(f'{index}')), None)
            if matching_index is not None:
                row = self.data.loc[matching_index]
                # Return the composition for the first matching index
                comp = self.get_composition(row)
                entry = PDEntry(comp, row.energy, name=str(row.name))
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
        matching_index = next((i for i in self.data.index if i.startswith(f'{index}')), None)
        
        if matching_index is not None:
            # Return the composition for the first matching index
            return self.get_composition(self.data.loc[matching_index])
        else:
            # If no matching index is found, return None or raise an exception
            return None  # or raise an exception if that's more appropriate 
    


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
        
        
    def filter_composition(self, element, percent):
        print(f"Filtering out data with over {percent*100}% {element} composition.")
        #check if data has a fraction column for the element
        if f'{element}_frac' in self.data.columns:
            self.data = self.data[self.data[f'{element}_frac'] <= percent]
        else:
            print(f"Element {element} fraction not found in data. Please add columns for each element fraction using StructureParser.")
    
    def filter_energy_above_hull(self, threshold):
        print(f"Filtering data for energies less than {threshold}.")
        #check if data has an energy above hull column
        if 'eah' in self.data.columns:
            self.data = self.data[self.data['eah'] <= threshold]
        else:
            print(f"Energy above hull column not found in data. Please add column for energy above hull using StructureParser.")
        
    def filter_force(self, thresholdX, thresholdY, thresholdZ):
        print(f"Filtering data for forces less than {thresholdX}, {thresholdY}, {thresholdZ}.")
        print(f"Columns: {self.data.columns}")
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

    def export_to_pickle(self, filename):
        """
        Export the DataFrame to a pickle file.

        :param filename: Name of the pickle file.
        """
        self.data.to_pickle(filename)
        print(f'DataFrame has been saved to {filename}')
        

class Featurizer:
    def __init__(self, data, chemical_system, maps):
        """
        Initialize the Featurizer with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        self.chemical_system = chemical_system
        self.maps = maps
        self.bspline_config = bspline.BSplineBasis(chemical_system,
                                                   r_min_map=maps['r_min_map'], 
                                                   r_max_map=maps['r_max_map'],
                                                   resolution_map=maps['r_resolution_map'],
                                                   leading_trim=maps['leading_trim'],
                                                   trailing_trim=maps['trailing_trim'])
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
        return self.maps
    
    def set_maps(self, maps):
        """
        Set the maps.
        """
        self.maps = maps
    
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
    
    def featurize_data(self, num_cores, output_filename, batch_size=500):
        """
        Featurize the data.
        """
        # Check if the file already exists
        if os.path.exists(output_filename):
            raise FileExistsError(f"The file '{output_filename}' already exists. Featurization aborted.")
        
        table_template = "features_{}"
        client = ProcessPoolExecutor(max_workers=num_cores)
        self.representation.batched_to_hdf(output_filename,
                                           self.data,
                                           client,
                                           n_jobs=num_cores,
                                           batch_size=batch_size,  # It should be 5000 for final
                                           progress="bar",
                                           table_template=table_template)
        
class ModelTrainer:
    def __init__(self, data, chemical_system, maps, regularizer):
        """
        Initialize the ModelTrainer with a DataFrame containing parsed data.

        :param data: DataFrame containing parsed data.
        """
        self.data = data
        self.chemical_system = chemical_system
        self.maps = maps
        self.bspline_config = bspline.BSplineBasis(chemical_system,
                                                   r_min_map = maps['r_min_map'], 
                                                   r_max_map = maps['r_max_map'],
                                                   resolution_map=maps['r_resolution_map'],
                                                   leading_trim=maps['leading_trim'],
                                                   trailing_trim=maps['trailing_trim'])
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
        return self.maps
    def set_maps(self, maps):
        """
        Set the maps.
        """
        self.maps = maps
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
    def split_data(self, test_percent):
        """
        Split the data into training and testing sets.
        """
        
        index_prefixes = self.get_unique_indices()
        np.random.seed(42)  # for reproducibility
        num_indices_to_select = int(test_percent * len(index_prefixes))
        selected_prefixes = np.random.choice(index_prefixes, num_indices_to_select, replace=False)

        # Filter data for testing set based on the selected prefixes
        self.testing_set = self.data[self.data.index.to_series().apply(lambda x: '_'.join(x.split('_')[:2])).isin(selected_prefixes)]

        # Filter data for training set
        self.training_set = self.data[~self.data.index.isin(self.testing_set.index)]
        self.train_keys= self.training_set.index
        print(f"Training set length: {len(self.training_set)}. Testing set length: {len(self.testing_set)}")
        print(f"Training set columns: {self.training_set.columns}")
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
        self.model.fit_from_file(filename, self.train_keys, weight=weight, batch_size=batch_size, energy_key="energy", progress='bar')
    def train_model_multiple_files(self, filenames, weight=0.3, batch_size=2500):
        """
        Train the model using multiple files.
        """
        self.model.fit_from_files(filenames, self.train_keys, weight=weight, batch_size=batch_size, energy_key='energy', progress='bar')
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
    def __init__(self, chemical_system, model, bspline_config, training_data, testing_data, feature_filename, r_cut = 10, bins = 0.01):
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
        self.feature_filename = feature_filename
        self.r_cut = r_cut
        self.bins = bins
        self.bspline_config = bspline_config
    def analyze_model(self, r_cut = 10.0, bins = 0.01, training = False):
        """
        Analyze the model.
        """
        self.analyzer = analyze.DataAnalyzer(self.chemical_system, r_cut, bins)
        self.atoms_key = io.DataCoordinator().atoms_key
        histogram_slice = np.random.choice(np.arange(len(self.training_data)),
                                   min(1000, len(self.training_data)),
                                   replace=False)
        df_slice = self.training_data[self.atoms_key].iloc[histogram_slice]
        self.analyzer.load_entries(df_slice)
        analysis = self.analyzer.analyze()
        for element_list in self.chemical_system.interactions_map[2]:
            self.analyze_2b(element_list, analysis)
        self.test_model_prediction(training = training)
        self.generate_parity_plots()
        
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
        return solutions
    def test_model_prediction(self, training=False):
        """
        Test the model prediction.
        """
        if training:
            keys = self.training_keys
        else:
            keys = self.testing_keys

        y_e, p_e, y_f, p_f, rmse_e, rmse_f = self.model.batched_predict(self.feature_filename, 
                                                                         keys=keys)

        # Store results in a dictionary
        self.prediction_data = {
            "y_e": y_e,
            "p_e": p_e,
            "y_f": y_f,
            "p_f": p_f,
            "rmse_e": rmse_e,
            "rmse_f": rmse_f
        }

        return self.prediction_data
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

        plt.tight_layout()
        plt.show()
    def export_model_to_lammps(self, directory, filename=None, author="author", units="metal", knots_spacing_type="nk"):
        """
        Export the model to a LAMMPS potential file using the UF3LammpsPotentialGenerator.

        :param directory: Directory where the LAMMPS potential file will be saved.
        :param author: Author name for the LAMMPS potential file.
        :param units: Units for the LAMMPS potential file.
        :param knots_spacing_type: Knot spacing type, 'uk' (uniform) or 'nk' (non-uniform).
        """
        generator = UF3LammpsPotentialGenerator(self.model, self.chemical_system, self.bspline_config)
        generator.generate_potential_file(directory=directory, filename=filename, author=author, units=units, knots_spacing_type=knots_spacing_type)
        
    def validate_potential(self, potential_version, potential_file, element_list):
        """
        Validate the potential.
        """
        current_directory = os.getcwd()
        model_file = "model_file.lmp"
        dft_file = "POSCAR_661"
        element_list_str = ""
        for element in element_list:
            element_list_str += f"{element} "
        data = ValidationData(potential_version, potential_file)
        lattice_constant = LatticeConstantValidation(potential_file, model_file, dft_file, element_list_str)
        lattice_percent_error = lattice_constant.start_validation(overwrite=True, local=False)
        data.add_validation("lattice_constant", lattice_percent_error)
        os.chdir(current_directory)
        # langevin = LangevinValidation(potential_file, model_file, dft_file, element_list_str, npt_enable='')
        # langevin_percent_error = langevin.start_validation(overwrite=True, local=False)
        # data.add_validation("langevin", langevin_percent_error)
        return data

from datetime import datetime     
class UF3LammpsPotentialGenerator:
    def __init__(self, model, chemical_system, bspline_config):
        """
        Initialize the UF3LammpsPotentialGenerator with a model, chemical system, and B-spline configuration.

        :param model: The UF3 model object.
        :param chemical_system: The chemical system object associated with the model.
        :param bspline_config: The B-spline configuration object.
        """
        self.model = model
        self.chemical_system = chemical_system
        self.bspline_config = bspline_config

    def export_model_to_json(self, filename):
        """
        Export the model to a JSON file.

        :param filename: Name of the JSON file.
        """
        self.model.to_json(filename)

    def generate_potential_file(self, directory, filename=None, author="author", units="metal", knots_spacing_type="nk"):
        """
        Generate the UF3 LAMMPS potential file.

        :param directory: Directory path where the potential file will be saved.
        :param filename: Filename for the potential file. If not provided, it will be generated from the chemical system.
        :param author: Author of the potential file.
        :param units: LAMMPS units.
        :param knots_spacing_type: Knot spacing type, 'uk' (uniform) or 'nk' (non-uniform).
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if filename is None:
            filename = "".join(self.chemical_system.element_list) + ".uf3"
        
        full_path = os.path.join(directory, filename)

        # Check if file exists and add a timestamp if necessary
        if os.path.exists(full_path):
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filename + "_" + current_datetime
            full_path = os.path.join(directory, filename)

        self._write_uf3_lammps_pot_files(full_path, author, units, knots_spacing_type)

        # Create the pair_style and pair_coeff lines for LAMMPS
        lines = "pair_style\tuf3 %i %i" % (self.bspline_config.degree, len(self.chemical_system.element_list))
        lines += "\n"
        lines += "pair_coeff\t* * " + full_path + " "
        lines += " ".join(self.chemical_system.element_list)

        print("\n\n***Add the following line to the LAMMPS input script***\n\n")
        print(lines)
        print("\n\nCitation meta-data has been left blank. Please enter appropriate citation for the generated UF3 LAMMPS potential\n\n")

    def _write_uf3_lammps_pot_files(self, full_path, author, lammps_units, knots_spacing_type):
        """
        Creates and writes UF3 LAMMPS potential files.

        :param full_path: Full path to the potential file to be generated.
        :param author: Author of the potential file.
        :param lammps_units: LAMMPS units.
        :param knots_spacing_type: Knot spacing type, 'uk' (uniform) or 'nk' (non-uniform).
        """
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        files = {}

        # Process 2-body interactions
        for interaction in self.chemical_system.interactions_map[2]:
            key = '_'.join(interaction)
            files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n"
            files[key] += f"2B {interaction[0]} {interaction[1]} {self.bspline_config.leading_trim} {self.bspline_config.trailing_trim}"
            files[key] += f" {knots_spacing_type}\n"
            files[key] += str(self.bspline_config.r_max_map[interaction]) + " " + str(len(self.bspline_config.knots_map[interaction])) + "\n"
            files[key] += " ".join(['{:.17g}'.format(v) for v in self.bspline_config.knots_map[interaction]]) + "\n"
            files[key] += str(self.bspline_config.get_interaction_partitions()[0][interaction]) + "\n"
            start_index = self.bspline_config.get_interaction_partitions()[1][interaction]
            length = self.bspline_config.get_interaction_partitions()[0][interaction]
            files[key] += " ".join(['{:.17g}'.format(v) for v in self.model.coefficients[start_index:start_index + length]]) + "\n"
            files[key] += "#\n"

        # Process 3-body interactions if they exist
        if 3 in self.chemical_system.interactions_map:
            for interaction in self.chemical_system.interactions_map[3]:
                key = '_'.join(interaction)
                files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n"
                files[key] += f"3B {interaction[0]} {interaction[1]} {interaction[2]} {self.bspline_config.leading_trim} {self.bspline_config.trailing_trim}"
                files[key] += f" {knots_spacing_type}\n"
                files[key] += str(self.bspline_config.r_max_map[interaction][2]) + " " + str(self.bspline_config.r_max_map[interaction][1]) + " " + str(self.bspline_config.r_max_map[interaction][0]) + " "
                files[key] += str(len(self.bspline_config.knots_map[interaction][2])) + " " + str(len(self.bspline_config.knots_map[interaction][1])) + " " + str(len(self.bspline_config.knots_map[interaction][0])) + "\n"
                files[key] += " ".join(['{:.17g}'.format(v) for v in self.bspline_config.knots_map[interaction][2]]) + "\n"
                files[key] += " ".join(['{:.17g}'.format(v) for v in self.bspline_config.knots_map[interaction][1]]) + "\n"
                files[key] += " ".join(['{:.17g}'.format(v) for v in self.bspline_config.knots_map[interaction][0]]) + "\n"
                solutions = least_squares.arrange_coefficients(self.model.coefficients, self.bspline_config)
                decompressed = self.bspline_config.decompress_3B(solutions[(interaction[0], interaction[1], interaction[2])], (interaction[0], interaction[1], interaction[2]))
                files[key] += str(decompressed.shape[0]) + " " + str(decompressed.shape[1]) + " " + str(decompressed.shape[2]) + "\n"
                for i in range(decompressed.shape[0]):
                    for j in range(decompressed.shape[1]):
                        files[key] += ' '.join(map(str, decompressed[i, j])) + "\n"
                files[key] += "#\n"

        with open(full_path, "w") as fp:
            for k, v in files.items():
                fp.write(v)