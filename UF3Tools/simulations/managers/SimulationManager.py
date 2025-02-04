import os
import shutil
import re
import subprocess
import os
import sys

class SimulationManager:
    def __init__(self, parent_directory):
        """
        Initialize the SimulationManager with a parent directory.

        :param parent_directory: Path to the parent directory where simulations will be set up.
        """
        self.parent_directory = os.path.join(os.getcwd(), parent_directory)
        self.root_directory = os.path.join(self.parent_directory, "simulations/")
        self.basic_files_directory = None

    def setup_simulation(self, simulation_type, **kwargs):
        """
        Set up a simulation directory and call the create_simulation method.
        """
        
        print(f"simulation manager: {self.parent_directory}, {self.root_directory}, {self.basic_files_directory}")
        # Remove underscores, capitalize each word segment, and concatenate
        formatted_simulation_type = simulation_type.replace('_', ' ').title().replace(' ', '')
        
        # Set up the simulation directory path
        self.simulation_dir = os.path.join(self.root_directory, formatted_simulation_type + "Simulation")
        print(f"simulation directory: {self.simulation_dir}")

        # Create the simulation directory if it doesn't exist
        if not os.path.exists(self.simulation_dir):
            os.makedirs(self.simulation_dir)
        print("directory made")
        self.create_simulation(simulation_type, **kwargs)
        print("simulation created")
        # Copy basic files into the simulation directory
        self.copy_basic_files(self.simulation_dir)
        print("basic files copied")
        # Create the specific simulation setup
        

        



        

    def create_simulation(self, simulation_type, **kwargs):
        """
        Create the specific simulation. This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def copy_basic_files(self, simulation_dir):
        """
        Copy basic files needed for the simulation into the simulation directory.
        If the simulation directory contains subdirectories, copy the basic files into each subdirectory.
        Otherwise, copy the basic files directly into the simulation directory.
        
        :param simulation_dir: The directory where the simulation will be set up.
        """
        if self.basic_files_directory is None:
            self.find_basic_files_directory()
        print("basic files directory: ", self.basic_files_directory)
        # Check for subdirectories in the simulation directory
        subdirs = [os.path.join(simulation_dir, d) for d in os.listdir(simulation_dir) if os.path.isdir(os.path.join(simulation_dir, d)) and not d.startswith('.')]

        if subdirs:
            # Copy files into each subdirectory
            for subdir in subdirs:
                for item in os.listdir(self.basic_files_directory):
                    s = os.path.join(self.basic_files_directory, item)
                    d = os.path.join(subdir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
        else:
            # If no subdirectories, copy files directly to simulation_dir
            for item in os.listdir(self.basic_files_directory):
                s = os.path.join(self.basic_files_directory, item)
                d = os.path.join(simulation_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

    def start_simulation(self, command, overwrite=False):
        """
        Start the simulation by running the specified command in all subdirectories.
        If no subdirectories are present, run the simulation in the simulation directory.

        :param command: The command to run the simulation.
        :param overwrite: Whether to overwrite existing results.
        :return: Processed simulation results.
        """
        original_directory = os.getcwd()
        os.chdir(self.simulation_dir)
        
        # Get list of all subdirectories in the simulation directory
        subdirectories = [d for d in os.listdir() if os.path.isdir(d) and not d.startswith(".")]
        #remove directories that start with "."
        #subdirectories = [d for d in subdirectories if not d.startswith(".")]
        
        results = []
        
        # If there are no subdirectories, just run the simulation in the current directory
        if not subdirectories:
            subdirectories = [self.simulation_dir]
        # Iterate through each subdirectory and run the simulation if required
        for subdir in subdirectories:
            os.chdir(subdir)
            
            if self.simulation.check_existing_results() and not overwrite:
                #print(f"Existing results found in {subdir}. Using existing results...")
                continue
            else:
                subprocess.run(command, shell=True,stdout=subprocess.DEVNULL)
            os.chdir(self.simulation_dir)  # Go back to the main simulation directory
        
        result = self.simulation.process_simulation_results()
        results.append(result)
        
        os.chdir(original_directory)  # Go back to the original directory
        
        # Combine or process all results as needed here, or return them as a list
        return results
    def find_basic_files_directory(self):
        basic_files_dir = "basic_files"
        current_dir_basic_files = os.path.join(os.getcwd(), basic_files_dir)
        parent_dir_basic_files = os.path.join(self.parent_directory, basic_files_dir)

        if os.path.exists(current_dir_basic_files):
            source_dir = current_dir_basic_files
        elif os.path.exists(parent_dir_basic_files):
            source_dir = parent_dir_basic_files
        else:
            raise FileNotFoundError(f"'basic_files' directory not found in current directory or parent directory.")
        self.basic_files_directory = source_dir
        return source_dir
        
    def register_basic_file(self, file_name):
        """
        Insert a basic file into the basic_files directory.

        :param file_name: The name of the basic file to insert.
        """
        if(self.basic_files_directory == None):
            self.find_basic_files_directory()  
        destination_file = os.path.join(self.basic_files_directory, file_name)
        shutil.copyfile(file_name, destination_file)
    def register_simulation_file(self, file_name):
        """
        Insert a simulation file into the simulation directory.
        If there are subdirectories in the simulation directory, the file will be copied into each of them.
        
        :param file_name: The name of the simulation file to insert.
        """
        # Check for subdirectories in the simulation directory
        subdirectories = [d for d in os.listdir(self.simulation_dir) if os.path.isdir(os.path.join(self.simulation_dir, d)) and not d.startswith('.')]

        if subdirectories:
            # If there are subdirectories, copy the file into each of them
            for subdir in subdirectories:
                destination_file = os.path.join(self.simulation_dir, subdir, file_name)
                destination_dir = os.path.dirname(destination_file)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                shutil.copy(file_name, destination_file)
        else:
            # If there are no subdirectories, copy the file directly into simulation_dir
            destination_file = os.path.join(self.simulation_dir, file_name)
            destination_dir = os.path.dirname(destination_file)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            shutil.copy(file_name, destination_file)
    def perform_model_modification(self):
        """
        Perform any necessary model modifications before running the simulation.
        """
        self.simulation.perform_model_modification()
    
