import os
import shutil
import re
import subprocess
import os
import sys
import time

class SimulationManager:
    def __init__(self, parent_directory):
        """
        Initialize the SimulationManager with a parent directory.

        :param parent_directory: Path to the parent directory where simulations will be set up.
        """
        self.parent_directory = os.path.join(os.getcwd(), parent_directory)
        self.root_directory = os.path.join(self.parent_directory, "simulations/")
        self.basic_files_directory = os.path.join(self.parent_directory, "basic_files")
        self.template_directory = os.path.join(self.parent_directory, "templates")
    def get_root_directory(self):
        return self.root_directory
    def get_template_directory(self):
        return self.template_directory
    def get_simulation_directory(self):
        return self.simulation_dir
    def setup_simulation(self, simulation_type):
        """
        Set up a simulation directory and call the create_simulation method.
        """
        
         # Remove underscores, capitalize each word segment, and concatenate
        if "_" in simulation_type:
            formatted_simulation_type = simulation_type.replace('_', ' ').title().replace(' ', '')
        else:
            formatted_simulation_type = simulation_type
        if "Simulation" not in simulation_type:
            formatted_simulation_type += "Simulation"
        # Set up the simulation directory path
        self.simulation_dir = os.path.join(self.root_directory, formatted_simulation_type)

        # Create the simulation directory if it doesn't exist
        if not os.path.exists(self.simulation_dir):
            os.makedirs(self.simulation_dir)
        # Copy basic files into the simulation directory
        self.copy_basic_files(self.simulation_dir)
        # Create the specific simulation setup
    
    def copy_basic_files(self, simulation_dir):
        """
        Copy basic files needed for the simulation into the simulation directory.
        If the simulation directory contains subdirectories, copy the basic files into each subdirectory.
        Otherwise, copy the basic files directly into the simulation directory.
        
        :param simulation_dir: The directory where the simulation will be set up.
        """
        if self.basic_files_directory is None:
            self.find_basic_files_directory()
        # Check for subdirectories in the simulation directory
        subdirs = [os.path.join(simulation_dir, d) for d in os.listdir(simulation_dir) if os.path.isdir(os.path.join(simulation_dir, d)) and not d.startswith('.')]
        def copy_with_retries(src, dst, retries=5, delay=0.5):
            """Retry file copy if file is busy"""
            for _ in range(retries):
                try:
                    shutil.copy2(src, dst)
                    return  # Successful copy
                except (OSError, IOError) as e:
                    if "text file busy" in str(e).lower():
                        time.sleep(delay)
                    else:
                        raise
        if subdirs:
            for subdir in subdirs:
                for item in os.listdir(self.basic_files_directory):
                    src = os.path.join(self.basic_files_directory, item)
                    dst = os.path.join(subdir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        copy_with_retries(src, dst)
        else:
            for item in os.listdir(self.basic_files_directory):
                src = os.path.join(self.basic_files_directory, item)
                dst = os.path.join(simulation_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    copy_with_retries(src, dst)

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
        print(f"FILENAME {file_name} FROM {os.getcwd()} IS BEING COPIED TO {self.simulation_dir}")
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
    
