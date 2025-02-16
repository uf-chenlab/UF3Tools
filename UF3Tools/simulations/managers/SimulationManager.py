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
        """
        self.parent_directory = os.path.abspath(parent_directory)
        print(f"Parent directory: {self.parent_directory}")
        self.root_directory = os.path.join(self.parent_directory, "simulations")
        print(f"Root directory: {self.root_directory}")
        print(f"Parent directory: {self.parent_directory}")
        print(f"Root directory: {self.root_directory}")
        # Ensure basic_files_directory is correctly set
        self.basic_files_directory = self.find_basic_files_directory()

        self.template_directory = os.path.join(self.parent_directory, "templates")

    def find_basic_files_directory(self):
        """
        Find and set the correct path for basic_files_directory.
        """
        possible_paths = [
            os.path.join(self.parent_directory, "basic_files"),
            os.path.join(self.root_directory, "basic_files"),
            os.path.join(os.getcwd(), "basic_files")
        ]
        print(f"Possible paths for basic_files_directory: {possible_paths}")
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not locate the 'basic_files' directory in expected locations.")

    def get_root_directory(self):
        return self.root_directory

    def get_template_directory(self):
        return self.template_directory

    def get_simulation_directory(self):
        return self.simulation_dir

    def setup_simulation(self, simulation_type):
        """
        Set up a simulation directory correctly without mixing it with potential files.
        """

        if "Simulation" not in simulation_type:
            simulation_type += "Simulation"

        # Ensure simulation directory is different from potential file location
        if "validation/structures" and "lammps" in self.parent_directory:
            # If inside a structure validation directory, use that path
            self.simulation_dir = os.path.join(self.parent_directory, "simulations", simulation_type)
        else:
            # Otherwise, use the normal root directory
            self.simulation_dir = os.path.join(self.root_directory, simulation_type)

        os.makedirs(self.simulation_dir, exist_ok=True)

        # Make sure it's not trying to copy into itself
        if os.path.abspath(self.simulation_dir) == os.path.abspath(self.basic_files_directory):
            print(f"Simulation directory ({self.simulation_dir}) conflicts with basic_files. Adjusting path.")
            self.simulation_dir = os.path.join(self.root_directory, "simulations", simulation_type)

        if os.path.exists(self.basic_files_directory):
            self.copy_basic_files(self.simulation_dir)
        else:
            raise FileNotFoundError(f"Basic files directory not found: {self.basic_files_directory}")

    def copy_basic_files(self, simulation_dir):
        if self.basic_files_directory is None:
            self.find_basic_files_directory()

        subdirs = [
            os.path.join(simulation_dir, d)
            for d in os.listdir(simulation_dir)
            if os.path.isdir(os.path.join(simulation_dir, d)) and not d.startswith(".")
        ]

        def copy_with_retries(src, dst, retries=5, delay=0.5):
            for _ in range(retries):
                try:
                    shutil.copy2(src, dst)
                    return
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

    def register_basic_file(self, file_name):
        if self.basic_files_directory is None:
            self.find_basic_files_directory()
        destination_file = os.path.join(self.basic_files_directory, file_name)
        shutil.copyfile(file_name, destination_file)

    def register_simulation_file(self, file_name):
        """
        Insert a simulation file into the simulation directory. If subdirectories exist,
        the file will be copied into each of them.
        """
        print(f"\nDebug: Trying to copy {file_name} into {self.simulation_dir}")
        
        subdirectories = [
            d for d in os.listdir(self.simulation_dir)
            if os.path.isdir(os.path.join(self.simulation_dir, d)) and not d.startswith(".")
        ]

        if subdirectories:
            for subdir in subdirectories:
                destination_file = os.path.join(self.simulation_dir, subdir, os.path.basename(file_name))
                print(f"Source: {os.path.abspath(file_name)}")
                print(f"Destination: {os.path.abspath(destination_file)}")

                if os.path.abspath(file_name) == os.path.abspath(destination_file):
                    print(" Skipping copy: Source and destination are the same")
                    continue  # Skip copying to itself

                destination_dir = os.path.dirname(destination_file)
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy(file_name, destination_file)
                print(f"Copied {file_name} to {destination_file}")

        else:
            destination_file = os.path.join(self.simulation_dir, os.path.basename(file_name))
            print(f"Source: {os.path.abspath(file_name)}")
            print(f"Destination: {os.path.abspath(destination_file)}")

            if os.path.abspath(file_name) == os.path.abspath(destination_file):
                print("Skipping copy: Source and destination are the same")
                return  # Skip copying to itself

            destination_dir = os.path.dirname(destination_file)
            os.makedirs(destination_dir, exist_ok=True)
            shutil.copy(file_name, destination_file)
            print(f"Copied {file_name} to {destination_file}")
    
