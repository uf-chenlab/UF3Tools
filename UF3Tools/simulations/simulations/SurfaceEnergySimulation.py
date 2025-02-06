import os 
import shutil
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser

class SurfaceEnergyBulkSimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "in.surface"
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        os.makedirs(simulation_directory, exist_ok=True)
        self.output_file = os.path.join(simulation_directory, "lammps.in")
        self.parameters = {
            'x_boundary': 'p',
            'y_boundary': 'p',
            'z_boundary': 'p',
            'change_box': '',
        }
        self.parameters.update(kwargs)
        
    def set_potential_file(self, potential_file):
        """
        Set the potential file for the LAMMPS simulation.
        """
        self.parameters['uf3_file'] = potential_file
        self.setup_input_file()
    def setup_input_file(self):
        """
        Set up the LAMMPS input file for the relaxation simulation.
        """
        handler = LammpsInputFileHandler(self.template_file, self.output_file) 
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if output file exists
        """
        
        print(f"existing bulk results: {os.path.exists(self.simulation_directory + '/energy.txt')}")
        return os.path.exists(self.simulation_directory + "/energy.txt")
    def get_results(self):
        """
        Get the results of the surface energy simulation and calculate the area.
        """
        # Paths to the necessary files
        energy_file_path = os.path.join(self.simulation_directory, "energy.txt")
        box_file_path = os.path.join(self.simulation_directory, "box.txt")
        results = {}

        try:
            # Read the contents of the energy.txt file
            with open(energy_file_path, "r") as file:
                energy_text = file.read()

            # Regular expression pattern to match the surface energy result
            energy_pattern = r'pe:\s*([-.\d]+)'  # Modified pattern to handle negative numbers

            energy_match = re.search(energy_pattern, energy_text)

            if energy_match:
                energy = float(energy_match.group(1))
                results['energy'] = energy
            else:
                print("Error: Surface energy could not be extracted. Please check the format of energy.txt.")
                results['energy'] = None

            # Read the contents of the box.txt file
            with open(box_file_path, "r") as file:
                box_text = file.read()

            # Regular expression pattern to match Lx and Ly values
            box_pattern = r'Lx:\s*([\d.]+)\s*Ly:\s*([\d.]+)'

            box_match = re.search(box_pattern, box_text)

            if box_match:
                Lx = float(box_match.group(1))
                Ly = float(box_match.group(2))
                area = Lx * Ly
                results['area'] = area
            else:
                print("Error: Lx and Ly values could not be extracted. Please check the format of box.txt.")
                results['area'] = None

            return results

        except:
            return {
                'energy': 999999,
                'area': 1
            }
        

class SurfaceEnergyFreeSimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "in.surface"
        self.original_directory = os.getcwd()
        os.makedirs(simulation_directory, exist_ok=True)
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "lammps.in")
        self.parameters = {
            'change_box': 'change_box all z final 0.0 15 units box'
        }
        self.parameters.update(kwargs)
        
    def set_potential_file(self, potential_file):
        """
        Set the potential file for the LAMMPS simulation.
        """
        self.parameters['uf3_file'] = potential_file
        self.setup_input_file()
    def setup_input_file(self):
        """
        Set up the LAMMPS input file for the relaxation simulation.
        """
        handler = LammpsInputFileHandler(self.template_file, self.output_file) 
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if output file exists
        """
        print(f"existing free results: {os.path.exists(self.simulation_directory + '/energy.txt')}")
        return os.path.exists(self.simulation_directory + "/energy.txt")
    def get_results(self):
        """
        Get the results of the surface energy simulation and calculate the area.
        """
        # Paths to the necessary files
        energy_file_path = os.path.join(self.simulation_directory, "energy.txt")
        box_file_path = os.path.join(self.simulation_directory, "box.txt")
        results = {}

        try:
            # Read the contents of the energy.txt file
            with open(energy_file_path, "r") as file:
                energy_text = file.read()

            # Regular expression pattern to match the surface energy result
            energy_pattern = r'pe:\s*([-.\d]+)'  # Modified pattern to handle negative numbers

            energy_match = re.search(energy_pattern, energy_text)
            print("energy path: ", energy_file_path)
            print("energy  text: ", energy_text)

            if energy_match:
                energy = float(energy_match.group(1))
                print("LAMMPS Surface energy extracted successfully:", energy)
                results['energy'] = energy
            else:
                print("Error: Surface energy could not be extracted. Please check the format of energy.txt.")
                results['energy'] = None

            # Read the contents of the box.txt file
            with open(box_file_path, "r") as file:
                box_text = file.read()

            # Regular expression pattern to match Lx and Ly values
            box_pattern = r'Lx:\s*([\d.]+)\s*Ly:\s*([\d.]+)'

            box_match = re.search(box_pattern, box_text)

            if box_match:
                Lx = float(box_match.group(1))
                Ly = float(box_match.group(2))
                area = Lx * Ly
                print(f"Area calculated successfully: Lx = {Lx}, Ly = {Ly}, Area = {area}")
                results['area'] = area
            else:
                print("Error: Lx and Ly values could not be extracted. Please check the format of box.txt.")
                results['area'] = None

            return results

        except:
            return {
                'energy': 999999,
                'area': 1
            }

class SurfaceEnergyRelaxationSimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "in.surface"
        self.original_directory = os.getcwd()
        os.makedirs(simulation_directory, exist_ok=True)
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "lammps.in")
        self.parameters = {
            'x_boundary': 'p',
            'y_boundary': 'p',
            'z_boundary': 'p',
        }
        self.parameters.update(kwargs)
        
    def set_potential_file(self, potential_file):
        """
        Set the potential file for the LAMMPS simulation.
        """
        self.parameters['uf3_file'] = potential_file
        self.setup_input_file()
    def setup_input_file(self):
        """
        Set up the LAMMPS input file for the relaxation simulation.
        """
        handler = LammpsInputFileHandler(self.template_file, self.output_file) 
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if output file exists
        """
        print(f"existing free results: {os.path.exists(self.simulation_directory + '/energy.txt')}")
        return os.path.exists(self.simulation_directory + "data.relaxed")
    def get_results(self):
        """
        Get the results of the surface energy simulation.
        """
        # Read the contents of the energy.txt file
        energy_file_path = os.path.join(self.simulation_directory, "data.relaxed")
        with open(energy_file_path, "r") as file:
            energy_text = file.read()

        return energy_text     

class SurfaceEnergySimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "in.surface"
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "lammps.in")
        self.parameters = {
            'atom_style': 'atomic',
            'uf3_degree': '3',
            'uf3_file': 'AlN.uf3',
            'uf3_elements': 'Al N',
            'box_relax_settings': 'fix 1 all box/relax iso 0.0 vmax 0.001',
            'min_style': 'cg',
            'etol': '1e-25',
            'ftol': '1e-25',
            'maxiter': '10000',
            'maxeval': '50000'
        }
        self.parameters.update(kwargs)
        #self.relaxation_simulation = SurfaceEnergyRelaxationSimulation(self.simulation_directory + "/SurfaceEnergyRelaxationSimulation", template_directory, **self.parameters)
        self.free_simulation = SurfaceEnergyFreeSimulation(self.simulation_directory + "/SurfaceEnergyFreeSimulation", template_directory, **self.parameters)
        self.bulk_simulation = SurfaceEnergyBulkSimulation(self.simulation_directory + "/SurfaceEnergyBulkSimulation", template_directory, **self.parameters)
    def set_potential_file(self, potential_file):
        """
        Set the potential file for the LAMMPS simulation.
        """
        self.parameters['uf3_file'] = potential_file
        #self.relaxation_simulation.set_potential_file(potential_file)
        self.free_simulation.set_potential_file(potential_file)
        self.bulk_simulation.set_potential_file(potential_file)
    def setup_input_file(self):
        """
        Set up the LAMMPS input file for the surface energy simulation.
        """
        #self.relaxation_simulation.setup_input_file()
        self.free_simulation.setup_input_file()
        self.bulk_simulation.setup_input_file()
        
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        
        return self.free_simulation.check_existing_results() and self.bulk_simulation.check_existing_results()
       
    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
        
        # initialize an empty dictionary to store energies
        results = {}
        bulk_energy = self.bulk_simulation.get_results()
        free_energy = self.free_simulation.get_results()
        results['bulk_energy'] = bulk_energy['energy']
        results['free_energy'] = free_energy['energy']
        results['bulk_area'] = bulk_energy['area']
        results['free_area'] = free_energy['area']

        return results
    
class DFTSurfaceEnergyFreeSimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "INCAR"
        self.original_directory = os.getcwd()
        os.makedirs(simulation_directory, exist_ok=True)
        self.simulation_directory = simulation_directory
        self.z_increase = 15
        self.output_file = os.path.join(simulation_directory, "INCAR")
        self.parameters = {
            'algo': 'Fast',
            'prec': 'Accurate',
            'ibrion': '2',
            'isif': '3',
            'lcharg': 'FALSE',
            'lwave': 'FALSE',
            'lreal': '.FALSE.',
            'ismear': '0',
            'sigma': '0.05',
            'npar': '4',
            'nsw': '1',
            'ncore': '8',
            'encut': '600',
            'extras': ''
        }
        self.parameters.update(kwargs)
        self.lattice_parameters = {}
    def setup_input_file(self):
        """
        Set up the VASP INCAR file for the relaxation simulation.
        """
        handler = LammpsInputFileHandler(self.template_file, self.output_file) 
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        outcar_exists = os.path.exists(os.path.join(self.simulation_directory, "OUTCAR"))
        contcar_exists = os.path.exists(os.path.join(self.simulation_directory, "CONTCAR"))
        return outcar_exists and contcar_exists
    def perform_model_modification(self):
        #open the POSCAR file
        with open(os.path.join(self.simulation_directory, "POSCAR")) as f:
            poscar_content = f.read()
        lines = poscar_content.splitlines()
        
        # The third line contains the lattice vectors. We modify the Z component of the third vector.
        lattice_z_vector = lines[4].split()
        lattice_z_vector[2] = str(float(lattice_z_vector[2]) + self.z_increase)
        
        # Rejoin the modified vector
        lines[4] = "       " + "        ".join(lattice_z_vector)
        
        temp = "\n".join(lines)
        # Write the modified POSCAR file
        with open(os.path.join(self.simulation_directory, "POSCAR"), 'w') as f:
            f.write(temp)


    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
        
        parser = DFTDataParser(self.simulation_directory)
        energy = parser.get_energy()

        return energy
    
class DFTSurfaceEnergyBulkSimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "INCAR"
        self.original_directory = os.getcwd()
        os.makedirs(simulation_directory, exist_ok=True)
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "INCAR")
        self.parameters = {
            'algo': 'Fast',
            'prec': 'Accurate',
            'ibrion': '2',
            'isif': '3',
            'lcharg': 'FALSE',
            'lwave': 'FALSE',
            'lreal': '.FALSE.',
            'ismear': '0',
            'sigma': '0.05',
            'npar': '4',
            'nsw': '1',
            'ncore': '8',
            'encut': '600',
            'extras': ''
        }
        self.parameters.update(kwargs)
        self.lattice_parameters = {}
    def setup_input_file(self):
        """
        Set up the VASP INCAR file for the relaxation simulation.
        """
        handler = LammpsInputFileHandler(self.template_file, self.output_file) 
        handler.update_simulation_parameters(**self.parameters)
        handler.generate_input_file()
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        outcar_exists = os.path.exists(os.path.join(self.simulation_directory, "OUTCAR"))
        contcar_exists = os.path.exists(os.path.join(self.simulation_directory, "CONTCAR"))
        return outcar_exists and contcar_exists

    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
        
        parser = DFTDataParser(self.simulation_directory)
        energy = parser.get_energy()

        return energy
    
class DFTSurfaceEnergySimulation:
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the DFTRelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "INCAR"
        self.original_directory = os.getcwd()
        self.simulation_directory = simulation_directory
        self.output_file = os.path.join(simulation_directory, "INCAR")
        self.parameters = {
            'algo': 'Fast',
            'prec': 'Accurate',
            'ibrion': '2',
            'isif': '3',
            'lcharg': 'FALSE',
            'lwave': 'FALSE',
            'lreal': 'Auto',
            'ismear': '0',
            'sigma': '0.05',
            'npar': '4',
            'nsw': '1',
            'ncore': '8',
            'encut': '600',
            'extras': ''
        }
        self.parameters.update(kwargs)
        self.free_simulation = DFTSurfaceEnergyFreeSimulation(self.simulation_directory + "/DFTSurfaceEnergyFreeSimulation", template_directory, **self.parameters)
        self.bulk_simulation = DFTSurfaceEnergyBulkSimulation(self.simulation_directory + "/DFTSurfaceEnergyBulkSimulation", template_directory, **self.parameters)
    def setup_input_file(self):
        """
        Set up the VASP INCAR file for the relaxation simulation.
        """
        self.free_simulation.setup_input_file() 
        self.bulk_simulation.setup_input_file()
    def check_existing_results(self):
        """
        Check if the simulation directory already contains results.

        :return: True if both OUTCAR and CONTCAR exist, False otherwise.
        """
        return self.free_simulation.check_existing_results() and self.bulk_simulation.check_existing_results()

    def process_simulation_results(self):
        """
        Process the results of the relaxation simulation.
        """
        
        results = {}
        bulk_energy = self.bulk_simulation.process_simulation_results()
        free_energy = self.free_simulation.process_simulation_results()

        return results
    def perform_model_modification(self):
        self.free_simulation.perform_model_modification()