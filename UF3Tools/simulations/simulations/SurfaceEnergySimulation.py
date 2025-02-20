import os
import re
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser
from .Simulation import CompleteLammpsSimulation

class SurfaceEnergySimulation(CompleteLammpsSimulation):
    simulation_type = "SurfaceEnergy"
    result_type = "surface_energy"
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
        self.parameters = {
            "atom_style": "atomic",
            "change_box": "",
            "box_relax_settings": "",
            "uf3_degree": "3",
            "uf3_file": "AlN.uf3",
            "uf3_elements": "Al N",
            "min_style": "cg",
            "etol": "1e-25",
            "ftol": "1e-25",
            "maxiter": "10000",
            "maxeval": "50000",
        }
        if parameters is not None:
            self.parameters.update(parameters)
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, parameters=self.parameters)

    def process_simulation_results(self):
        self.result["bulk_energy"] = self._read_bulk_energy()
        self.result["slab_energy"] = self._read_slab_energy()
        self.result["unrelaxed_slab_energy"] = self._read_unrelaxed_slab_energy()
        self.result.update(self._read_area())
        return self.result

    def _read_bulk_energy(self):
        bulk_file_path = os.path.join(self.simulation_directory, "bulk.txt")
        pattern = r"pe\s*=\s*([-.\d]+)"
        match = self.read_result_file(bulk_file_path, pattern)
        if match:
            return float(match.group(1))
        return None

    def _read_slab_energy(self):
        slab_file_path = os.path.join(self.simulation_directory, "relaxed_slab.txt")
        pattern = r"pe\s*=\s*([-.\d]+)"
        match = self.read_result_file(slab_file_path, pattern)
        if match:
            return float(match.group(1))
        return None

    def _read_unrelaxed_slab_energy(self):
        slab_file_path = os.path.join(self.simulation_directory, "unrelaxed_slab.txt")
        pattern = r"pe\s*=\s*([-.\d]+)"
        match = self.read_result_file(slab_file_path, pattern)
        if match:
            return float(match.group(1))
        return None

    def _read_area(self):
        bulk_area = self._read_area_file(os.path.join(self.simulation_directory, "bulk_area.txt"))
        slab_area = self._read_area_file(os.path.join(self.simulation_directory, "relaxed_slab_area.txt"))
        unrelaxed_area = self._read_area_file(os.path.join(self.simulation_directory, "unrelaxed_slab_area.txt"))
        return {
            "bulk_area": bulk_area,
            "relaxed_slab_area": slab_area,
            "unrelaxed_slab_area": unrelaxed_area
        }

    def _read_area_file(self, path):
        pattern = r"lx\s*=\s*([-.\d]+)\s*ly\s*=\s*([-.\d]+)"
        match = self.read_result_file(path, pattern)
        if match:
            return float(match.group(1)) * float(match.group(2))
        return None
        
    
    
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