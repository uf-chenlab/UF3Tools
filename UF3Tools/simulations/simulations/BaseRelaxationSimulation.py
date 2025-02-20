import os
import re
from .Simulation import BaseSimulation
class BaseRelaxationSimulation(BaseSimulation):
    simulation_type = "BaseRelaxation"
    required_supercell = (1,1,1)
    def __init__(self, input_file, model_file, potential_file, simulation_root_directory = os.getcwd(), run_script = "start_simulation.sh", simulation_name = None, extra_files = [], parameters=None):
        self.original_directory = os.getcwd()
        self.input_file = os.path.abspath(input_file)
        self.parameters = {
            'atom_style': 'atomic',
            'x_boundary': 'p',
            'y_boundary': 'p',
            'z_boundary': 'p',
            'change_box': '',
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
        if parameters is not None:
            self.parameters.update(parameters)
        if simulation_name is None:
            simulation_name = self.simulation_type
        super().__init__(simulation_name, os.path.abspath(input_file), os.path.abspath(model_file), os.path.abspath(potential_file),simulation_root_directory, run_script, extra_files, parameters=self.parameters)
    
    def get_result(self):
        if(self.result != {}):
            return self.result
        else:
            return self.process_simulation_results()
        
    def process_simulation_results(self):
        self._read_cohesive_energy()
        self._read_lattice_constants()
        self._read_total_energy()
        self._read_num_atoms()
        return self.result
        
    def _read_lattice_constants(self):
        lattice_file_path = os.path.join(self.simulation_directory, "lattice.txt")
        pattern = r'a:\s*([\d.]+)\s*b:\s*([\d.]+)\s*c:\s*([\d.]+)\s*alpha:\s*([\d.]+)\s*beta:\s*([\d.]+)\s*gamma:\s*([\d.]+)'
        match = self.read_result_file(lattice_file_path, pattern)
        if match:
            self.result['lattice_constant'] = {
                key: float(match.group(i + 1))
                for i, key in enumerate(['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
            }
        else:
            self.result['lattice_constant'] = None
        return self.result['lattice_constant']

    def _read_cohesive_energy(self):
        energy_file_path = os.path.join(self.simulation_directory, "energy.txt")
        pattern = r'ecoh\s*=\s*([-.\d]+)'
        match = self.read_result_file(energy_file_path, pattern)
        if match:
            ecoh = float(match.group(1))
            self.result['cohesive_energy'] = ecoh
            self.result['energy_per_atom'] = ecoh
        else:
            self.result['cohesive_energy'] = None
            self.result['energy_per_atom'] = None
        return self.result['cohesive_energy']

    def _read_total_energy(self):
        energy_file_path = os.path.join(self.simulation_directory, "total_energy.txt")
        pattern = r'total\s*energy\s*=\s*([-.\d]+)'
        match = self.read_result_file(energy_file_path, pattern)
        self.result['bulk_energy'] = float(match.group(1)) if match else None
        return self.result['bulk_energy']

    def _read_num_atoms(self):
        num_atoms_file_path = os.path.join(self.simulation_directory, "num_atoms.txt")
        pattern = r'Number\s*of\s*atoms\s*=\s*(\d+)'
        match = self.read_result_file(num_atoms_file_path, pattern)
        self.result['num_atoms'] = int(match.group(1)) if match else None
        return self.result['num_atoms']