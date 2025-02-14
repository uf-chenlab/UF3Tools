import os
import re
from .Simulation import BaseSimulation
from ..common.LammpsInputFileHandler import LammpsInputFileHandler
from ..common.DFTDataParser import DFTDataParser

class BaseRelaxationSimulation(BaseSimulation):
    simulation_type = "BaseRelaxation"
    requires_supercell = False
    def __init__(self, simulation_directory, template_directory, **kwargs):
        """
        Initialize the RelaxationSimulation with a simulation directory and any additional parameters.

        :param simulation_directory: The directory where the relaxation simulation will be set up.
        :param kwargs: Additional parameters for setting up the simulation.
        """
        self.template_file = template_directory + "in.base_relax"
        self.original_directory = os.getcwd()
        self.simulation_directory = os.path.join(simulation_directory, (self.simulation_type+'Simulation'))
        print("simulation directory: ", self.simulation_directory)
        print("pased simulation directory: ",simulation_directory)
        self.output_file = os.path.join(simulation_directory, "lammps.in")
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
        self.parameters.update(kwargs)
        super().__init__(simulation_directory, os.path.join(template_directory, (self.simulation_type + 'Simulation')), "in.base_relax", "lammps.in", self.parameters, **kwargs)
        self.cohesive_energy = {}
    
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
        try:
            pattern = r'a:\s*([\d.]+)\s*b:\s*([\d.]+)\s*c:\s*([\d.]+)\s*alpha:\s*([\d.]+)\s*beta:\s*([\d.]+)\s*gamma:\s*([\d.]+)'
            match = self.read_result_file(lattice_file_path, pattern)
            if match:
                self.result['lattice_constant'] = {key: float(match.group(i+1)) for i, key in enumerate(['a', 'b', 'c', 'alpha', 'beta', 'gamma'])}
            else:
                print("Error: Lattice parameters could not be extracted.")
                self.result['lattice_constant'] = None
        except:
            self.result['lattice_constant'] = None
        return self.result['lattice_constant']
    def _read_cohesive_energy(self):
        energy_file_path = os.path.join(self.simulation_directory, "energy.txt")
        try:
            energy_pattern = r'ecoh\s*=\s*([-.\d]+)'

            energy_match = self.read_result_file(energy_file_path, energy_pattern)

            if energy_match:
                energy = float(energy_match.group(1))
                self.result['cohesive_energy'] = energy
                self.result['energy_per_atom'] = energy
            else:
                print("Error: Surface energy could not be extracted. Please check the format of energy.txt.")
                self.result['cohesive_energy'] = None
                self.result['energy_per_atom'] = None

        except:
            self.result['cohesive_energy'] = None
            self.result['energy_per_atom'] = None
        return self.result['cohesive_energy']
    
    def _read_total_energy(self):
        energy_file_path = os.path.join(self.simulation_directory, "total_energy.txt")
        try:
            energy_pattern = r'total\s*energy\s*=\s*([-.\d]+)'
            energy_match = self.read_result_file(energy_file_path, energy_pattern)

            if energy_match:
                energy = float(energy_match.group(1))
                self.result['bulk_energy'] = energy
            else:
                print("Error: Surface energy could not be extracted. Please check the format of energy.txt.")
                self.result['bulk_energy'] = None

        except:
            self.result['bulk_energy'] = None
        return self.result['bulk_energy']
    def _read_num_atoms(self):
        num_atoms_file_path = os.path.join(self.simulation_directory, "num_atoms.txt")
        try:
            num_atoms_pattern = r'Number\s*of\s*atoms\s*=\s*(\d+)'
            num_atoms_match = self.read_result_file(num_atoms_file_path, num_atoms_pattern)

            if num_atoms_match:
                num_atoms = int(num_atoms_match.group(1))
                self.result['num_atoms'] = num_atoms
            else:
                print("Error: Surface energy could not be extracted. Please check the format of energy.txt.")
                self.result['num_atoms'] = None

        except:
            self.result['num_atoms'] = None
        return self.result['num_atoms']