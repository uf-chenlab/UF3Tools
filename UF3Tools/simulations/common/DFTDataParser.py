import os
import re
import shutil
import numpy as np
from pymatgen.io.vasp import Poscar, Outcar
from pymatgen.core import Structure
from pymatgen.core import Element
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
import phonopy
import yaml
class DFTDataParser:
    def __init__(self, directory):
        """
        Initialize the DFTDataParser with a directory containing DFT data.

        :param directory: Path to the directory containing DFT data.
        """
        self.directory = directory
        self.data = None
    def get_lattice_constants(self, filename=None):
        """Read relaxed lattice constants from CONTCAR file or specified POSCAR file
        :param filename: Name of the POSCAR file to read
        :return: Dictionary of lattice constants
        """
        # Use the CONTCAR file by default
        if filename is None:
            filename = os.path.join(self.directory, "CONTCAR")
        else:
            filename = os.path.join(self.directory, filename)
        # Read the lattice constants from the POSCAR file
        structure = Poscar.from_file(filename).structure
        lattice = structure.lattice
        a, b, c = lattice.abc  # Lattice parameters a, b, c
        alpha, beta, gamma = lattice.angles  # Angles between lattice vectors
        lattice_parameters = {
            'a': a,
            'b': b,
            'c': c,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }
        return lattice_parameters
    def read_result_file(self, result_file, pattern):
        results = {}
        try:
            with open(result_file, 'r') as f:
                lines = f.read()
            match = re.search(pattern, lines)
            return match
        except:
            print(f"Error: Could not open file {result_file}")
            return None
    def get_surface_energy(self, path):
        """
        Calculate the surface energy by subtracting the final energy of the
        bulk simulation from the free surface simulation.

        :return: Surface energy in eV
        """

        # Get the energies from the respective OUTCAR files
        bulk_energy_path = os.path.join(path, "bulk_energy.txt")
        slab_energy_path = os.path.join(path, "slab_energy.txt")
        contcar_path = os.path.join(path, "CONTCAR")
        energy_pattern = r'total\s*energy\s*=\s*([-.\d]+)'
        bulk_match = self.read_result_file(bulk_energy_path, energy_pattern)
        bulk_energy = float(bulk_match.group(1))
        slab_match = self.read_result_file(slab_energy_path, energy_pattern)
        slab_energy = float(slab_match.group(1))
        structure = Structure.from_file(contcar_path)
        lattice = structure.lattice.matrix
        a_vector = lattice[0]
        b_vector = lattice[1]
        area = np.linalg.norm(np.cross(a_vector, b_vector))
        
        print("Bulk Energy: ", bulk_energy)
        print("Free Energy: ", slab_energy)
        print("Area: ", area)
        # Calculate the surface energy
        surface_energy = (slab_energy - bulk_energy)/area
        array = {
            "bulk_energy": bulk_energy,
            "slab_energy": slab_energy,
            "surface_energy": surface_energy,
            "area": area
        }
        return array
    def get_energy(self, filename=None):
        """
        Read the final energy from the OUTCAR file

        :param filename: Path to the OUTCAR file to read
        :return: Final energy in eV
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"OUTCAR file not found: {filename}")

        outcar = Outcar(filename)
        energy = outcar.final_energy
        return energy
    def get_elastic_tensor(self, dft_path):
        dft_data = {
            "elastic_tensor": None,
            "bulk_modulus_voigt": None,
            "bulk_modulus_reuss": None,
            "bulk_modulus_hill": None,
            "shear_modulus_voigt": None,
            "shear_modulus_reuss": None,
            "shear_modulus_hill": None,
        }
        with open(dft_path, 'r') as file:
            lines = file.readlines()
            
        elastic_tensor_start = False
        elastic_tensor = []
        for line in lines:
            if "Stiffness Tensor C_ij (in GPa):" in line:
                elastic_tensor_start = True
                continue
            if elastic_tensor_start:
                if line.strip() == "":
                    break
                tensor_row = list(map(float, line.split()))
                elastic_tensor.append(tensor_row)
                
        dft_data["elastic_tensor"] = elastic_tensor
        
        # Reading bulk and shear modulus (Voigt, Reuss, Hill)
        for line in lines:
            if "Bulk Modulus B (GPa)" in line:
                match = re.search(r"\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|", line)
                if match:
                    dft_data["bulk_modulus_voigt"] = float(match.group(1))
                    dft_data["bulk_modulus_reuss"] = float(match.group(2))
                    dft_data["bulk_modulus_hill"] = float(match.group(3))
            
            if "Shear Modulus G (GPa)" in line:
                match = re.search(r"\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|", line)
                if match:
                    dft_data["shear_modulus_voigt"] = float(match.group(1))
                    dft_data["shear_modulus_reuss"] = float(match.group(2))
                    dft_data["shear_modulus_hill"] = float(match.group(3))

        return dft_data
    def get_cohesive_energy(self, dft_path):
        dft_data = {
            'cohesive_energy': None,
            'free_atom_energy': None,
            'bulk_energy': None
        }
        with open(dft_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "ecoh" in line:
                    # Replace special minus sign with standard minus sign
                    dft_data["cohesive_energy"] = float(line.split()[1].replace("−", "-"))
                elif "efree" in line:
                    dft_data["free_atom_energy"] = float(line.split()[1].replace("−", "-"))  
                elif "ebulk" in line:
                    dft_data["bulk_energy"] = float(line.split()[1].replace("−", "-"))
        return dft_data
    def poscarToLAMMPS(self, poscar, outputFilename):
        """
        Converts a VASP POSCAR/CONTCAR file to a LAMMPS data file.

        Parameters:
        poscar (str): Path to the POSCAR or CONTCAR file.
        outputFilename (str): Path to the output LAMMPS data file.
        """
        # Load the structure from the POSCAR/CONTCAR file
        structure = Poscar.from_file(poscar).structure

        # Extract unique species from the structure
        species_list = structure.symbol_set  # Sorted list of unique species

        # Open the output file for writing
        with open(outputFilename, "w") as file:
            # Header section
            file.write(f"LAMMPS data file generated from VASP {poscar}\n\n")
            file.write(f"{structure.num_sites} atoms\n")
            file.write(f"{len(species_list)} atom types\n\n")

            # Lattice vectors (box dimensions)
            lattice = structure.lattice
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles

            # Convert angles to radians
            alpha_r = np.radians(alpha)
            beta_r = np.radians(beta)
            gamma_r = np.radians(gamma)

            # Calculate the correct y vector and the triclinic tilt factors
            lx = a
            xy = b * np.cos(gamma_r)
            ly = np.sqrt(b**2 - xy**2)
            xz = c * np.cos(beta_r)
            yz = (b * c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r))) / ly
            lz = np.sqrt(c**2 - xz**2 - yz**2)

            # Write the lattice dimensions and tilt factors
            file.write(f"0.0 {lx:.6f} xlo xhi\n")
            file.write(f"0.0 {ly:.6f} ylo yhi\n")
            file.write(f"0.0 {lz:.6f} zlo zhi\n")
            file.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n\n")

            # Masses section
            file.write("Masses\n\n")
            for i, species in enumerate(species_list, start=1):
                mass = float(Element(species).atomic_mass)  # Convert the mass to a float
                file.write(f"{i} {mass}\n")

            # Atoms section
            file.write("\nAtoms\n\n")
            for i, site in enumerate(structure.sites, start=1):
                atom_type = species_list.index(site.species_string) + 1
                file.write(f"{i} {atom_type} {site.x:.6f} {site.y:.6f} {site.z:.6f}\n")
    def get_phonopy_data(self, path):
        """
        Reads Phonopy output data and extracts:
        - Band structure
        - Total DOS
        - Projected DOS
        - Thermal properties

        :param path: Directory where Phonopy output files are stored.
        :return: Dictionary with phonon band structure, DOS, PDOS, and thermal properties.
        """
        data = {}

        # Load Phonopy object from phonopy_disp.yaml
        phonopy_yaml_file = os.path.join(path, "phonopy_disp.yaml")

        if os.path.exists(phonopy_yaml_file):
            phonon = phonopy.load(phonopy_yaml_file)  # Correct way to access phonopy object
        else:
            raise FileNotFoundError("Phonopy output files not found in the provided directory.")

        # Read Band Structure
        band_yaml_file = os.path.join(path, "band.yaml")
        if os.path.exists(band_yaml_file):
            with open(band_yaml_file, "r") as f:
                band_data = yaml.safe_load(f)

            qpoints = [entry["q-position"] for entry in band_data["phonon"]]
            frequencies = [entry["band"] for entry in band_data["phonon"]]

            data["band_structure"] = {
                "qpoints": np.array(qpoints),
                "frequencies": np.array([[band["frequency"] for band in bands] for bands in frequencies])
            }

        # Read Total DOS
        total_dos_file = os.path.join(path, "total_dos.dat")
        if os.path.exists(total_dos_file):
            dos_data = np.loadtxt(total_dos_file)
            data["total_dos"] = {
                "frequency_points": dos_data[:, 0],
                "total_dos": dos_data[:, 1]
            }

        # Read Projected DOS
        projected_dos_file = os.path.join(path, "projected_dos.dat")
        if os.path.exists(projected_dos_file):
            pdos_data = np.loadtxt(projected_dos_file)
            data["projected_dos"] = {
                "frequency_points": pdos_data[:, 0],
                "projected_dos": pdos_data[:, 1:]
            }

        # Read Thermal Properties
        thermal_yaml_file = os.path.join(path, "thermal_properties.yaml")
        if os.path.exists(thermal_yaml_file):
            with open(thermal_yaml_file, "r") as f:
                thermal_data = yaml.safe_load(f)

            data["thermal_properties"] = {
                "temperatures": np.array(thermal_data["temperature"]),
                "free_energy": np.array(thermal_data["free_energy"]),
                "entropy": np.array(thermal_data["entropy"]),
                "heat_capacity": np.array(thermal_data["heat_capacity"])
            }

        return data