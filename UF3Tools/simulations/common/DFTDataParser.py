import os
import re
import shutil
import numpy as np
from pymatgen.io.vasp import Poscar, Outcar
from pymatgen.core import Element

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
    def get_energy(self, filename=None):
        """Read the final energy from the OUTCAR file
        :param filename: Name of the OUTCAR file to read
        :return: Final energy in eV
        """
        # Use the OUTCAR file by default
        if filename is None:
            filename = os.path.join(self.directory, "OUTCAR")
        else:
            filename = os.path.join(self.directory, filename)
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

        print(f"LAMMPS data file '{outputFilename}' has been created.")