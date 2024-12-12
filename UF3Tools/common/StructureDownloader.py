from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar, Outcar
from pymatgen.core import Element
import re
import shutil
import numpy as np
import os

class StructureDownloader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.mpr = MPRester(self.api_key)
        
    def get_structure_list(self, material_formula):
        summaries = self.mpr.summary.search(formula=material_formula)
        mpids = []
        for summary in summaries:
            print(f"{summary.material_id}: {summary.formula_pretty}")
            #convert summary.material_id to a string and append to mpids
            mpid = str(summary.material_id)
            mpids.append(mpid)
        return mpids
            
    
    def download_all_structures(self, material_formula, directory):
        # Get all materials associated with the material formula
        summaries = self.mpr.summary.search(formula=material_formula)
        
        if not summaries:
            print(f"No materials found for {material_formula}")
            return
        
        for summary in summaries:
            material_id = summary.material_id
            
            # Get the structure
            structure = self.mpr.get_structure_by_material_id(material_id)
            
            # Print information about the primitive unit cell
            primitive_analyzer = SpacegroupAnalyzer(structure)
            primitive_structure = primitive_analyzer.get_primitive_standard_structure()
            print(f"Primitive structure for {material_id} with space group: {primitive_analyzer.get_space_group_symbol()}")
            
            # Convert to conventional unit cell
            conventional_analyzer = SpacegroupAnalyzer(structure)
            conventional_structure = conventional_analyzer.get_conventional_standard_structure()
            print(f"Conventional structure for {material_id} with space group: {conventional_analyzer.get_space_group_symbol()}")
            
            
            # Get the crystal system
            crystal_system = conventional_analyzer.get_crystal_system()
            
            # Create subdirectory for the crystal system if it doesn't exist
            system_directory = os.path.join(directory, crystal_system.capitalize())
            os.makedirs(system_directory, exist_ok=True)
            
            # Construct filename and save the POSCAR file
            filename = f'{crystal_system}_POSCAR_{material_id.split("-")[1]}'
            poscar_path = os.path.join(system_directory, filename)
            conventional_structure.to(fmt='poscar', filename=poscar_path)
            print(f'Saved {material_id} as {poscar_path}')
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
    
    def download_single_structure(self, mpid, directory):
        # Get the structure
        structure = self.mpr.get_structure_by_material_id(mpid)
        
        # Print information about the primitive unit cell
        primitive_analyzer = SpacegroupAnalyzer(structure)
        primitive_structure = primitive_analyzer.get_primitive_standard_structure()
        print(f"Primitive structure for {mpid} with space group: {primitive_analyzer.get_space_group_symbol()}")
        
        # Convert to conventional unit cell
        conventional_analyzer = SpacegroupAnalyzer(structure)
        conventional_structure = conventional_analyzer.get_conventional_standard_structure()
        print(f"Conventional structure for {mpid} with space group: {conventional_analyzer.get_space_group_symbol()}")
        
        # Check if structures are the same
        if primitive_structure == conventional_structure:
            print(f"The primitive and conventional structures are identical for {mpid}.")
        
        # Get the crystal system
        crystal_system = conventional_analyzer.get_crystal_system()
        
        # Create subdirectory for the crystal system if it doesn't exist
        system_directory = directory
        os.makedirs(system_directory, exist_ok=True)
        
        # Construct filename and save the POSCAR file
        filename = f'POSCAR_{mpid.split("-")[1]}'
        poscar_path = os.path.join(system_directory, filename)
        conventional_structure.to(fmt='poscar', filename=poscar_path)
        print(f'Saved {mpid} as {poscar_path}')
        
        # Convert the POSCAR file to a LAMMPS data file
        lammps_filename = f'LAMMPS_{mpid.split("-")[1]}.data'
        lammps_path = os.path.join(system_directory, lammps_filename)
        self.poscarToLAMMPS(poscar_path, lammps_path)
        print(f'Converted {poscar_path} to {lammps_path}')