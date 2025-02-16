from pymatgen.io.lammps.data import LammpsData
from pymatgen.transformations.standard_transformations import SupercellTransformation
import numpy as np
from pymatgen.core.lattice import Lattice
import os

class LAMMPSModifier:
    def __init__(self, lammps_file, atom_style='atomic'):
        self.lammps_file = os.path.abspath(lammps_file)
        self.atom_style = atom_style
        self.structure = self.load_structure_from_file()
    
    def load_structure_from_file(self):
        """Loads structure from a LAMMPS data file."""
        return LammpsData.from_file(self.lammps_file, atom_style=self.atom_style).structure

    def generate_supercell(self, supercell_dimensions):
        """Creates a supercell using SupercellTransformation."""
        transformation = SupercellTransformation(supercell_dimensions)
        supercell = transformation.apply_transformation(self.structure)
        
        output_file = self.lammps_file.replace(".data", "_supercell.data")
        self.write_lammps_data(supercell, output_file)
        print(f"{supercell_dimensions} Supercell written to {output_file}")
        return output_file
    
    def generate_surface_slab(self, supercell_dimensions, vacuum_padding):
        """Creates a surface slab with vacuum padding along the Z direction."""
        slab_dimensions = (supercell_dimensions[0], supercell_dimensions[1], supercell_dimensions[2])
        transformation = SupercellTransformation(slab_dimensions)
        slab = transformation.apply_transformation(self.structure)
        
        lammps_data = LammpsData.from_structure(slab, atom_style=self.atom_style)
        lammps_data.box.bounds[2][1] = lammps_data.box.bounds[2][1] + 15
        
        output_file = self.lammps_file.replace(".data", "_slab.data")
        lammps_data.write_file(output_file)
        print(f"Surface slab with vacuum written to {output_file}")
        return output_file
    
    def write_lammps_data(self, structure, output_file):
        """Writes the modified structure back to a LAMMPS data file."""
        lammps_data = LammpsData.from_structure(structure, atom_style=self.atom_style)
        lammps_data.write_file(output_file)

# Example usage:
# modifier = LAMMPSModifier("input.data")
# modifier.generate_supercell((2, 2, 2))
# modifier.generate_surface_slab((2, 2, 2), vacuum_padding=10)