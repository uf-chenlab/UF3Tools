from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

class VASPModifier:
    def __init__(self, poscar_file):
        self.poscar_file = poscar_file
        self.structure = self.load_structure_from_poscar()

    def load_structure_from_poscar(self):
        return Structure.from_file(self.poscar_file)

    def generate_supercell(self, supercell_dimensions):
        supercell = self.structure * supercell_dimensions
        output_file = self.poscar_file + "_supercell"
        self.write_supercell_to_poscar(supercell, output_file)
        print(f'{supercell_dimensions} Supercell written to {output_file}')
        return supercell

    def write_supercell_to_poscar(self, supercell, output_file):
        poscar = Poscar(supercell)
        poscar.write_file(output_file)
        