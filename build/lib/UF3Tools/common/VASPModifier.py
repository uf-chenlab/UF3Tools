from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

class VASPModifier:
    def __init__(self, poscar_file):
        self.poscar_path = Path(poscar_file)
        self.structure = self.load_structure_from_poscar()

    def load_structure_from_poscar(self):
        return Structure.from_file(self.poscar_path)

    def _validate_dims(self, dims):
        if not isinstance(dims, (list, tuple)) or len(dims) != 3:
            raise ValueError("supercell_dimensions must be a 3-element list/tuple like [2,2,2].")
        if any((not isinstance(x, int) or x <= 0) for x in dims):
            raise ValueError("supercell_dimensions must contain positive integers.")
        return tuple(dims)

    def _output_path(self, dims):
        a, b, c = dims
        # keep original extension if present; many POSCARs have no suffix
        if self.poscar_path.suffix:
            return self.poscar_path.with_name(f"{self.poscar_path.stem}_supercell_{a}x{b}x{c}{self.poscar_path.suffix}")
        return self.poscar_path.with_name(f"{self.poscar_path.name}_supercell_{a}x{b}x{c}")

    def generate_supercell(self, supercell_dimensions):
        dims = self._validate_dims(supercell_dimensions)
        supercell = self.structure * dims
        output_file = self._output_path(dims)
        self.write_supercell_to_poscar(supercell, output_file)
        print(f"{list(dims)} supercell written to {output_file}")
        return supercell

    def write_supercell_to_poscar(self, supercell, output_file):
        # write a POSCAR with dimension-tagged filename
        Poscar(supercell).write_file(str(output_file))

        