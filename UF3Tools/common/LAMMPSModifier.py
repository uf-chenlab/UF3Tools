from pymatgen.io.lammps.data import LammpsData
from pymatgen.transformations.standard_transformations import SupercellTransformation
import numpy as np
from pymatgen.core.lattice import Lattice
import os

class BasisModifier:
    """
    Reads a crystal basis file, allows for scaling of its lattice constants
    and atom positions, and writes the result to a new basis file.
    """
    def __init__(self, basis_file):
        """
        Initializes the modifier by reading and parsing the basis file.

        Args:
            basis_file (str): The path to the basis file.
        """
        if not os.path.exists(basis_file):
            raise FileNotFoundError(f"Error: The basis file '{basis_file}' was not found.")
        
        self.basis_file = basis_file
        self._read_basis()

    def _read_basis(self):
        """
        Private method to read and parse the crystal definition from the basis file.
        This version is tailored to handle numeric atom types.
        """
        with open(self.basis_file, 'r') as f:
            lines = [line for line in f.readlines() if line.strip()] # Ignore empty lines
            
            self.title = lines[0].strip()
            self.num_basis_atoms = int(lines[1].strip())
            
            # Read cell vectors
            cell_flat = []
            cell_flat.extend(map(float, lines[2].strip().split()))
            cell_flat.extend(map(float, lines[3].strip().split()))
            cell_flat.extend(map(float, lines[4].strip().split()))
            self.cell = np.array(cell_flat).reshape(3, 3).T
            
            # Read dimensions
            self.dims = tuple(map(int, lines[5].strip().split()))
            
            # Read basis atoms
            self.basis_atoms = []
            for i in range(self.num_basis_atoms):
                parts = lines[6+i].strip().split()
                self.basis_atoms.append({
                    'type': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3])
                })
    def scale_lattice_anisotropic(self, scale_a=1.0, scale_b=1.0, scale_c=1.0):
        """
        Scales the lattice constants and atomic positions anisotropically.

        This method scales each lattice vector (a, b, c) by a specific factor
        and applies the corresponding transformation to the absolute Cartesian
        coordinates of the atoms, preserving their fractional coordinates.

        Args:
            scale_a (float): The scaling factor for the 'a' lattice vector. Defaults to 1.0.
            scale_b (float): The scaling factor for the 'b' lattice vector. Defaults to 1.0.
            scale_c (float): The scaling factor for the 'c' lattice vector. Defaults to 1.0.
        """
        # Store the original cell matrix to calculate the coordinate transformation
        original_cell = self.cell.copy()

        # Create the new cell matrix by scaling each lattice vector (column)
        new_cell = original_cell.copy()
        new_cell[:, 0] *= scale_a
        new_cell[:, 1] *= scale_b
        new_cell[:, 2] *= scale_c
        
        # The transformation from original to new Cartesian coordinates is derived
        # from keeping the fractional coordinates constant.
        # pos_new = C_new * fractional_coords
        # fractional_coords = inv(C_old) * pos_old
        # So, pos_new = C_new * inv(C_old) * pos_old
        try:
            inv_original_cell = np.linalg.inv(original_cell)
        except np.linalg.LinAlgError:
            print("Error: The original cell matrix is singular and cannot be inverted.")
            return

        transform_matrix = new_cell @ inv_original_cell
        
        # Apply the transformation to each basis atom's coordinates
        for atom in self.basis_atoms:
            original_pos = np.array([atom['x'], atom['y'], atom['z']])
            new_pos = transform_matrix @ original_pos
            atom['x'] = new_pos[0]
            atom['y'] = new_pos[1]
            atom['z'] = new_pos[2]
            
        # Update the cell matrix to the new scaled version
        self.cell = new_cell
        
        print(f"Scaled lattice constants by factors a={scale_a}, b={scale_b}, c={scale_c}.")
        
    def scale_lattice(self, scale_factor):
        """
        Scales the lattice constants and atomic positions by a given factor.

        Args:
            scale_factor (float): The factor to multiply the lattice constants
                                  and coordinates by (e.g., 1.02 for 2% expansion).
        """
        # Scale the cell vectors
        self.cell *= scale_factor
        
        # Scale the coordinates of each basis atom
        for atom in self.basis_atoms:
            atom['x'] *= scale_factor
            atom['y'] *= scale_factor
            atom['z'] *= scale_factor
        print(f"Scaled lattice and atomic positions by a factor of {scale_factor}.")

    def write_modified_basis(self, output_file):
        """
        Writes the current (possibly modified) basis data to a new file.

        Args:
            output_file (str): The path for the new basis file.
        """
        with open(output_file, 'w') as f:
            f.write(f"{self.title}\n")
            f.write(f"{self.num_basis_atoms}\n")
            # Reshape cell matrix back to write row by row
            cell_rows = self.cell.T
            for row in cell_rows:
                f.write(f"{row[0]:.6f}\t{row[1]:.6f}\t{row[2]:.6f}\n")
            f.write(f"{self.dims[0]} {self.dims[1]} {self.dims[2]}\n")
            for atom in self.basis_atoms:
                f.write(f"{atom['type']}  {atom['x']:.6f}  {atom['y']:.6f}  {atom['z']:.6f}\n")
        print(f"Modified basis file written to '{output_file}'.")
        
class WurtziteBuilder:
    """
    A class to build a perfect wurtzite crystal structure based on a basis file
    and write it to a LAMMPS data file.

    This is a Python translation of the core crystal generation logic from the
    provided FORTRAN code, ignoring the dislocation generation parts.
    """
    def __init__(self, basis_file):
        """
        Initializes the builder by reading and parsing the basis file.

        Args:
            basis_file (str): The path to the basis file.
        """
        if not os.path.exists(basis_file):
            raise FileNotFoundError(f"Error: The basis file '{basis_file}' was not found.")
            
        self.basis_file = basis_file
        self._read_basis()
        self.atoms = []

    def _read_basis(self):
        """
        Private method to read and parse the crystal definition from the basis file.
        """
        with open(self.basis_file, 'r') as f:
            lines = f.readlines()
            
            # Read title
            self.title = lines[0].strip()
            
            # Read number of basis atoms
            self.num_basis_atoms = int(lines[1].strip())
            
            # Read cell vectors (lattice parameters)
            # FORTRAN stores in column-major, we'll use a 3x3 matrix
            # cell(1) cell(2) cell(3) -> a_x, a_y, a_z
            # cell(4) cell(5) cell(6) -> b_x, b_y, b_z
            # cell(7) cell(8) cell(9) -> c_x, c_y, c_z
            cell_flat = []
            cell_flat.extend(map(float, lines[2].strip().split()))
            cell_flat.extend(map(float, lines[3].strip().split()))
            cell_flat.extend(map(float, lines[4].strip().split()))
            self.cell = np.array(cell_flat).reshape(3, 3).T

            # Read number of unit cells to generate in each direction
            nx, ny, nz = map(int, lines[5].strip().split())
            self.dims = (nx, ny, nz)

            # Read basis atom names and fractional coordinates
            self.basis_atoms = []
            for i in range(self.num_basis_atoms):
                parts = lines[6+i].strip().split()
                self.basis_atoms.append({
                    'name': parts[0],
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3])
                })

    def build_crystal(self, supercell_dims=(1, 1, 1)):
        """
        Generates the atomic coordinates for the crystal by replicating
        the unit cell. Can create supercells by specifying multipliers.

        Args:
            supercell_dims (tuple, optional): A tuple of 3 integers (mx, my, mz)
                to multiply the base dimensions by. Defaults to (1, 1, 1), which
                creates the base crystal defined in the basis file.
        """
        self.atoms = []
        atom_id = 0
        
        # Get base dimensions from the basis file
        nx_base, ny_base, nz_base = self.dims
        
        # Calculate final dimensions based on supercell multipliers
        nx = nx_base * supercell_dims[0]
        ny = ny_base * supercell_dims[1]
        nz = nz_base * supercell_dims[2]

        print(f"Building a crystal with dimensions: {nx} x {ny} x {nz} unit cells.")

        # This section replicates the nested DO loops from the FORTRAN code
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    for n_idx, basis_atom in enumerate(self.basis_atoms):
                        atom_id += 1
                        
                        # Get basis atom position vector
                        base_pos = np.array([basis_atom['x'], basis_atom['y'], basis_atom['z']])
                        
                        # Get lattice translation vector
                        translation = i * self.cell[:, 0] + j * self.cell[:, 1] + k * self.cell[:, 2]

                        # Calculate final atomic position
                        pos = base_pos + translation

                        # Determine atom type based on the logic from the FORTRAN code
                        # Atoms at basis indices 0, 1, 4, 5 (1, 2, 5, 6 in FORTRAN) are type 1
                        # All others are type 2
                        atom_type = 1 if n_idx in [0, 1, 4, 5] else 2

                        self.atoms.append({
                            'id': atom_id,
                            'type': atom_type,
                            'x': pos[0],
                            'y': pos[1],
                            'z': pos[2]
                        })
        print(f"Successfully built crystal with {len(self.atoms)} atoms.")

    def write_lammps_data(self, output_file, padding=0.0):
        """
        Writes the generated atom data to a file in LAMMPS data format.

        Args:
            output_file (str): The path to the output data file.
            padding (float): Extra padding to add to the simulation box boundaries.
        """
        if not self.atoms:
            print("Warning: No atoms have been generated. Call build_crystal() first.")
            return

        # Determine simulation box boundaries
        coords = np.array([atom['x'] for atom in self.atoms])
        xlo, xhi = coords.min(), coords.max()
        coords = np.array([atom['y'] for atom in self.atoms])
        ylo, yhi = coords.min(), coords.max()
        coords = np.array([atom['z'] for atom in self.atoms])
        zlo, zhi = coords.min(), coords.max()
        

        with open(output_file, 'w') as f:
            # Header
            f.write(f"{self.title}\n\n")
            f.write(f"{len(self.atoms):10d} atoms\n")
            f.write(f"{2:10d} atom types\n\n") # Hardcoded as 2 types from FORTRAN

            # Box dimensions
            f.write(f"{xlo - 1:15.6f} {xhi + 1:15.6f} xlo xhi\n")
            f.write(f"{ylo - 0.65:15.6f} {yhi + 0.65:15.6f} ylo yhi\n")
            f.write(f"{zlo - 0.3:15.6f} {zhi + 0.3:15.6f} zlo zhi\n\n")

            # Masses - hardcoded from the FORTRAN file
            f.write("Masses\n\n")
            f.write(f"{1:10d} {26.981:20.6f}\n") # Al
            f.write(f"{2:10d} {14.0067:20.6f}\n\n") # N

            # Atoms
            f.write("Atoms\n\n")
            for atom in self.atoms:
                f.write(
                    f"{atom['id']:10d} {atom['type']:3d} {atom['x']:20.6f} {atom['y']:20.6f} {atom['z']:20.6f}\n"
                )
        
        print(f"LAMMPS data file written to '{output_file}'.")

class LAMMPSModifier:
    def __init__(self, lammps_file, atom_style='atomic'):
        self.lammps_file = os.path.abspath(lammps_file)
        self.atom_style = atom_style
        self.structure = self.load_structure_from_file()
    
    def load_structure_from_file(self):
        """Loads structure from a LAMMPS data file."""
        return LammpsData.from_file(self.lammps_file, atom_style=self.atom_style).structure
    def get_structure(self):
        """Returns the structure loaded from the LAMMPS data file."""
        return self.structure
    def get_lammps_file(self):
        return self.lammps_file
    def get_lattice(self):
        """Returns the lattice of the structure."""
        return self.structure.lattice if self.structure else None
    def get_elements(self):
        """Returns a space-separated string of element symbols present in the structure."""
        elements = sorted(set(site.specie.symbol for site in self.structure))
        return " ".join(elements)
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

    def add_atom_types(self, new_atom_masses: dict):
        """
        Adds new atom types and their masses to the LAMMPS data file.

        This method directly modifies the data file in place.

        Args:
            new_atom_masses (dict): A dictionary where keys are the new atom
                                    symbols (str) and values are their masses (float).
                                    Example: {'Ga': 69.723, 'As': 74.922}
        """
        try:
            with open(self.lammps_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: The file {self.lammps_file} was not found.")
            return

        # --- Find key locations and current values ---
        atom_types_line_index = -1
        masses_section_index = -1
        current_n_types = 0

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if "atom types" in stripped_line:
                try:
                    # Capture the current number of atom types
                    current_n_types = int(stripped_line.split()[0])
                    atom_types_line_index = i
                except (ValueError, IndexError):
                    continue  # Ignore lines that contain the text but aren't the header
            elif stripped_line.lower() == "masses":
                masses_section_index = i + 1
        
        # --- Validate that we found the required sections ---
        if atom_types_line_index == -1:
            print("Error: Could not find the 'atom types' header in the data file.")
            return
        if masses_section_index == -1:
            print("Error: Could not find the 'Masses' section in the data file.")
            return

        # --- Update the atom types count ---
        n_new_types = len(new_atom_masses)
        new_total_types = current_n_types + n_new_types
        
        # Reconstruct the line to preserve original spacing and comments
        parts = lines[atom_types_line_index].split('atom types')
        lines[atom_types_line_index] = f"{new_total_types} atom types{parts[1]}"
        
        # --- Find where to insert new mass entries ---
        # The insertion point is the first non-mass line after the 'Masses' header,
        # which is typically a blank line or the start of a new section (e.g., "Atoms").
        insertion_index = -1
        for i in range(masses_section_index + 1, len(lines)):
            line = lines[i].strip()
            if not line or (line and line[0].isalpha()):
                insertion_index = i
                break
        
        # If no boundary is found, Masses is the last section. Append to the end.
        if insertion_index == -1:
            insertion_index = len(lines)
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += '\n' # Ensure there's a newline before appending

        # --- Generate the new mass lines ---
        new_mass_lines = []
        atom_type_index = current_n_types
        for symbol, mass in new_atom_masses.items():
            atom_type_index += 1
            # Format: type_index  mass      # symbol
            new_line = f"  {atom_type_index} {mass:<10.4f} # {symbol}\n"
            new_mass_lines.append(new_line)
        
        # --- Insert the new lines into the list of all lines ---
        lines[insertion_index:insertion_index] = new_mass_lines

        # --- Write the modified content back to the file ---
        try:
            with open(self.lammps_file, 'w') as f:
                f.writelines(lines)
            print(f"added {n_new_types} atom type(s) to {os.path.basename(self.lammps_file)}.")
            print(f"   New total atom types: {new_total_types}")
        except IOError as e:
            print(f"Error writing to file {self.lammps_file}: {e}")

# Example usage:
# modifier = LAMMPSModifier("input.data")
# modifier.generate_supercell((2, 2, 2))
# modifier.generate_surface_slab((2, 2, 2), vacuum_padding=10)