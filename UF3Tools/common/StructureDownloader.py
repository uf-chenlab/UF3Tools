from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os

class StructureDownloader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.mpr = MPRester(self.api_key)
    
    def download_structures(self, material_formula, directory):
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
            
            # Check if structures are the same
            if primitive_structure == conventional_structure:
                print(f"The primitive and conventional structures are identical for {material_id}.")
            
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