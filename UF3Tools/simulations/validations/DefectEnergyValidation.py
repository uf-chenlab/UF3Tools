from .PotentialValidation import *
from ..simulations.DefectEnergySimulation import DefectEnergySimulation
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.vasp.inputs import Poscar
import re

class DefectEnergyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return None
    computational_weight=4
    required_supercell = (6,6,6)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, run_scripts, template_directory, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "DefectEnergySimulation",
            potential_file,
            lammps_model_file,
            dft_model_file,
            element_list,
            abs_validation_path,
            run_scripts,
            use_dft=False
        )
        self.default_kwargs = {
            "min_style": "cg",
            "pair_style": "uf3",
            "uf3_elements": element_list
        }
        self.kwargs = self.default_kwargs
        if parameters is not None:
            self.kwargs.update(parameters)
        self.dft_data = {}
        self.lammps_data = {}
        self.use_dft = False

        self.sim_template_file = os.path.join(
            template_directory,
            "in.defect_energy"
        )
        self.input_file = "lammps.in"
        self.simulation = DefectEnergySimulation(
            self.sim_template_file,
            self.lammps_model_file,
            self.potential_file,
            os.path.join(validation_folder_path, 'lammps', 'simulations'),
            run_scripts['lammps'],
            parameters=parameters
            
        )
        self.setup_input_file(os.path.join(self.simulation.get_directory(), self.sim_template_file), os.path.join(self.simulation.get_directory(), self.input_file), self.simulation.get_parameters())

    def retrieve_dft_result(self, path):
        from ..common.DFTDataParser import DFTDataParser
        abs_path = os.path.abspath(path)
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = dft_parser.get_defect_energy(abs_path)
        self.has_dft_data = True

    def start_validation(self, overwrite=True, local=False):
        if self.dft_data is None:
            raise ValueError("DFT data is not available. Please run DFT calculations first.")
        for defect_type, defect_data in self.dft_data.items():
            print(f'Testing defect type: {defect_type}')
            epa, structure = defect_data
            # create a folder for each defect type
            defect_folder = os.path.join(self.simulation.get_directory(), defect_type)
            os.makedirs(defect_folder, exist_ok=True)
            # create the lammps model from the structure 
            lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
            print("got structure")
            # write the lammps data file
            lammps_data.write_file(os.path.join(defect_folder, "data.lammps"))
            print("wrote structure")
            #create the simulation object 
            defect_simulation = DefectEnergySimulation(
                self.sim_template_file,
                os.path.join(defect_folder, "data.lammps"),
                self.potential_file,
                defect_folder,
                self.run_scripts['lammps'],
                parameters=self.kwargs
            )
            self.setup_input_file(os.path.join(defect_simulation.get_directory(), self.sim_template_file), os.path.join(defect_simulation.get_directory(), self.input_file), defect_simulation.get_parameters())
            print('created simulation')
            self.lammps_data[defect_type] = defect_simulation.run(overwrite=overwrite, local=local)
            print('ran simulation')
            
        output_dict = {}
        dft_dict = {}
        lammps_dict = {}
        #put only the dft energies in the dft output
        for defect_type, defect_data in self.dft_data.items():
            epa, structure = defect_data
            dft_dict[defect_type] = {
                "cohesive_energy": epa
            }
        self.dft_results = dft_dict
        for defect_type, defect_data in self.lammps_data.items():
            if defect_data is None:
                continue
            cohesive_energy = defect_data['cohesive_energy']
            lammps_dict[defect_type] = {
                "cohesive_energy": cohesive_energy
            }
        self.lammps_results = lammps_dict
        
        #compare delta energy between each defect type and the perfect structure (defecttype none)
        dft_perfect = self.dft_results.get('DefectType.None', {}).get('cohesive_energy', None)
        lammps_perfect = self.lammps_results.get('DefectType.None', {}).get('cohesive_energy', None)
        if dft_perfect is not None and lammps_perfect is not None:
            for defect_type in self.dft_results.keys():
                if defect_type == 'DefectType.None':
                    continue
                dft_defect_energy = self.dft_results[defect_type]['cohesive_energy']
                lammps_defect_energy = self.lammps_results.get(defect_type, {}).get('cohesive_energy', None)
                if dft_defect_energy is not None and lammps_defect_energy is not None:
                    abs_delta_dft = dft_defect_energy - dft_perfect
                    abs_delta_lammps = lammps_defect_energy - lammps_perfect
                    percent_delta_dft = (abs_delta_dft / (dft_perfect)) * 100 if dft_perfect != 0 else None
                    percent_delta_lammps = (abs_delta_lammps / (lammps_perfect)) * 100 if lammps_perfect != 0 else None
                    abs_error = (abs_delta_dft - abs_delta_lammps)/(abs_delta_dft) * 100 if abs_delta_dft != 0 else None
                    percent_error = (percent_delta_dft - percent_delta_lammps)/(percent_delta_dft) * 100 if percent_delta_dft is not None and percent_delta_lammps is not None and percent_delta_dft != 0 else None
                    output_dict[defect_type] = {
                        'abs_error': abs_error,
                        'percent_error': percent_error,
                        'percent_delta_dft': percent_delta_dft,
                        'percent_delta_lammps': percent_delta_lammps,
                        'abs_delta_dft': abs_delta_dft,
                        'abs_delta_lammps': abs_delta_lammps,
                        'dft_perfect': dft_perfect,
                        'lammps_perfect': lammps_perfect,
                        'dft_defect_energy': dft_defect_energy,
                        'lammps_defect_energy': lammps_defect_energy,
                    }
                    self.dft_results[defect_type]['delta_cohesive_energy'] = abs_delta_dft
                    self.lammps_results[defect_type]['delta_cohesive_energy'] = abs_delta_lammps
                    self.dft_results[defect_type]['percent_delta_cohesive_energy'] = percent_delta_dft
                    self.lammps_results[defect_type]['percent_delta_cohesive_energy'] = percent_delta_lammps
                    self.lammps_results[defect_type]['abs_error'] = abs_error
        return output_dict

    