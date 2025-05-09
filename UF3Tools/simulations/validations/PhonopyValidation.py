from .PotentialValidation import *
from phonolammps import Phonolammps
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy import Phonopy
from ..simulations.PhonopySimulation import PhonopySimulation

class PhonopyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return PhonopySimulation.get_required_sub_simulations()
    
    @classmethod 
    def get_required_single_crystal_simulations(cls):
        return PhonopySimulation.get_required_sub_simulations()
    computational_weight=2
    required_supercell = (1,1,1)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, run_scripts, template_directory, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "PhonopySimulation",
            potential_file,
            lammps_model_file,
            dft_model_file,
            element_list,
            abs_validation_path,
            run_scripts,
            use_dft=False
        )
        print(f'parameters: {parameters}')
        self.default_kwargs = {
            "min_style": "cg",
            "uf3_elements": element_list
        }
        self.kwargs = self.default_kwargs
        print(f'Kwargs: {self.kwargs}')
        if parameters is not None:
            self.kwargs.update(parameters)
        print(f'Kwargs: {self.kwargs}')
        self.lammps_data = dict(self.dft_data)
        self.use_dft = False

        self.sim_template_file = os.path.join(
            template_directory,
            "in.phonopy"
        )
        self.input_file = "lammps.in"
        self.simulation = PhonopySimulation(
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
        print(f'path: {path}')
        abs_path = os.path.abspath(path)
        print(f'abs_path: {abs_path}')
        dft_parser = DFTDataParser(abs_path)
        self.dft_data = dft_parser.get_phonopy_data(abs_path)
        print(f'dft data: {self.dft_data}')
        self.has_dft_data = True
    def run_simulation(self):
        #get the relaxed structure from the base simulation
        results = self.simulation.process_simulation_results()
        print("results:", results)
        structure = results['relaxed_structure']
        #write the relaxed structure to a POSCAR file
        with open(self.lammps_model_file, 'w') as f:
            for line in structure:
                f.write(line + '\n')
        
        phlammps = Phonolammps(os.path.join(self.simulation.get_directory(), 'lammps.in'),
                       supercell_matrix=[[3, 0, 0],
                                         [0, 3, 0],
                                         [0, 0, 3]])
        unitcell = phlammps.get_unitcell()
        force_constants = phlammps.get_force_constants()
        supercell_matrix = phlammps.get_supercell_matrix()
        phonon = Phonopy(unitcell,
                        supercell_matrix)
        phonon.set_force_constants(force_constants)
        phonon.run_mesh([30,30,30], with_eigenvectors=True, is_mesh_symmetry=False)
        path = [[
            [2/3, 1/3, 0.0],   # K
            [0.0, 0.0, 0.0],   # Γ
            [0.5, 0.0, 0.0],   # M
            [2/3, 1/3, 0.0],   # K
            [2/3, 1/3, 0.5],   # H
            [0.0, 0.0, 0.5],   # A
            [0.5, 0.0, 0.5],   # L
            [0.5, 0.0, 0.0],   # M
            [0.0, 0.0, 0.0],   # Γ
            [0.0, 0.0, 0.5],   # A
        ]]
        labels = ["K", "$\\Gamma$", "M", "K", "H", "A", "L", "M", "$\\Gamma$", "A"]

        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
        phonon.run_band_structure(qpoints, labels)
        #phonon.plot_band_structure().show()
        phonon.run_total_dos()
        #phonon.plot_total_dos().show()
        phonon.run_thermal_properties(t_step=10,
                              t_max=1000,
                              t_min=0)
        tp_dict = phonon.get_thermal_properties_dict()
        temperatures = tp_dict['temperatures']
        free_energy = tp_dict['free_energy']
        entropy = tp_dict['entropy']
        heat_capacity = tp_dict['heat_capacity']
        lammps_results = {
            "phonon_band_structure": phonon.get_band_structure(),
            "phonon_dos": phonon.get_total_DOS(),
            "thermal_properties": {
                "temperatures": temperatures,
                "free_energy": free_energy,
                "entropy": entropy,
                "heat_capacity": heat_capacity
            }
        }
        return lammps_results
    def start_validation(self, overwrite=True, local=False):
        self.lammps_data = self.run_simulation()
        if not self.dft_data or not self.lammps_data:
            raise ValueError("DFT or LAMMPS data is missing. Ensure both datasets are available.")

        comparison_results = {}

        dft_bands = np.array(self.dft_data["band_structure"]["frequencies"])
        lammps_bands = np.concatenate(self.lammps_data["phonon_band_structure"][2], axis=0)

        if np.any(lammps_bands < -0.5):
            print("Negative frequencies detected in band structure. Skipping RMSE calculation.")
            band_rmse = None
        elif dft_bands.shape != lammps_bands.shape:
            print("Mismatch in band structure shape. Skipping RMSE calculation.")
            band_rmse = None
        else:
            band_rmse = np.sqrt(np.mean((dft_bands - lammps_bands) ** 2))

        comparison_results["band_structure_rmse"] = band_rmse

        # Compare Total DOS
        dft_dos = self.dft_data["total_dos"]["total_dos"]
        lammps_dos = self.lammps_data["phonon_dos"][1]

        if dft_dos.shape == lammps_dos.shape:
            dos_rmse = np.sqrt(np.mean((dft_dos - lammps_dos) ** 2))
        else:
            dos_rmse = None  # Mismatch in shape, cannot compare directly

        comparison_results["total_dos_rmse"] = dos_rmse


        temps_dft = np.array(self.dft_data["thermal_properties"]["temperatures"])
        temps_lammps = np.array(self.lammps_data["thermal_properties"]["temperatures"])

        if not np.array_equal(temps_dft, temps_lammps):
            raise ValueError("Temperature arrays do not match between DFT and LAMMPS.")

        thermal_keys = ["free_energy", "entropy", "heat_capacity"]

        for key in thermal_keys:
            dft_values = np.array(self.dft_data["thermal_properties"][key])
            lammps_values = np.array(self.lammps_data["thermal_properties"][key])

            if dft_values.shape != lammps_values.shape:
                comparison_results[f"{key}_percent_error"] = None
                continue

            nonzero_mask = dft_values != 0
            if not np.any(nonzero_mask):
                comparison_results[f"{key}_percent_error"] = None
                continue

            rel_error = np.abs((dft_values[nonzero_mask] - lammps_values[nonzero_mask]) / dft_values[nonzero_mask])
            comparison_results[f"{key}_percent_error"] = np.mean(rel_error) * 100

        return comparison_results

    