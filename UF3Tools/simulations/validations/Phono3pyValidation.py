from .PotentialValidation import *
from phonolammps import Phonolammps
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import numpy as np
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from phonopy.interface.calculator import read_crystal_structure
from phono3py import Phono3py
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from tqdm import tqdm
from phonopy import Phonopy
import yaml
import subprocess
from ..simulations.Phono3pySimulation import Phono3pySimulation

class PhonopyValidation(PotentialValidation):
    @classmethod
    def get_required_base_simulations(cls):
        return Phono3pySimulation.get_required_sub_simulations()
    
    @classmethod 
    def get_required_single_crystal_simulations(cls):
        return Phono3pySimulation.get_required_sub_simulations()
    computational_weight=2
    required_supercell = (1,1,1)
    def __init__(self, potential_file, lammps_model_file, dft_model_file, element_list, validation_folder_path, run_scripts, template_directory, parameters=None):
        abs_validation_path = os.path.abspath(validation_folder_path)
        super().__init__(
            "Phono3pySimulation",
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
        self.dft_data = {
            "elastic_tensor": None,
            "bulk_modulus_voigt": None,
            "bulk_modulus_reuss": None,
            "bulk_modulus_hill": None,
            "shear_modulus_voigt": None,
            "shear_modulus_reuss": None,
            "shear_modulus_hill": None,
        }
        self.lammps_data = dict(self.dft_data)
        self.use_dft = False

        self.sim_template_file = os.path.join(
            template_directory,
            "in.phonopy"
        )
        self.input_file = "lammps.in"
        self.simulation = Phono3pySimulation(
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
        
    def get_lmp_calc(self, pot_file, specorder, lmp_command):
        # create ASE calc for LAMMPS
        calc = LAMMPS(
            label=f"tmp",
            keep_tmp_files=True,
            tmp_dir="tmp",
            files=[pot_file],
            specorder=specorder,
        )
        calc.set(
            command=lmp_command,
            pair_style="uf3 3",
            pair_coeff=[f"* * {pot_file} {self.element_list}"],
        )
        return calc
    def run_phonopy_jobs(self):
        num_jobs = 16

        with open("ir_grid_points.yaml") as f:
            data = yaml.safe_load(f)
            gps = [gp["grid_point"] for gp in data["ir_grid_points"]]
            print(f"Number of grid points: {len(gps)}")
            print(f"Number of jobs: {num_jobs}")
            print(f"Number of grid points per job: {len(gps) // num_jobs}")
            gp_lists = [[] for _ in range(num_jobs)]
            for i, gp in enumerate(gps):
                gp_lists[i % num_jobs].append(gp)

        with open("gps.txt", "w") as out:
            for gp_set in gp_lists:
                out.write(",".join(str(gp) for gp in gp_set) + "\n")

        job_dir = Path("slurm_jobs")
        job_dir.mkdir(exist_ok=True)

        shared_files = ["phono3py_disp.yaml", "fc2.hdf5", "fc3.hdf5"]
        missing = [f for f in shared_files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files: {missing}")

        for j, gp_set in enumerate(gp_lists, 1):
            job_subdir = job_dir / f"job_{j}"
            job_subdir.mkdir(exist_ok=True)

            for f in shared_files:
                shutil.copy(f, job_subdir / f)

            gp_arg = ",".join(str(gp) for gp in gp_set)
            job_script = f"""#!/bin/bash

        #SBATCH --job-name=ph3py_{j}
        #SBATCH --output=j_%x_%j.out
        #SBATCH --error=j_%x_%j.err
        #SBATCH --account=phillpot
        #SBATCH --qos=phillpot
        #SBATCH --ntasks=1
        #SBATCH --time=4:00:00
        #SBATCH --mem=32gb

        module purge
        module load conda
        export PATH=/home/ntaormina/miniconda38/envs/uf3_env/bin:$PATH
        cd {job_subdir.resolve()}

        phono3py-load --mesh 20 20 20 --gp {gp_arg} --write-gamma --br
        echo 'Done.'
        """
            script_path = job_subdir / "job.sh"
            script_path.write_text(job_script)
            subprocess.run(["sbatch", str(script_path)])

    def run_simulation(self):
        #get the relaxed structure from the base simulation
        results = self.simulation.process_simulation_results()
        print("results:", results)
        structure = results['relaxed_structure']
        #write the relaxed structure to a POSCAR file
        with open(self.lammps_model_file, 'w') as f:
            for line in structure:
                f.write(line + '\n')
        
        # generate displacements
        unitcell, _ = read_crystal_structure(self.lammps_model_file, interface_mode='lammps')
        ph3 = Phono3py(
            unitcell,
            supercell_matrix=[3,3,3],
            phonon_supercell_matrix=[6,6,6],
        )
        ph3.generate_displacements()
        ph3.save("phono3py_disp.yaml")
        print("Generated displacements")

        pot_file="GaNAlN.tersoff"
        lmp_command="./start_simulation_hpg.sh"
        specorder=self.element_list

        # get forces for FC2
        print("Computing forces for FC2")
        forces = []
        for sc in tqdm(ph3.phonon_supercells_with_displacements):
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc = self.get_lmp_calc(pot_file, specorder, lmp_command)
            f = atoms.get_forces()
            forces.append(f)

        # compute 2nd order force constants
        ph3.phonon_forces = np.array(forces)

        print("Computing FC2")
        ph3.produce_fc2()
        write_fc2_to_hdf5(
            ph3.fc2,
            p2s_map=ph3.phonon_primitive.p2s_map,
            physical_unit="eV/angstrom^2",
        )

        # get forces for FC3
        print("Computing forces for FC3")
        forces = []
        nat = len(ph3.supercells_with_displacements[0])
        for sc in tqdm(ph3.supercells_with_displacements):
            if sc is not None:
                atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
                atoms.calc = self.get_lmp_calc(pot_file, specorder, lmp_command)
                f = atoms.get_forces()
            else:
                f = np.zeros((nat, 3))
            forces.append(f)

        # compute 3rd order force constants
        ph3.forces = np.array(forces)

        print("Computing FC3")
        ph3.produce_fc3()
        write_fc3_to_hdf5(
            ph3.fc3,
            p2s_map=ph3.primitive.p2s_map,
        )
        subprocess.run(["phono3py-load", "--mesh", "20", "20", "20", "--wgp"])


    def start_validation(self, overwrite=True, local=False):
        self.lammps_data = self.run_simulation()
        if not self.dft_data or not self.lammps_data:
            raise ValueError("DFT or LAMMPS data is missing. Ensure both datasets are available.")

        comparison_results = {}

        # Compare Band Structure
        dft_bands = np.array(self.dft_data["band_structure"]["frequencies"])
        
        lammps_bands = np.concatenate(self.lammps_data["phonon_band_structure"][2], axis=0)  # Merge all q-points
        if dft_bands.shape == lammps_bands.shape:
            band_rmse = np.sqrt(np.mean((dft_bands - lammps_bands) ** 2))
        else:
            band_rmse = None  # Mismatch in shape, cannot compare directly

        comparison_results["band_structure_rmse"] = band_rmse

        # Compare Total DOS
        dft_dos = self.dft_data["total_dos"]["total_dos"]
        lammps_dos = self.lammps_data["phonon_dos"][1]

        if dft_dos.shape == lammps_dos.shape:
            dos_rmse = np.sqrt(np.mean((dft_dos - lammps_dos) ** 2))
        else:
            dos_rmse = None  # Mismatch in shape, cannot compare directly

        comparison_results["total_dos_rmse"] = dos_rmse * 100

        # # Compare Thermal Properties
        thermal_keys = ["free_energy", "entropy", "heat_capacity"]
        print(f'dft data: {self.dft_data}')
        for key in thermal_keys:
            dft_values = np.array(self.dft_data["thermal_properties"][key])
            lammps_values = np.array(self.lammps_data["thermal_properties"][key])

            if dft_values.shape == lammps_values.shape and np.all(dft_values != 0):
                percent_error = np.mean(np.abs((dft_values - lammps_values) / dft_values) * 100)
            else:
                percent_error = None  # Mismatch or division by zero

            comparison_results[f"{key}_percent_error"] = percent_error

        return comparison_results

    