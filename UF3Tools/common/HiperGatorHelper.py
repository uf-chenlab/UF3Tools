import os
import shutil
import subprocess
import getpass

class HiperGatorHelper:
    def __init__(self, job_limit=10, cpu_limit = 150):
        """
        Initializes the HiperGatorHelper class, sets the current working directory,
        retrieves the current user, and retrieves the list of jobs that were running 
        for the user prior to the creation of this object.
        """
        print('HiperGator Helper initialized')
        self.current_directory = os.getcwd()
        self.user = getpass.getuser()
        print(f"Current user: {self.user}")
        self.jobs = []
        # self.previous_jobs = self._get_currently_running_jobs()
        # self.nodes, self.cpus = self.get_resource_usage()
        print(f"Current directory: {self._poll_jobs()}")

    def create_directory(self, directory_name):
        """
        Creates a directory with the given name if it does not already exist.

        :param directory_name: Name of the directory to create.
        """
        os.makedirs(directory_name, exist_ok=True)

    def change_directory(self, directory_name):
        """
        Changes the current working directory to the specified directory.

        :param directory_name: Name of the directory to change to.
        """
        os.chdir(directory_name)

    def copy_files(self, source_file_path, destination_file_path):
        """
        Copies a file from the source path to the destination path.

        :param source_file_path: Path of the file to copy.
        :param destination_file_path: Path where the file should be copied.
        """
        shutil.copy(source_file_path, destination_file_path)

    def submit_job(self, runjob_path):
        """
        Submits a job using the 'sbatch' command and stores the job ID.

        :param runjob_path: Path to the job script to be submitted.
        """
        result = subprocess.run(['sbatch', runjob_path], check=True, capture_output=True, text=True)
        job_id = self._extract_job_id(result.stdout)
        if job_id:
            self.jobs.append(job_id)
            print(f"Job submitted with ID: {job_id}")
        else:
            print("Failed to extract Job ID from submission output.")
    def _poll_jobs(self):
        """Sends a 'squeue' command to check the status of the jobs.
        :return: the raw output of the 'squeue' command.
        """
        result = subprocess.run(['squeue', '--user', self.user, '--noheader', '--format=%all'], capture_output=True, text=True)
        return result.stdout

    def _extract_job_id(self, submission_output):
        """
        Extracts the job ID from the sbatch submission output.

        :param submission_output: The output string from the sbatch command.
        :return: The extracted job ID, or None if extraction fails.
        """
        parts = submission_output.split()
        for part in parts:
            if part.isdigit():
                return part
        return None

    def run_command(self, command):
        """
        Runs a shell command in the current working directory.

        :param command: The shell command to execute.
        """
        subprocess.run(command, shell=True)

    def run_script_in_all_subdirectories(self, script_name):
        """
        Executes a script in all subdirectories of the current directory.

        :param script_name: The name of the script to run.
        """
        for directory in os.listdir():
            if os.path.isdir(directory) and not directory.startswith('.') and not directory.startswith('basic_files') and not directory.startswith('template'):
                os.chdir(directory)
                print(f'Running {script_name} in {directory}')
                self.run_command(f'chmod 777 {script_name} && ./{script_name}')
                os.chdir('..')

    def delete_last_x_jobs(self, x, include_previous=False):
        """
        Deletes the last x jobs from the tracked list and cancels them using 'scancel'.
        
        :param x: Number of jobs to delete.
        :param include_previous: If True, considers both currently tracked jobs and previously running jobs. 
        """
        if include_previous:
            jobs_to_consider = self.previous_jobs + self.jobs
        else:
            jobs_to_consider = self.jobs

        to_delete = jobs_to_consider[-x:]

        for job_id in to_delete:
            self._delete_job(job_id)
        
        # Update the jobs lists accordingly
        self.jobs = [job for job in self.jobs if job not in to_delete]
        if include_previous:
            self.previous_jobs = [job for job in self.previous_jobs if job not in to_delete]

    def delete_specific_job(self, job_id):
        """
        Deletes a specific job by its ID from the tracked list and cancels it using 'scancel'.

        :param job_id: The job ID to delete.
        """
        if job_id in self.jobs:
            self._delete_job(job_id)
            self.jobs.remove(job_id)
        elif job_id in self.previous_jobs:
            self._delete_job(job_id)
            self.previous_jobs.remove(job_id)
        else:
            print(f"Job ID {job_id} not found in the list of tracked jobs.")

    def _delete_job(self, job_id):
        """
        Cancels a job using its job ID with the 'scancel' command.

        :param job_id: The job ID to cancel.
        """
        subprocess.run(['scancel', job_id], check=True)
        print(f"Deleted job with ID: {job_id}")

    def check_jobs_status(self):
        """
        Checks the status of all tracked jobs. If a job is no longer running, it is removed from the list.
        """
        for job_id in self.jobs[:]:
            if not self._is_job_running(job_id):
                self.jobs.remove(job_id)

        for job_id in self.previous_jobs[:]:
            if not self._is_job_running(job_id):
                self.previous_jobs.remove(job_id)

    def _is_job_running(self, job_id):
        """
        Checks if a job with a given ID is still running using 'squeue' for the current user.

        :param job_id: The job ID to check.
        :return: True if the job is still running, False otherwise.
        """
        result = subprocess.run(['squeue', '--user', self.user, '--job', job_id], capture_output=True, text=True)
        if job_id in result.stdout:
            return True
        else:
            print(f"Job {job_id} is no longer running.")
            return False
    def _get_job_ids_from_output(self, output):
        """
        Extracts job IDs from the output of a command that lists job IDs.

        :param output: The output string to extract job IDs from.
        :return: A list of job IDs.
        """
        return output.strip().split('\n')
    def _get_resource_usage_from_output(self, output):
        """
        Extracts the total number of nodes and CPUs being used from the output of 'squeue'.

        :param output: The output string to extract resource usage information from.
        :return: The total number of nodes and CPUs being used.
        """
        total_nodes = 0
        total_cpus = 0

        lines = output.strip().split('\n')
        for line in lines:
            if line:
                nodes, cpus = map(int, line.split())
                total_nodes += nodes
                total_cpus += cpus
        return total_nodes, total_cpus
    def _get_currently_running_jobs(self):
        """
        Retrieves a list of currently running jobs for the current user by querying 'squeue'.

        :return: A list of job IDs that are currently running.
        """
        result = subprocess.run(['squeue', '--user', self.user, '--noheader', '--format=%A'], capture_output=True, text=True)
        job_ids = self._get_job_ids_from_output(result.stdout)
        return job_ids if job_ids != [''] else []
    def get_job_count(self):
        """
        Returns the total number of tracked jobs.
        """
        return len(self._get_currently_running_jobs())
    def get_resource_usage(self):
        """
        Retrieves the number of nodes and CPUs currently being used by the specified user.

        :param user: The username for which to check resource usage.
        :return: A dictionary with the total number of nodes and CPUs being used.
        """
        result = subprocess.run(['squeue', '--user', self.user, '--noheader', '--format=%D %C'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to retrieve resource usage information.")
            return None

        total_nodes, total_cpus = self._get_resource_usage_from_output(result.stdout)
        print(f"Total nodes in use: {total_nodes}, Total CPUs in use: {total_cpus}")
        return total_nodes, total_cpus
    def get_current_jobs(self):
        """
        Returns the list of currently tracked jobs.
        """
        #print the previous jobs and the current jobs
        print(f"Previous jobs: {self.previous_jobs}")
        print(f"Current jobs: {self.jobs}")
        return self.jobs
    def get_job_summary(self):
        """
        Retrieves the currently running jobs and resource usage (nodes and CPUs) for the current user
        and stores them in the class attributes `previous_jobs`, `total_nodes`, and `total_cpus`.
        """
        result = subprocess.run(['squeue', '--user', self.user, '--noheader', '--format=%A %D %C'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to retrieve job and resource usage information.")
            return

        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line:
                job_id, nodes, cpus = line.split()
                self.previous_jobs.append(job_id)
                self.total_nodes += int(nodes)
                self.total_cpus += int(cpus)
        
        print(f"Previously running jobs for user {self.user}: {self.previous_jobs}")
        print(f"Total Nodes: {self.total_nodes}, Total CPUs: {self.total_cpus}")
