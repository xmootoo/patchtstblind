# std python module
import os
import subprocess
import yaml
import argparse

# Distributed computing module
import submitit

def run_script(virtual_env, modules, cli_input, script_path, ddp=False):
    """
    Run the script in a SLURM job (e.g., training) on a cluster. This function
    will run on the cluster and execute the script.

    Args:
        cli_input: A string of input arguments to pass to the training script.
        virtual_env (str): The name of the virtual environment to activate.
        modules (str): The list of modules to load.
        script_path (str): Path to the script to run.
    Returns:
        None
    """

    # Rich console (terminal output formatting)
    from rich.console import Console
    console = Console()

    py_env_activate = os.path.join(os.path.expanduser("~"),virtual_env, "bin", "activate")
    if not os.path.isfile(py_env_activate):
        console.log(f"FileNotFound: the file {py_env_activate} does not exist")
        return

    # Load modules and environment
    load_modules = f"module load {modules}"
    activate_env = f"source {py_env_activate}"

    # Form the command with the full path
    if ddp:
        command = f"{load_modules} && {activate_env} && python -u -m torch.distributed.launch --use_env {script_path} {cli_input}"
    else:
        command = f"{load_modules} && {activate_env} && python {script_path} {cli_input}"
    console.log(f"Run command {command}")

	# Run the command on the Operating System using subprocess.run
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check if the command failed
    if result.returncode != 0:
        console.log(f"Command failed with return code {result.returncode}")
        console.log(f"Error output: {result.stderr}")
    else:
        console.log(f"Command succeeded with return code {result.returncode}")
        console.log(f"Command output: {result.stdout}")


def submit_job(logdir, virtual_env, modules, cli_input, script_path, /, **kwds):
    """
    Submits a job to the cluster using SLURM, from your local computer.

    Args:
        logdir (str): The directory to store the logs and output of the job.
        cli_input (str): The input argument to pass to the training script.
        virtual_env (str): The name of the virtual environment to activate.
        modules (str): The list of modules to load.
        script_path (str): Path to the script to run.
    Returns:
        None
    """

    # Rich console (terminal output formatting)
    from rich.console import Console
    console = Console()

    # Create an executor for SLURM
    executor = submitit.AutoExecutor(folder=logdir)

    # Set SLURM job parameters
    executor.update_parameters(
        gpus_per_node=kwds.get("gpus_per_node", 1),
        tasks_per_node=kwds.get("tasks_per_node", 1),
        cpus_per_task=kwds.get("cpus_per_task", 1),
        slurm_mem_per_cpu=kwds.get("slurm_mem_per_cpu", 12) * 1024, # Memory per CPU (arg is GB)
        slurm_time=kwds.get("slurm_time","00:05:00"),
        slurm_array_parallelism=kwds.get("slurm_array_parallelism", 1),
        slurm_account=kwds.get("slurm_account", "def-milad777"), # Replace with your account
        slurm_mail_user=kwds.get("slurm_mail_user", "xmootoo@gmail.com"),  # Email for notifications
        slurm_mail_type=kwds.get("slurm_mail_type", "ALL"),
        )

    # Submit the job
    job = executor.submit(run_script, virtual_env, modules, cli_input, script_path)
    console.log(f"Job submitted: {job.job_id}")


def load_configs(exp_name):
    """
    Load the configuration files for the experiment.

    Args:
        exp_name (str): The folder name for the experiment  within 'exp' directory. This can include single folders, such as
                         'patchtst_electricity_96' or subfolders for nested experiments, such as 'patchtst/electricity_96'
    Returns:
        cc_config (dict): The Compute Canada configuration.
        slurm_config (dict): The SLURM configuration.
        cli_inputs (str): The command line arguments in argparse format (e.g., "--epochs 15 --batch_size 32")
    """

    # Compute Canada Configuration
    with open(os.path.join("exp", exp_name, "compute_canada.yaml")) as f:
        cc_config = yaml.safe_load(f)

    # SLURM Configuration
    with open(os.path.join("exp", exp_name, "slurm.yaml")) as f:
        slurm_config = yaml.safe_load(f)

    # CLI Arguments
    with open(os.path.join("exp", exp_name, "args.yaml")) as f:
        cli_args = yaml.safe_load(f)

    # Convert the data to a string
    cli_input = ' '.join(f'--{k} {v}' for k, v in cli_args.items())

    return cc_config, slurm_config, cli_input



if __name__ == "__main__":
    
    # Job submission arguments
    parser = argparse.ArgumentParser(description="HPC jobs submission")

    parser.add_argument("--exp_name", type=str, default="patchtst_electricity_96", help="Name of experiment to run")
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, help="Whether to use torch.nn.parallel.DistributedDataParallel")

    args = parser.parse_args()

    cc_config, slurm_config, cli_input = load_configs(args.exp_name)

    submit_job(os.path.join(cc_config["logdir"], args.exp_name),
               cc_config["virtual_env"],
               cc_config["modules"],
               cli_input,
               cc_config["script_path"],
               ddp=args.ddp,
               **slurm_config)


    # # Test 1 (simple script)
    # with open("test1/compute_canada.yaml") as f:
    #     test1_config = yaml.safe_load(f)
    # with open("test1/slurm.yaml") as f:
    #     test1_slurm = yaml.safe_load(f)

    # submit_job(test1_config["logdir"],
    #            test1_config["virtual_env"],
    #            test1_config["modules"],
    #            "-n 5", # CLI argument
    #            test1_config["script_path"],
    #            **test1_slurm)

    # # Test 2 (torch training script)
    # with open("test2/compute_canada.yaml") as f:
    #     test2_config = yaml.safe_load(f)
    # with open("test2/slurm.yaml") as f:
    #     test2_slurm = yaml.safe_load(f)

    # submit_job(test2_config["logdir"],
    #            test2_config["virtual_env"],
    #            test2_config["modules"],
    #            "--epochs 15", # CLI argument
    #            test2_config["script_path"],
    #            **test2_slurm)
