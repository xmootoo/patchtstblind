# std python module
import os
import yaml
import argparse
from rich.console import Console

def run_script(cli_input, script_path):
    """
        Run a script locally (e.g., training).
    Args:
        cli_input: A string of input arguments to pass to the training script.
        script_path (str): Path to the script to run.
    Returns:
        None
    """

    # Rich console (terminal output formatting)
    console = Console()

    # Form the command with the full path
    command = f"python {script_path} {cli_input}"
    console.log(f"Run command {command}")
    exit_status = os.system(command)

    if exit_status == 0:
        print("Command executed successfully")
    else:
        print(f"Command failed with exit status {exit_status}")


def load_configs(exp_name):
    """
        Load the configuration files for the experiment.

    Args:
        exp_name (str): The folder name for the experiment  within 'exp' directory. This can include single folders, such as
                         'patchtst_electricity_96' or subfolders for nested experiments, such as 'patchtst/electricity_96'
    Returns:
        cli_inputs (str): The command line arguments in argparse format (e.g., "--epochs 15 --batch_size 32")
        script_path (str): The path to the script to run.
    """

    # CLI Arguments
    with open(os.path.join("../exp", exp_name, "args.yaml")) as f:
        cli_args = yaml.safe_load(f)

    # Convert the data to a string
    def format_arg(key, value):
        if isinstance(value, str) and "," in value:
            # Ensure string values, especially those containing commas, are quoted properly
            return f'--{key} "{value}"'
        elif isinstance(value, str):
            return f"--{key} {value}"
        else:
            return f"--{key} {value}"

    cli_input = " ".join(format_arg(key, value) for key, value in cli_args.items())

    # Script path
    with open(os.path.join("../exp", exp_name, "compute_canada.yaml")) as f:
        script_path = yaml.safe_load(f)["script_path"]

    return cli_input, script_path


if __name__ == "__main__":

    # Job submission arguments
    parser = argparse.ArgumentParser(description="HPC jobs submission")

    parser.add_argument("--exp_name", type=str, default="patchtst/electricity_512_96", help="Experiment name")

    args = parser.parse_args()

    cli_input, script_path = load_configs(args.exp_name)

    run_script(cli_input, script_path)
