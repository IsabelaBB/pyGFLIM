import os
import shutil
import argparse
import sys
import subprocess

def existing_file_or_dir(path):
    if not os.path.isfile(path) and not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file or directory.")
    return path

def existing_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return path

def existing_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory.")
    return path

def boolean_value(value):
    if value.lower() in ['true', '1']:
        return True
    elif value.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(f"{value} is not a valid boolean value. Use 'true' or 'false'.")

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid positive integer.")
    return ivalue

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_str(arg):
    return list(map(str, arg.split(',')))

IFT_dir="/home/isabela/Codes/LibIFT"
EXEC_DIR = os.path.join(IFT_dir, 'bin')
RESULTS_DIR = os.path.join(IFT_dir, 'Results')
DATASETS_DIR = os.path.join(IFT_dir, 'Datasets')
ARCH_DIR = os.path.join(DATASETS_DIR, 'archs')

def read_dir_or_CSV(path, read_extension : bool = True):
    files = []
    if(os.path.isdir(path)):
        if(read_extension):
            files = [os.path.basename(file) for file in os.listdir(path)]
        else:
            files = [os.path.basename(file).split('.')[0] for file in os.listdir(path)]
    else:
        with open(path, 'r') as f:
            if(read_extension):
                files = [os.path.basename(file) for file in f.read().splitlines()]
            else:
                files = [os.path.basename(file).split('.')[0] for file in f.read().splitlines()]

    return files

def CheckFilesRelation_dirOrCSV(path1, path2, same_extension : bool = True):
    """
    Check if path1 and path2 contain the same files.
    Args:
        path1 (str): Path to the first directory or file containing the list of files.
        path2 (str): Path to the second directory or file containing the list of files.
    """

    if(not os.path.exists(path1)):
        raise ValueError("check_files_dirOrCSV: path1 does not exist")

    if(not os.path.exists(path2)):
        raise ValueError("check_files_dirOrCSV: path2 does not exist")

    files1 = read_dir_or_CSV(path1, read_extension = same_extension)
    files2 = read_dir_or_CSV(path2, read_extension = same_extension)

    if set(files1) > set(files2):
        return "gt"
    elif set(files1) < set(files2):
        return "lt"
    elif set(files1) == set(files2):
        return "eq"
    else:
        return "diff"

def copyGraphsAndLabels(src_dir, input_graphs_path, input_labels_path, model_dir):
    """
    Copy the graphs and labels from the source directory to the model directory.
    Args:
        src_dir (str): Source directory containing the filenames to be copied.
        input_graphs_path (str): Path to the input graphs directory.
        input_labels_path (str): Path to the input labels directory.
        model_dir (str): Destination model directory where the graphs and labels will be copied.
    """
    if(CheckFilesRelation_dirOrCSV(input_graphs_path, input_labels_path, same_extension = False) != 'eq'):
        raise ValueError("The files in input_graphs_path and input_labels_path do not match.")

    rel_src = CheckFilesRelation_dirOrCSV(src_dir, input_labels_path, same_extension = False)

    files1 = read_dir_or_CSV(src_dir, read_extension = False)
    files2 = read_dir_or_CSV(input_labels_path, read_extension = False)
    for f in files1:
        if f not in files2:
            raise ValueError(f"The file {f} in {src_dir} does not exist in {input_labels_path}.")

    if(rel_src == 'diff' or rel_src == 'gt'):
        raise ValueError(f"The files in {src_dir} and {input_labels_path} do not match: {rel_src}.")

    # remove any existing layer and label directories in the model directory
    shutil.rmtree(os.path.join(model_dir, 'layer0'), ignore_errors=True)
    shutil.rmtree(os.path.join(model_dir, 'labels0'), ignore_errors=True)

    # create the output directories
    os.makedirs(os.path.join(model_dir, 'layer0'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'labels0'), exist_ok=True)

    if(os.path.isdir(src_dir)):
        # read the files from the directory
        filenames = os.listdir(src_dir)
    else:
        # read the files paths from a csv file
        with open(src_dir, 'r') as f:
            # Remove any leading/trailing whitespace characters and the only the filenames
            filenames = [f.strip().split('/')[-1] for f in f.read().splitlines()]

    for file_name in filenames:

        # get the file name
        if file_name.endswith('-seeds.txt'):
            graph_name = file_name.split('-')[0]
        else:
            graph_name = file_name.split('.')[0]

        # create symlinks for the graph and label files
        graph_src_file = os.path.join(input_graphs_path, f'{graph_name}.json')
        os.symlink(graph_src_file, os.path.join(model_dir, 'layer0', f'{graph_name}.json'))

        labels_src_file = os.path.join(input_labels_path, f'{graph_name}.pgm')
        os.symlink(labels_src_file, os.path.join(model_dir, 'labels0', f'{graph_name}.pgm'))

def run_script(command_name, args, tmp_file=None):
    """
    Call a command with the given arguments.
    Args:
        args (list): The arguments to pass to the command.
        msg (str, optional): A message to display while calling the function.
    """
    cmd = "{} {}".format(command_name, " ".join([str(i) for i in args]))
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if tmp_file is not None:
        with open(tmp_file, "a+") as f:
            f.write(process.stdout)
    if process.returncode != 0:
        print(f"Error running command: {cmd}")
        print(process.stderr)
        sys.exit(process.returncode)
    return process.stdout
