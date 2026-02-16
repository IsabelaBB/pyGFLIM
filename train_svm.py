import argparse
from setup import *

def parse_args():
    parse = argparse.ArgumentParser("Train SVM classifier on GFLIM-BoF activations")
    # mandatory
    parse.add_argument("train_path", type=existing_file_or_dir, help="Path to the original images (folder or csv file)")
    parse.add_argument("activations_dir", type=existing_dir, help="Activations directory with graph files")
    parse.add_argument("model_dir", type=existing_dir, help="model directory")
    parse.add_argument("svm_file", type=str, help="output filename for SVM model")
    # optional
    parse.add_argument("--sorted_vector", type=boolean_value, help="Whether to sort the features (1) or not (0) in graph embedding", 
                       default='1')
    parse.add_argument("--C", type=float, help="SVM C parameter", default=1e2)
    parse.add_argument("--kernel", type=str, help="SVM kernel", default='Linear', choices=['Linear', 'RBF'])
    parse.add_argument("--multiclass_type", type=str, help="SVM multiclass type", default='OVO', choices=['OVO', 'OVA'])
    parse.add_argument("--kernelization", type=boolean_value, help="Whether to apply kernelization (1) or not (0) to the features before training the SVM", default=0, choices=[0,1])
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    log_file = os.path.join(args.model_dir, 'log.txt')
    args.kernel = 0 if args.kernel == 'Linear' else 1
    args.multiclass_type = 0 if args.multiclass_type == 'OVO' else 1
    args.svm_file = os.path.join(args.model_dir, args.svm_file)

    # transform json grapha into a ift dataset (.zip file)
    run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
               [args.activations_dir, args.train_path, 
                1 if args.sorted_vector else 0], 
                log_file)

    # train a SVM classifier
    run_script(f"{EXEC_DIR}/iftSupTrainBySVM",
              [f"{args.activations_dir}.zip", args.kernel, 
               args.multiclass_type, args.C, 
               1 if args.kernelization else 0, args.svm_file],
               log_file)
