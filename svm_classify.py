import os
import argparse
from setup import *

def parse_args():
    parse = argparse.ArgumentParser("Classify GFLIM-BoF activations using an SVM")
    # mandatory
    parse.add_argument("images_path", type=existing_file_or_dir, help="Path to the original images (folder or csv file)")
    parse.add_argument("model_dir", type=existing_dir, help="Activations directory with graph files")
    parse.add_argument("layer", type=int, help="Activations directory with graph files")
    parse.add_argument("svm_file", type=existing_file, help="SVM file name")
    # optional
    parse.add_argument("--sorted_vector", type=boolean_value, help="Whether to sort the features (1) or not (0) in graph embedding", 
                       default='1')
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    log_file = os.path.join(args.model_dir, 'log.txt')
    activations_dir = os.path.join(args.model_dir, f'layer{args.layer}')
    args.svm_file = os.path.join(args.model_dir, args.svm_file)
    results_dir = os.path.join(args.model_dir, f'results_{os.path.basename(args.images_path).split(".")[0]}_layer{args.layer}')
    os.makedirs(results_dir, exist_ok=True)
    
    # transform json grapha into a ift dataset (.zip file)
    run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
               [activations_dir, args.images_path, 
                1 if args.sorted_vector else 0], 
                log_file)

    # train a SVM classifier
    classification_result = run_script(f"{EXEC_DIR}/iftClassifyBySVM", 
              [f"{activations_dir}.zip", args.svm_file,
               results_dir], log_file)

    # save classification results
    with open(os.path.join(results_dir, f'classification.txt'), 'w') as f:
        f.write(classification_result)