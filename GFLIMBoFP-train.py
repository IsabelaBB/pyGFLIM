import os
import argparse
from GFLIMFunctions import GFLIM_BoFP, Dataset, write_gflim_config_file
from setup import *

def parse_args():
    parser = argparse.ArgumentParser("Train a model using GFLIM-BoF")

    # mandatory 
    parser.add_argument("images_folder", type=existing_file_or_dir, help="Path to the training images folder / csv file.")
    parser.add_argument("model_dir", type=existing_dir, help="Directory where the model is stored.")
    parser.add_argument("spx", type=positive_int, help="Final number of superpixels.") 
    parser.add_argument("arch_file", type=existing_file, help="Name of the architecture file.")

    # optional
    parser.add_argument("--spx_alg", type=str, choices=['DISF', 'ERGC'],help="Superpixel algorithm.", default='DISF')
    parser.add_argument("--graph_feat", type=str, help="Type of graph features to use.", default='MEAN_CONCAT_STD_SIZE', choices=['SUM', 'MEAN', 'MEAN_CONCAT_STD_SIZE'])
    parser.add_argument("--stride", type=str, help="Stride type.", default='graph-based', choices=['graph-based', 'none'])
    parser.add_argument("--attention", type=str, help="Attention type.", default='CLASS_DOMINANCE', choices=['NONE', 'CLASS_DOMINANCE'])
    parser.add_argument("--aggregation", type=str, help="Aggregation strategy.", default='SORTFEAT', choices=["CONCAT","MEAN","WMEAN","WCONCAT","SORTFEAT","SORTFEAT_V2"])
    parser.add_argument("--markers_dir", type=existing_dir, help="Directory where the image markers are stored.", default=None)
    parser.add_argument("--init_seeds", type=positive_int, help="Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).", default=300)
    parser.add_argument("--batch_size", help="Batch size for training.", type=positive_int, default=64)
    parser.add_argument("--use_mask", help="Whether to use masks during training.", type=boolean_value, default=True)
    parser.add_argument("--nlayers", help="Number of layers", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    tmp_file = os.path.join(args.model_dir, 'log.txt')
    
    dataset = Dataset(args.images_folder, 
                      use_masks=args.use_mask,
                      tmp_file=tmp_file)
    dataset.create_graphs_and_labels(args.spx_alg, args.spx, args.graph_feat, 
                                    graphs_path_src=args.model_dir, 
                                    initial_seeds=args.init_seeds)

    write_gflim_config_file(args.model_dir, args.spx_alg, 
                            args.spx, args.init_seeds, 
                            args.graph_feat, args.stride, 
                            args.attention, args.aggregation)

    model = GFLIM_BoFP(args.model_dir, args.arch_file,
                       tmp_file, args.batch_size)    
    model.create_seeds(dataset, args.markers_dir)
    for l in range(1,args.nlayers+1):
        model.train_layer(l, dataset)
    
