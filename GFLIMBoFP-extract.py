import os
import argparse
from GFLIMFunctions import GFLIM_BoFP, Dataset
from setup import *

def parse_args():
    parser = argparse.ArgumentParser("Extract features using a model trained with GFLIM-BoF")

    # mandatory 
    parser.add_argument("images_folder", type=existing_file_or_dir, help="Path to the images folder / csv file.")
    parser.add_argument("model_dir", type=existing_dir, help="Directory where the model is stored.")
    
    # optional
    parser.add_argument("--batch_size", help="Batch size for training.", type=positive_int, default=64)
    parser.add_argument("--decoders", help="Decoders to use.", type=list_of_str, default=['graph-adaptive'])
    parser.add_argument("--use_mask", help="Whether to use masks during training.", type=boolean_value, default=True)
    parser.add_argument("--layers_outputs", help="Layers to save output graphs and superpixel maps (ex: 1,2,3)", type=list_of_ints, default=[])
    parser.add_argument("--layers_activations", help="Layers to save image activations (ex: 1,2,3)", type=list_of_ints, default=[])
    parser.add_argument("--layers_decode", help="Layers to decode (ex: 1,2,3).", type=list_of_ints, default=[])
    parser.add_argument("--nlayers", help="Number of layers", type=positive_int, default=1)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    tmp_file = os.path.join(args.model_dir, 'log.txt')
    
    dataset = Dataset(args.images_folder, 
                      use_masks=args.use_mask,
                      tmp_file=tmp_file)
    
    model = GFLIM_BoFP(args.model_dir, 
                        os.path.join(args.model_dir, 'arch.json'), 
                        tmp_file, batch_size=args.batch_size)
    model.extract(dataset, args.nlayers, args.layers_outputs, 
                  args.layers_activations, args.layers_decode, 
                  args.decoders)