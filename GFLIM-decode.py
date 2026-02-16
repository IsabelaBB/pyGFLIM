import os
import argparse
from GFLIMFunctions import GFLIM_BoFP
from setup import *

def parse_args():
    parser = argparse.ArgumentParser("Write the activations for a model trained with GFLIM")
    # mandatory 
    parser.add_argument("model_dir", type=existing_dir, help="Directory where the model is stored.")
    # optional
    parser.add_argument("--decoder", help="Decoder to use for decoding.", type=str, default='graph-adaptive', 
                    choices=['graph-adaptive', 'hybrid', 'tristate', 'mean'])
    parser.add_argument("--batch_size", help="Batch size for training.", type=positive_int, default=64)
    parser.add_argument("--layer", help="Layer to write activations for.", type=positive_int, default=1)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    tmp_file = os.path.join(args.model_dir, 'log.txt')
    arch_file = os.path.join(args.model_dir, 'arch.json')
    
    model = GFLIM_BoFP(args.model_dir, arch_file,
                       tmp_file, args.batch_size)
    model.decode_layer(args.layer, decoder=args.decoder)
    
