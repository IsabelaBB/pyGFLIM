import os
import argparse
from setup import *
import json

def parse_args():
    parser = argparse.ArgumentParser("Create a CSV file from a JSON file containing image sets")
    parser.add_argument("imgSets_file", type=existing_file_or_dir, help="Path to the image sets JSON file.")
    parser.add_argument("images_folder", type=existing_dir, help="original images directory.")
    parser.add_argument("set_name", type=str, help="Image set to extract from the json file. Ex: training or validation or test.")
    parser.add_argument("out_csv_file", type=str, help="Path to the output csv file.")
    return parser.parse_args()


args = parse_args()
os.makedirs(os.path.dirname(args.out_csv_file), exist_ok=True)

with open(args.imgSets_file, 'r') as f:
    data = json.load(f)

if args.set_name not in data:
    raise ValueError(f"Set name {args.set_name} not found in the JSON file. Available sets: {list(data.keys())}")

with open(args.out_csv_file, 'w') as f:
    for img_path in data[args.set_name]:
        f.write(f"{args.images_folder}/{img_path}\n")



