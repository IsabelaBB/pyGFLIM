"""
Select images from a CSV file or directory, choosing a specified number per class.
"""

import os
import random
import argparse
from setup import *

def parse_args():
    parser = argparse.ArgumentParser("Choose images per class")
    # mandatory
    parser.add_argument("images_path", type=existing_file_or_dir, help="Path to the original images (folder or csv file)")
    parser.add_argument("output_path", type=str, help="Path to the output CSV file where selected image paths will be saved")
    # optional
    parser.add_argument("--images_per_class", type=int, help="Number of images to select per class", default=1)
    parser.add_argument("--random_state", type=int, help="Random seed for reproducibility", default=42)
    return parser.parse_args()

def read_image_names(image_path):
    # Read image names from the CSV file and sort them
    if os.path.isfile(image_path):
        with open(image_path, "r") as f:
            image_names = [line.strip() for line in f]
    else:
        image_names = []
        for root, _, files in os.walk(image_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_names.append(os.path.join(root, file))
    return sorted(image_names)

def group_images_by_label(image_names):
    # Group images by their label (assumed to be the first part of the filename)
    labels_to_images = {}
    for file_path in image_names:
        name = file_path.split("/")[-1] # Get the filename only
        label = int(name.split("_")[0]) # Extract label from filename
        labels_to_images.setdefault(label, []).append(file_path) # Group images by label
    return labels_to_images

def select_images(labels_to_images, num_images_per_class, random_state):
    # Select a specified number of images per class using the given random state
    selected_images = []
    for class_idx in labels_to_images:
        random.seed(random_state)
        selected = random.sample(labels_to_images[class_idx], num_images_per_class)
        selected_images.extend(selected)
    return selected_images

def copy_selected_images(selected_images, output_path):
    # Copy selected images to the output directory for the split
    with open(output_path, "w") as f:
        for image in selected_images:
            f.write(f"{image}\n")

if __name__ == "__main__":
    args = parse_args()
    if args.output_path.endswith('.csv') == False:
        raise ValueError("Output path must be a CSV file.")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    image_names = read_image_names(args.images_path)
    labels_to_images = group_images_by_label(image_names)
    selected_images = select_images(labels_to_images, 
                                    args.images_per_class, 
                                    args.random_state)
    copy_selected_images(selected_images, args.output_path)
