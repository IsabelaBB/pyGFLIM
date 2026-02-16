
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import math
import argparse
import sys
import json
from skimage import io
from tqdm import tqdm
import multiprocessing

def normalize_by_band(feature):
    max_ = feature.max()
    min_ = feature.min()
    norm = 255 * ((feature - min_) / (max_ - min_))
    
    return norm

def show_two_fig(image1, image2):
    fig = plt.figure(figsize=(20,20))
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1, cmap="gray")
    plt.axis('off')
    plt.title("First")
    
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2, cmap="gray")
    plt.axis('off')
    plt.title("Second")

def getImageFromGraph(graph, spx_file):
    # Vectorized version for better performance
    h, w = spx_file.shape
    num_feats = graph['num_feats']
    feat_img = np.zeros((h, w, num_feats), dtype=np.float32)
    spx_flat = spx_file.flatten()
    feat_img_flat = feat_img.reshape(-1, num_feats)

    # Precompute all features in an array for fast assignment
    node_feats = np.array([graph[f'node{i}']['feats'] for i in range(graph['num_nodes'])], dtype=np.float32)
    # spx_file values are assumed to be 1-based (i+1), so subtract 1 for zero-based indexing
    valid_mask = (spx_flat > 0) & (spx_flat <= graph['num_nodes'])
    feat_img_flat[valid_mask] = node_feats[spx_flat[valid_mask] - 1]
    # Reshape back to the original image shape
    feat_img = feat_img_flat.reshape(h, w, num_feats)
    return feat_img

def decoder_fb(feature, kernel_importances = None, threshold=0.5, filter_by_size=False, size_range=[2000,15000], foreground_kernels=None, background_kernels=None):
    
    if(kernel_importances is None):
        kernel_importances = np.ones(feature.shape[2])
        
    img_size = [feature.shape[0], feature.shape[1]]

    positives = np.zeros(img_size)
    negatives = np.zeros(img_size)

    counter = 0
    n_counter = 0
    
    for b in range(feature.shape[2]):
        if(feature[:,:,b].max() != 0.0):
            band_image = feature[:,:,b]
            band_mean = (band_image/band_image.max()).mean()
            norm_band_image = normalize_by_band(band_image)

            if(foreground_kernels is None):
                if(band_mean < threshold):
                    positives += (band_image*kernel_importances[b])
                    counter+=kernel_importances[b]
                else:
                    negatives += (band_image*kernel_importances[b])
                    n_counter+=kernel_importances[b]
            else:
                if(b in foreground_kernels):
                    positives += (band_image*kernel_importances[b])
                    counter+=kernel_importances[b]
                elif(background_kernels is None or b in background_kernels):
                    negatives += (band_image*kernel_importances[b])
                    n_counter+=kernel_importances[b]
                

    if(counter != 0):
        positives/=counter
        positives = normalize_by_band(positives)
    if(n_counter != 0):
        negatives/=n_counter
        negatives = normalize_by_band(negatives)
       
    if(not math.isnan(negatives.max())):
        decoded = positives - negatives
    else:
        decoded = positives
        
    
    decoded[decoded < 0] = 0
    #decoded = decoded.astype(np.uint8)
    
    if(filter_by_size):
        nb_components, components_image, stats, centroids = cv2.connectedComponentsWithStats((decoded>0).astype(np.uint8), connectivity=8)
        for c in range(1,nb_components):
            indices = np.argwhere(components_image==c)
            if(indices.shape[0] < size_range[0] or indices.shape[0] > size_range[1]):
                for i in indices:
                    decoded[i[0], i[1]] = 0
                
    if(decoded.max() == 0):
        return decoded.astype(np.uint8)
    decoded = normalize_by_band(decoded)
    decoded = decoded.astype(np.uint8)
    return decoded

def process_file(filename, graph_dir, spx_dir, threshold, foreground_kernels, background_kernels, output_dataset):
    filename_,ext = filename.split(".")
    graph_file = os.path.join(graph_dir, filename_+".json")
    with open(graph_file, 'r') as f:
        graph = json.load(f)
    spx_file = os.path.join(spx_dir, filename_+".pgm")
    spx_img = np.array(io.imread(spx_file, as_gray=True))
    feature_img = getImageFromGraph(graph, spx_img)
    decoded = decoder_fb(feature_img, threshold=threshold, foreground_kernels=foreground_kernels, background_kernels=background_kernels)
    cv2.imwrite(output_dataset+"/"+filename_+".png", decoded)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    
    ap.add_argument("-i", "--graph_dir", required=True,	help="path to the folder with the feature images")
    ap.add_argument("-s", "--spx_dir", required=True,	help="path to the folder with the superpixel images")
    ap.add_argument("-o", "--output_dataset", required=True,	help="path to the folder for the decoded images")
    ap.add_argument("-t", "--threshold", required=False, default=0.5, help="Threshold to determine based on the mean saliency if the image is to be considered foreground")
    ap.add_argument("-fg", "--foreground", required=False, default=None, help="A list of foreground kernel numbers")
    ap.add_argument("-bg", "--background", required=False, default=None, help="A list of background kernel numbers")
    
    args = vars(ap.parse_args())
    
    graph_dir = args["graph_dir"]
    threshold = float(args["threshold"])
    output_dataset = args["output_dataset"]
    spx_dir = args["spx_dir"]

    foreground_kernels = [int(x) for x in args["foreground"].split(",")]
    background_kernels = [int(x) for x in args["background"].split(",")]
    
    if not os.path.exists(output_dataset):
        os.makedirs(output_dataset)
    
    dataset_files = os.listdir(graph_dir)

    with multiprocessing.Pool() as pool:
        pool.starmap(process_file, ((filename, graph_dir, spx_dir, threshold, foreground_kernels, background_kernels, output_dataset) for filename in tqdm(dataset_files)))

    sys.stdout.write("]\n") # this ends the progress bar
    print("Done!")