"""
This file contains classes for GFLIM, GFLIM BoF, MGFLIM BoFP (multiscale), and RGFLIM BoFP (recurrent)
Please be aware that some functionalities may not be implemented in all GFLIM versions.

GFLIM : First version, published in CIARP 2023. 
To create a kernel bank for a layer l, performs a kmeans per marker, 
a kmeans per training image, and a final kmeans. On each kmeans, the centroids are selected as candidate kernels.
It does not have bias, so a z-normalization is performed during training and inference.
It performs pooling, but with stride 1.
The activations can be decoded with the User-guided decoder.

GFLIM Bag of Features (BoFP): It selects seeds per training image with a kmeans per marker. 
On each kmeans, the seeds closest to the centroids are selected as kernels.
Performs a z-normalization and unit normalization to compute the kernels.
Compute a bias vector per layer.
Performs pooling with stride >= 1. The stride can be graph-based or image-based.
The activations can be decoded with the User-guided decoder, Hybrid decoder, or Graph-based decoder.

[in construction]
"""

import os
import json
import filecmp
from setup import *
from abc import ABC, abstractmethod

# To run SH: source ~/.local/share/virtualenvs/SH-NTEVX1e9/bin/activate

def compute_kernels_maps(model_dir, rows_to_keep = None):
    def get_kernel_linei(k, l, kernels_maps):
        # find which kernel is k in line (or scale) l
        i = 0
        while(i < l):
            k = kernels_maps[i][k]
            i+=1
        return k
    
    # contruct a kernels map from a scale 0 to i+1
    lines = []
    with open(os.path.join(model_dir, 'model', 'kernels_map.txt'), 'r') as f:
        lines = f.readlines()

    if rows_to_keep is None:
        rows_to_keep = list(range(len(lines)))

    # process line 0
    ki = 0
    kernels_map = []
    for i,k0 in enumerate(lines[0].strip().split(' ')):
        k0 = int(k0) - 1
        if(k0 == i):
            kernels_map.append(ki) # new kernel
            ki += 1
        else:
            kernels_map.append(kernels_map[k0]) # repeated kernel

    kernels_maps = [kernels_map]

    # process each line i
    line_prev = lines[0].strip().split(' ')
    for l, line in enumerate(lines[1:]):
        line = line.strip().split(' ')
        kernels_map = []
        ki = 0
        for i,k0 in enumerate(line):
            k0 = int(k0) - 1 # i-th kernel in scale i indexed as scale 0
            # (e.g., i=5 k0=3 means kernels 3 and 5 in scale 0 are the same in scale i)

            k0_prev = int(line_prev[i]) - 1 # i-th kernel in scale i-1 indexed as scale 0
            # (e.g., i=5 k0_prev=4 means kernels 4 and 5 in scale 0 are the same in scale i-1. 
            # Considering the previous example, kernels 3, 4, and 5 in scale 0 are the same in scale i)

            if(k0_prev == i):
                if(k0 == k0_prev):
                    kernels_map.append(ki)
                    ki+=1
                else: # repeated (merged) kernel. Map from scale i-1 to scale i
                    k = get_kernel_linei(k0, l+1, kernels_maps) # map k0 from scale 0 to scale i-1
                    k = kernels_map[k]
                    kernels_map.append(k)

        line_prev = line.copy()
        kernels_maps.append(kernels_map)

    """
    # erase scales
    kernels_map_final = [kernels_maps[rows_to_keep[0]].copy()]
    for id,r in enumerate(rows_to_keep[:-1]):
        s = rows_to_keep[id+1]
        r+=1
        mapped_kernels = kernels_maps[r]
        r+=1
        while(r <= s):
            mapped_kernels = [kernels_maps[r][k] for k in mapped_kernels]
            r+=1
        kernels_map_final.append(mapped_kernels)
    """
    return kernels_maps    

def write_gflim_config_file(model_dir, spx_alg, 
                            spx, initial_seeds=0, graph_feat='MEAN', 
                            stride_type='none', attention='NONE', 
                            aggregation='CONCAT', kernel_neighbors='act', 
                            pool_neighbors='act', stride_neighbors='act',
                            merge_function='none'):
    config = {
        'spx_alg': spx_alg,
        'spx': spx,
        'initial_seeds': initial_seeds,
        'graph_feat': graph_feat,
        'stride_type': stride_type,
        'attention': attention,
        'aggregation': aggregation,
        'kernel_neighbors': kernel_neighbors,
        'pool_neighbors': pool_neighbors,
        'stride_neighbors': stride_neighbors,
        'merge_function': merge_function,
        'reduce_kernels': 0,
        'fixed_spx': 0
    }
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def read_gflim_config_file(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'config.json')):
        raise ValueError(f"read_gflim_config_file: Config file not found in {model_dir}. Please make sure to provide a valid model directory.")
    
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    return config

class GFLIMFunction():
    def __init__(self, binary_name, tmp_file=None):
        self.binary_name = binary_name
        self.tmp_file = tmp_file
    
    def __call__(self, args = []):
        """
        Call a command with the given arguments.
        Args:
            args (list): The arguments to pass to the command.
            msg (str, optional): A message to display while calling the function.
        """
        return run_script(f'{EXEC_DIR}/{self.binary_name}', args, self.tmp_file)

class Dataset():
    def __init__(self, images_path, graphs_path=None, 
                 use_masks=False, tmp_file=None):
        
        self.images_path = images_path
        self.graphs_path = graphs_path
        self.use_masks = use_masks
        self.tmp_file = tmp_file
        self.functions = {
            'CreateGraph': GFLIMFunction('iftCreateGFLIMGraph', tmp_file),
            'CreateNeighbMatricesAndGraphs': GFLIMFunction('iftCreateGFLIMNeighbMatricesAndGraphs', tmp_file),
            'CreateSpxImageSH': lambda x: run_script('python3 /home/isabela/Codes/SH/src/boruvkasupix.py', x, tmp_file),
            'CreateGraphFromLabels': GFLIMFunction('iftCreateGFLIMGraphFromLabels', tmp_file),
        }

    def create_graphs_and_labels(self, spx_alg, spx, graph_feats, 
                                 graphs_path_src, initial_seeds=0, 
                                 kernel_neighbors='act', 
                                 pool_neighbors='act', 
                                 stride_neighbors='act',
                                 arch_file=None):
        """
        Create the graphs for the dataset if they do not exist.
        Args:
            images_path (str): images directory
            spx_alg (str): Superpixel algorithm to use. Options: DISF, ERGC, SH. SH is only available when using MGFLIM BoFP
            spx (int or list): Number of superpixels (> 0). It can be either a number (spx_alg is DISF or ERGC) or a list (spx_alg is SH).
            graph_feats (str): Type of graph features to use. See iftGFLIMGraphNodeFeature at iftGFLIMGraph.h for more details.
            graphs_path_src (str): Path where the graphs and labels will be stored.
            initial_seeds (int): Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).
            kernel_neighbors (str): Kernel neighbors type (act or color).
            pool_neighbors (str): Pool neighbors type (act or color).
            stride_neighbors (str): Stride neighbors type (act or color).
            arch_file (str): Architecture file (only used when 'color' neighbor types are used).
        """
        if kernel_neighbors not in ['act', 'color']:
            raise ValueError(f"{self.__class__.__name__}.create_graphs: kernel_neighbors must be 'act' or 'color': {kernel_neighbors}")
        if pool_neighbors not in ['act', 'color']:
            raise ValueError(f"{self.__class__.__name__}.create_graphs: pool_neighbors must be 'act' or 'color': {pool_neighbors}")
        if stride_neighbors not in ['act', 'color']:
            raise ValueError(f"{self.__class__.__name__}.create_graphs: stride_neighbors must be 'act' or 'color': {stride_neighbors}")

        if spx_alg in ['DISF', 'ERGC'] and isinstance(spx, list):
            if len(spx) != 1:
                raise ValueError("create_graphs: spx must have only one value when spx_alg is DISF")
            else:
                spx = spx[0]

        if spx_alg == 'DISF':
            if initial_seeds <= 0 or initial_seeds < spx:
                raise ValueError(f"{self.__class__.__name__}.create_graphs: initial_seeds must be greater than 0 and greater than or equal to final_spx when spx_alg is DISF: {initial_seeds}")
        else:
            if self.use_masks:
                raise ValueError(f"{self.__class__.__name__}.create_graphs: use_masks can only be used when spx_alg is DISF")

        if pool_neighbors == 'act' and stride_neighbors == 'act' and pool_neighbors == 'act':
            if spx_alg in ['DISF', 'ERGC']:
                self.functions['CreateGraph']([self.images_path, spx, graphs_path_src, graph_feats, spx_alg, initial_seeds, 1 if self.use_masks else 0])
            else:
                self.functions['CreateSpxImageSH']([f'--img {self.images_path}', f'--save {graphs_path_src}', f'--node_feat {graph_feats}', f'--k {",".join([str(s) for s in spx])}'])
                for spx in enumerate(spx):
                    self.functions['CreateGraphFromLabels']([f'{self.images_path}', f'{graphs_path_src}/{spx[1]}/labels0', graph_feats, f'{graphs_path_src}/{spx[1]}/layer0', 1 if self.use_masks else 0])
        else:
            if spx_alg != 'DISF':
                raise ValueError(f"{self.__class__.__name__}.create_graphs: spx_alg must be DISF when using different neighbor types for kernel, pool, and stride: {spx_alg}")
            if arch_file is None:
                raise ValueError(f"{self.__class__.__name__}.create_graphs: arch_file must be provided when using different neighbor types for kernel, pool, and stride")
            self.functions['NeighbMatricesAndGraphs']([self.images_path, spx, initial_seeds, graphs_path_src, graph_feats, arch_file, 1 if self.use_masks else 0, '0', kernel_neighbors, pool_neighbors, stride_neighbors])

        self.graphs_path = graphs_path_src

class GFLIM_ABC(ABC):
    """
    Abstract class for GFLIM, GFLIM BoFP, MGFLIM BoFP, and RGFLIM BoFP.
    It contains the common functions for all GFLIM versions.
    """
    def __init__(self, model_dir, arch_file, 
                tmp_file, batch_size=64):
        
        self.tmp_file = tmp_file
        self.model_dir = model_dir
        self.arch_file = arch_file
        self.batch_size = batch_size

        self.nlayers = 1
        if os.path.exists(arch_file) == False:
            raise ValueError(f"{self.__class__.__name__}: Architecture file {arch_file} not found. Please make sure to provide a valid architecture file.")
        with open(arch_file) as f:
            arch = json.load(f)
            self.nlayers = arch['nlayers']

        config = read_gflim_config_file(model_dir)
        self.spx_alg = config['spx_alg']
        self.spx = config['spx']
        self.initial_seeds = config['initial_seeds']
        self.stride_type = config['stride_type']
        self.kernel_neighbors = config['kernel_neighbors']
        self.pool_neighbors = config['pool_neighbors']
        self.stride_neighbors = config['stride_neighbors']
        self.graph_feat = config['graph_feat']
        self.aggregation = config['aggregation']
        self.attention = config['attention']
        self.reduce_kernels = config['reduce_kernels']
        self.fixed_spx = config['fixed_spx']
        
        self.functions = {
            'WriteActivations': GFLIMFunction('iftGFLIMWriteActivations', tmp_file),
            'DecodeLayer': GFLIMFunction('iftGFLIMDecodeLayer', tmp_file),
            'ConvertSeeds': GFLIMFunction('iftGFLIMConvertSeeds', tmp_file),
            'CreateImgSeedsAllPix': lambda x: run_script(f'python3 {IFT_dir}/demo/GFLIM/CreateSeedsFromAllPix.py', x, tmp_file),
            'PrecConvertSeeds': GFLIMFunction('iftGFLIMPrecConvertSeeds', tmp_file)
        }

        if os.path.exists(self.model_dir) == False:
            os.makedirs(self.model_dir)

    def write_activations(self, layer, skip=None):
        if skip is None:
            self.functions['WriteActivations'](
                [self.model_dir, layer, self.batch_size])
        else:
            self.functions['WriteActivations'](
                [self.model_dir, layer, self.batch_size, 
                skip, self.arch_file])
    
    def decode_layer(self, layer, decoder):
        self.functions['DecodeLayer'](
            [self.model_dir, layer, self.arch_file, decoder, self.stride_type, self.batch_size])

    def train_layer(self, layer, dataset : Dataset = None):
        if layer < 1 or layer > self.nlayers:
            raise ValueError(f"{self.__class__.__name__}.train_layer: layer must be between 1 and {self.nlayers}: {layer}")
        if self.spx_alg != 'DISF' and dataset and dataset.use_masks:
            raise ValueError(f"{self.__class__.__name__}.train_layer: use_masks can only be used when spx_alg is DISF")
        if layer == 1:
            shutil.copyfile(self.arch_file, os.path.join(self.model_dir, f'arch.json'))
        else:
            if filecmp.cmp(os.path.join(self.model_dir, f'arch.json'), self.arch_file, shallow=False) == False:
                raise ValueError(f"{self.__class__.__name__}.train_layer: \
                    the architecture file {self.arch_file} is different from {os.path.join(self.model_dir, f'arch.json')}. \
                        Please make sure to use the same architecture file for all layers.")

    def create_seeds(self, dataset : Dataset, img_markers_dir=None):
        """
        Create the seeds for the given architecture.
        Args:
            img_markers_dir (str): Directory where the image markers are stored (optional).

        If img_markers_dir is not provided, the seeds will be created from all pixels in the training images.
        """
        if dataset is None:
            raise ValueError("create_seeds: dataset must be provided to create seeds")

        if img_markers_dir == None:
            img_markers_dir = f"{self.model_dir}/img_seeds"
            os.makedirs(img_markers_dir, exist_ok=True)
            args = [dataset.images_path, img_markers_dir]
            if dataset.use_masks:
                args.append("--use_mask")
                args.append("1")
            self.functions['CreateImgSeedsAllPix'](args)
        
        with open(self.arch_file) as f:
            arch = json.load(f)
            nfeat_markers = arch['layer1']['conv']["nkernels_per_marker"]
            kernel_size = arch['layer1']['conv']['kernel_size']

        if self.kernel_neighbors == 'act':
            self.functions['ConvertSeeds'](
                [self.model_dir, img_markers_dir, 
                 nfeat_markers, kernel_size, 
                 self.aggregation])
        else:
            self.functions['PrecConvertSeeds']([
                self.model_dir, img_markers_dir, 
                nfeat_markers, kernel_size, 
                self.aggregation])

    def extract(self, dataset : Dataset, nlayers, layers_save_graph_act=[], 
                layers_save_img_act=[], layers_decode=[], decoders=[]):
        if nlayers < 1 or nlayers > self.nlayers:
            raise ValueError(f"{self.__class__.__name__}.extract: nlayers must be between 1 and {self.nlayers}: {nlayers}")
        
        if self.spx_alg != 'DISF' and dataset.use_masks:
            raise ValueError(f"{self.__class__.__name__}.extract: use_masks can only be used when spx_alg is DISF")
        
        if len(decoders) == 0:
            if len(layers_decode) > 0:
                raise ValueError(f"{self.__class__.__name__}.extract: decoders list cannot be empty when layers_decode is not empty")
            else:
                decoders = ['graph-based']

        # if layers_save_graph_act contains negative values or values greater than nlayers, raise an error
        if any(l < 1 or l > nlayers for l in layers_save_graph_act):
            raise ValueError(f"{self.__class__.__name__}.extract: layers_save_graph_act must contain values between 1 and nlayers: {layers_save_graph_act}")
        
        # if layers_save_img_act contains negative values or values greater than nlayers, raise an error
        if any(l < 1 or l > nlayers for l in layers_save_img_act):
            raise ValueError(f"{self.__class__.__name__}.extract: layers_save_img_act must contain values between 1 and nlayers: {layers_save_img_act}")
        
        # if layers_decode contains negative values or values greater than nlayers, raise an error
        if any(l < 1 or l > nlayers for l in layers_decode):
            raise ValueError(f"{self.__class__.__name__}.extract: layers_decode must contain values between 1 and nlayers: {layers_decode}")

class GFLIM(GFLIM_ABC):
    def __init__(self, model_dir, arch_file,
                 tmp_file, batch_size=64):
        
        super().__init__(model_dir, arch_file,
                 tmp_file, batch_size=batch_size)
        
        if self.kernel_neighbors != 'act' or self.pool_neighbors != 'act' or self.stride_neighbors != 'act':
            raise ValueError(f"{self.__class__.__name__}: only 'act' neighbor types are implemented.")

        if self.spx_alg == 'SH':
            raise ValueError(f"{self.__class__.__name__}: spx_alg SH is not implemented. Please use MGFLIM BoFP to use SH superpixels.")

        self.functions.update({
            'TrainLayer': GFLIMFunction('iftGFLIMTrainLayer', tmp_file),
            'ExtractFromGraphs': GFLIMFunction('iftGFLIMExtract2', tmp_file),
            'ExtractFromImages': GFLIMFunction('iftGFLIMExtract', tmp_file),
        })

    def create_graph_seeds(self, dataset : Dataset, img_markers_dir=None):
        """
        Create the seeds for the given architecture.
        Args:
            dataset : Dataset object containing the images information.
            img_markers_dir (str): Directory where the image markers are stored.
        """
        if self.kernel_neighbors != 'act':
            raise ValueError(f"{self.__class__.__name__}.create_graph_seeds: only kernel_neighbors 'act' is implemented in GFLIM, not {self.kernel_neighbors}")

        super().create_graph_seeds(dataset, img_markers_dir)

    def train_layer(self, layer):
        """
        Train a GFLIM layer (not Bag of Features).
        Args:
            dataset (Dataset): The dataset to train on.
            layer (int): Layer number.
        """
        super().train_layer(layer)
        self.functions['TrainLayer']([self.model_dir, self.arch_file, layer])

    def extract(self, dataset : Dataset, nlayers, 
            layers_save_graph_act, layers_save_img_act=[], 
            layers_decode=[], decoders=['graph-based']):
        
        super().extract(dataset, nlayers, 
                        layers_save_graph_act=layers_save_graph_act,
                        layers_save_img_act=layers_save_img_act,
                        layers_decode=layers_decode, decoders=decoders)

        if dataset.graphs_path is not None:
            
            if os.path.exists(f"{dataset.graphs_path}/layer0") == False or \
               os.path.exists(f"{dataset.graphs_path}/labels0") == False:
                raise ValueError(f"{self.__class__.__name__}.extract: dataset.graphs_path does not contain layer0 and labels0: {dataset.graphs_path}")

            self.functions['ExtractFromGraphs']([
                                dataset.images_path, f"{dataset.graphs_path}/layer0", 
                                f"{dataset.graphs_path}/labels0", 
                                self.model_dir, self.arch_file, 
                                nlayers, self.batch_size,
                                self.stride_type, self.attention, 
                                self.aggregation, 
                                1 if self.reduce_kernels else 0,
                                ",".join(str(i) for i in layers_save_graph_act) if len(layers_save_graph_act) > 0 else -1,
                                ",".join(str(i) for i in layers_save_img_act) if len(layers_save_img_act) > 0 else -1,
                                ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                                ",".join(str(i) for i in decoders)])
        else:
            self.functions['ExtractFromImages']([dataset.images_path, 
                    self.model_dir, self.arch_file, 
                    nlayers, self.batch_size,
                    self.stride_type, self.attention, 
                    self.aggregation, 
                    1 if self.reduce_kernels else 0,
                    self.graph_feat,
                    self.spx_alg,
                    self.spx,
                    ",".join(str(i) for i in layers_save_img_act) if len(layers_save_img_act) > 0 else -1,
                    ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                    ",".join(str(i) for i in decoders), 
                    1 if dataset.use_masks else 0])

class GFLIM_BoFP(GFLIM_ABC):
    """
    GFLIM BoFP with no skip connections or multiscale.
    """
    def __init__(self, model_dir, arch_file, tmp_file, batch_size=64):

        super().__init__(model_dir, arch_file,
                    tmp_file, batch_size=batch_size)
        
        if self.stride_type not in ['graph-based', 'none'] and \
            (self.kernel_neighbors != 'act' or self.pool_neighbors != 'act' or self.stride_neighbors != 'act'):
            raise ValueError(f"{self.__class__.__name__}: when using neighbor types different than 'act', stride_type must be 'graph-based' or 'none'")
        
        self.functions.update({
            'CreateLayerModel': GFLIMFunction('iftGFLIMCreateLayerModel', tmp_file),
            'EncodeLayer': GFLIMFunction('iftGFLIMEncodeLayer', tmp_file),
            'MergeLayerModels': GFLIMFunction('iftGFLIMMergeLayerModels', tmp_file),
            'EncodeMergedLayer': GFLIMFunction('iftGFLIMEncodeMergedLayer', tmp_file),
            'EncodeMergedLayerBatch': GFLIMFunction('iftGFLIMEncodeMergedLayerBatch', tmp_file),
            'ExtractFromGraphs': GFLIMFunction('iftGFLIMExtract2', tmp_file),
            'ExtractFromImages': GFLIMFunction('iftGFLIMExtract', tmp_file),
            # options for kernel, pool and stride neighbors different than 'act'
            'ExtractFromImagesColorNeighb': GFLIMFunction('iftGFLIMExtractV3', tmp_file),
        })

    def train_layer(self, layer, dataset : Dataset = None):
        """
        Args:
            layer (int): Layer number.
            dataset (Dataset): The dataset to train on (required when using 'color' neighbor types for kernel or stride).
        The model_dir must contain the seeds created with create_seeds(),
        folders 'labelsX' and 'layerX' for the previous layers / model input,
        and a folder model with weights, bias, and class labels from previous layer (if layer > 1).
        """
        super().train_layer(layer, dataset=dataset)

        if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color": 
            if dataset is None:
                raise ValueError(f"{self.__class__.__name__}.train_layer: dataset must be provided when using 'color' neighbor types for kernel or stride")

        if self.kernel_neighbors == "color" or self.stride_neighbors == "color":
            self.functions['PrecCreateLayerModel']([self.model_dir,
                    self.arch_file, layer, 
                    self.kernel_neighbors,
                    self.stride_neighbors,
                    self.graph_feat,
                    self.aggregation,
                    dataset.images_path])
        else:
            self.functions['CreateLayerModel']([self.model_dir, 
                                           self.arch_file, layer, 
                                           self.stride_type, self.attention, 
                                           self.aggregation])

        self.functions['MergeLayerModels']([self.model_dir, layer])

        if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color": 
            self.functions['ExtractFromImagesColorNeighb']([
                self.model_dir, self.arch_file, layer, 
                self.batch_size, self.attention, 
                self.kernel_neighbors, self.pool_neighbors, 
                self.stride_neighbors, self.graph_feat, 
                self.aggregation, dataset.images_path
            ])
        else:
            if self.batch_size == 1:
                self.functions['EncodeMergedLayer']([self.model_dir, self.arch_file,
                                                    layer, self.stride_type, 
                                                    self.attention, 
                                                    self.aggregation, 
                                                    self.reduce_kernels])
            else:
                self.functions['EncodeMergedLayerBatch']([self.model_dir, 
                                                        self.arch_file, layer, 
                                                        self.batch_size,
                                                        self.stride_type, 
                                                        self.attention, 
                                                        self.aggregation, 
                                                        self.reduce_kernels])

    def extract(self, dataset : Dataset, nlayers, 
                layers_save_graph_act=[], layers_save_img_act=[], 
                layers_decode=[], decoders=['graph-based']):
        """
        Extract features from a model trained with GFLIM BoFP.
        Args:
            dataset (Dataset): The dataset to extract features from.
            nlayers (int): Number of layers.
            layers_save_graph_act (list): List of layers to save the graph activations (saved in 'model_dir/layer<layer>' as json files). 
            layers_save_img_act (list): List of layers to save the image activations (saved in 'model_dir/activations<layer>' as pgm images). 
            layers_decode (list): List of layers to decode. 
            decoders (list): List of decoders to use for decoding the layers. Options: 'user-guided', 'hybrid', 'graph-based'.
        """

        super().extract(dataset, nlayers, layers_save_graph_act, 
                    layers_save_img_act, layers_decode, decoders)

        if dataset is None:
            raise ValueError(f"{self.__class__.__name__}.extract: dataset must be provided to extract features")

        if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color":
            self.functions['ExtractFromImagesColorNeighb']([
                dataset.images_path, self.model_dir, 
                self.arch_file, nlayers, self.batch_size, 
                self.attention, self.aggregation, 
                self.graph_feat, self.spx_alg, 
                self.spx, self.fixed_spx,
                ",".join(str(i) for i in layers_save_graph_act) if len(layers_save_graph_act) > 0 else -1,
                ",".join(str(i) for i in layers_save_img_act) if len(layers_save_img_act) > 0 else -1,
                ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                ",".join(str(i) for i in decoders),
                self.kernel_neighbors, self.pool_neighbors, 
                self.stride_neighbors, 
                self.aggregation, 
                self.initial_seeds,
                1 if dataset.use_masks else 0
                ])
        else:
            if dataset.graphs_path is not None:
                
                if os.path.exists(f"{dataset.graphs_path}/layer0") == False or \
                os.path.exists(f"{dataset.graphs_path}/labels0") == False:
                    raise ValueError(f"{self.__class__.__name__}.extract: \
                        dataset.graphs_path does not contain layer0 and labels0: {dataset.graphs_path}")

                self.functions['ExtractFromGraphs']([
                                    dataset.images_path, f"{dataset.graphs_path}/layer0", 
                                    f"{dataset.graphs_path}/labels0", 
                                    self.model_dir, self.arch_file, 
                                    nlayers, self.batch_size,
                                    self.stride_type, self.attention, 
                                    self.aggregation, 
                                    1 if self.reduce_kernels else 0,
                                    ",".join(str(i) for i in layers_save_graph_act) if len(layers_save_graph_act) > 0 else -1,
                                    ",".join(str(i) for i in layers_save_img_act) if len(layers_save_img_act) > 0 else -1,
                                    ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                                    ",".join(str(i) for i in decoders)])
            else:
                args = [dataset.images_path, 
                        self.model_dir, self.arch_file, 
                        nlayers, self.batch_size,
                        self.stride_type, self.attention, 
                        self.aggregation, 
                        1 if self.reduce_kernels else 0,
                        self.graph_feat,
                        self.spx_alg,
                        self.spx,
                        ",".join(str(i) for i in layers_save_graph_act) if len(layers_save_graph_act) > 0 else -1,
                        ",".join(str(i) for i in layers_save_img_act) if len(layers_save_img_act) > 0 else -1,
                        ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                        ",".join(str(i) for i in decoders)]
                if self.spx_alg == 'DISF':
                    args.append(self.initial_seeds)
                    args.append(1 if dataset.use_masks else 0)

                self.functions['ExtractFromImages'](args)

class RGFLIM_BoFP(GFLIM_BoFP):
    """
    GFLIM BoFP with skip connections.
    """
    def __init__(self, model_dir, arch_file, tmp_file, batch_size=64):
        
        super().__init__(model_dir, arch_file, 
                    tmp_file, batch_size=batch_size)
        
        if self.stride_neighbors != "act" or self.pool_neighbors != "act" or self.kernel_neighbors != "act":
            raise ValueError(f"{self.__class__.__name__} only supports 'act' neighbor types for kernel, pool, and stride")

        self.functions.update({
            'WriteKernelsCorrespondence' : GFLIMFunction('iftGFLIMWriteKernelsCorrespondence', tmp_file),
            'SkipConnection': GFLIMFunction('iftGFLIMSkipConnection', tmp_file),
        })

    def train_layer(self, layer, dataset=None):
        """
        Args:
            layer (int): Layer number.
            dataset (Dataset): not used in RGFLIM_BoFP, but kept as argument for compatibility with GFLIM_BoFP and MGFLIM_BoFP.
        The model_dir must contain the seeds created with create_seeds(),
        folders 'labelsX' and 'layerX' for the previous layers / model input,
        and a folder model with weights, bias, and class labels from previous layer (if layer > 1).
        """

        if layer < 1 or layer > self.nlayers:
            raise ValueError(f"{self.__class__.__name__}.train_layer: layer must be between 1 and {self.nlayers}: {layer}")
        
        # Get the first scale of the layer, which will dictate the number of nodes and features in the merged graph.
        first_scale = self.get_first_scale(layer)

        super().train_layer(layer)

        if layer > 1:
            self.functions['WriteKernelsCorrespondence']([
                os.path.join(self.model_dir, f'markers'), 
                os.path.join(self.model_dir, 'labels0'), 
                os.path.join(self.model_dir, f'labels{layer-1}'), 
                os.path.join(self.model_dir, 'model', 'kernels_map.txt')])
        
        # merge layers output
        if first_scale != layer: # if there are skip connections
            self.functions['SkipConnection']([
                self.model_dir, self.arch_file, 
                layer, first_scale, self.merge_function])
        
    def get_first_scale(self, layer):
        """
        Get the first scale of the given layer considering skip connections.
        Args:
            layer (int): Layer number.
        Returns:
            int: The first scale of the layer.
        Note: If there are no skip connections, the first scale is the layer itself. 
        If there are skip connections, the first scale is the first layer that does not have a skip connection.
        """
        with open(self.arch_file) as f:
            self.arch = json.load(f)
        
        if 'skip_connection' not in self.arch[f'layer{layer}'].keys():
            return layer
        
        scale, parent_scale = layer, -1
        while(parent_scale != scale):
            parent_scale = scale
            for i, skip in enumerate(self.arch[f'layer{scale}']['skip_connection']):
                if skip > 0:
                    scale = i + 1
                    break
        return scale

    def extract(self, dataset : Dataset, nlayers, 
                layers_save_graph_act=[], layers_save_img_act=[], 
                layers_decode=[], decoders=[]):
        if nlayers < 1 or nlayers > self.nlayers:
            raise ValueError(f"{self.__class__.__name__}.extract: \
                nlayers must be between 1 and {self.nlayers}: {nlayers}")
        
        if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color":
            raise ValueError(f"{self.__class__.__name__}.extract: \
                RGFLIM only supports neighborhoods based on activations")
        
        super().extract(dataset, nlayers, 
                layers_save_graph_act=layers_save_graph_act, 
                layers_save_img_act=layers_save_img_act, 
                layers_decode=layers_decode, decoders=decoders)

class MGFLIM_BoFP(RGFLIM_BoFP):
    """
    GFLIM BoFP with skip connections and multiscale.
    """
    def __init__(self, model_dir, arch_file,
                 tmp_file, batch_size=64):

        super().__init__(model_dir, arch_file, 
                    tmp_file, batch_size=batch_size)
        
        if not isinstance(self.spx, list) or len(self.spx) < 2:
            raise ValueError(f"{self.__class__.__name__}: spx must be a list with at least 2 values for multiscale")
        
        if self.spx_alg != 'SH':
            raise ValueError(f"{self.__class__.__name__}: spx_alg must be SH for multiscale")
        
        self.parent_model_dir = model_dir
        self.functions.update({
            'MergeScales': GFLIMFunction('iftGFLIMMergeScales', tmp_file),
            'ExtractMultiscale': GFLIMFunction('iftGFLIMExtractMultiscale', tmp_file),
        })

    def create_seeds(self, dataset : Dataset, img_markers_dir=None):
        """
        Create the seeds for the given architecture.
        Args:
            dataset (Dataset): Dataset object containing the images information.
            img_markers_dir (str): Directory where the image markers are stored (optional).

        If img_markers_dir is not provided, the seeds will be created from all pixels in the training images.
        """
        for spx in self.spx:
            os.makedirs(f"{self.parent_model_dir}/scales/{spx}", exist_ok=True)
            shutil.copytree(f"{self.parent_model_dir}/{spx}", f"{self.parent_model_dir}/scales/{spx}")
            shutil.rmtree(f"{self.parent_model_dir}/{spx}")
            self.model_dir = f"{self.parent_model_dir}/scales/{spx}"
            super().create_seeds(dataset, img_markers_dir)
        self.model_dir = self.parent_model_dir

    def train_layer(self, layer, dataset : Dataset = None):
        """
        Args:
            layer (int): Layer number.
            dataset (Dataset): The dataset to train on (mandatory).
        The model_dir must contain:
            - a folder 'scales', and, for each superpixel value X, a folder 'model_dir/scales/X' with:
                - the seeds created with create_seeds(),
                  folders 'labelsX' and 'layerX' for the previous layers / model input, 
                  and a folder 'model' with the weights, bias, and class labels of the first layer.
            - a folder 'model' with the weights, bias, and class labels of the layers > 1.
            - folders 'labelsX' and 'layerX' for the previous layers of the layers > 1.
        """

        if dataset == None:
            raise ValueError(f"{self.__class__.__name__}.train_layer: dataset must be provided.")

        if dataset.use_masks:
            raise ValueError(f"{self.__class__.__name__}.train_layer: use_masks can only be used with DISF superpixels, and {self.__class__.__name__} only supports SH superpixels.")

        if layer > 1:
            super().train_layer(layer, dataset=dataset)
            return

        for spx in self.spx:
            self.model_dir = f"{self.parent_model_dir}/scales/{spx}"
            super().train_layer(layer, dataset=dataset)

        self.model_dir = self.parent_model_dir
        os.makedirs(os.path.join(self.model_dir, 'model'), exist_ok=True)
        self.functions['MergeScales']([self.dataset.images_path, 
                                       ",".join(str(i) for i in self.spx), 
                                       self.parent_model_dir, 
                                       os.path.join(self.model_dir, 'model', f'kernels_map_{layer}_scales.txt'), 
                                       layer,
                                       self.merge_function])

    def get_first_scale(self, layer):
        """
        Get the first scale of the given layer considering skip connections.
        Args:
            layer (int): Layer number.
        Returns:
            int: The first scale of the layer (as defined in the architecture file).
        Note: If there are no skip connections, the first scale is the layer itself. 
        If there are skip connections, the first scale is the first layer that does not have a skip connection.
        """
        with open(self.arch_file) as f:
            self.arch = json.load(f)
        
        if 'skip_connection' not in self.arch[f'layer{layer}'].keys():
            return layer
        
        scale, parent_scale = layer, -1
        while(parent_scale != scale):
            parent_scale = scale
            for i, skip in enumerate(self.arch[f'layer{scale}']['skip_connection']):
                if skip > 0:
                    scale = i + 1
                    break
        return scale

    def extract_multiscale(self, nlayers, 
                           list_of_graph_dirs : list, 
                           list_of_labels_dirs : list, 
                           layers_save_graph_act=[],
                           layers_save_img_act=[],
                           layers_decode=[], 
                           decoders=[]):
        
        if len(list_of_graph_dirs) < 2:
            raise ValueError(f"{self.__class__.__name__}.extract_multiscale: \
                list_of_graph_dirs must have at least 2 graph directories for multiscale extraction")
        if len(list_of_graph_dirs) != len(list_of_labels_dirs):
            raise ValueError(f"{self.__class__.__name__}.extract_multiscale: \
                list_of_graph_dirs and list_of_labels_dirs must have the same length")
        
        self.functions['ExtractMultiscale']([
            self.dataset.images_path, 
            list_of_graph_dirs,
            list_of_labels_dirs,
            self.parent_model_dir, 
            self.arch_file, nlayers, self.batch_size, 
            self.stride_type, self.attention, self.aggregation, 
            1 if self.reduce_kernels else 0,
            ",".join(str(i) for i in layers_save_graph_act) if len(layers_save_graph_act) > 0 else -1,
            ",".join(str(i) for i in layers_save_img_act) if len(layers_save_img_act) > 0 else -1,
            ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
            ",".join(str(i) for i in decoders) if len(decoders) > 0 else -1,
            ",".join(str(i) for i in self.spx),
            os.path.join(self.model_dir, 'model', f'kernels_map_{nlayers}_scales.txt')
        ])
