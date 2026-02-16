# Python scripts

This folder contain scripts to facilitate GFLIM usage.

After downloading the LibIFT project along with GFLIM and Classify modules and this pyGFLIM folder with scripts, you must have the folowing structure:

```bash
    LibIFT
    ├── bin                             # Executable files
    ├── build                           # Files generated during the compilation and linking process
    ├── cmake
    ├── Datasets                        # Dataset used
    |   ├── archs/                      # Architecture files in .json format
    │   └── helminth-eggs               # Helminth eggs dataset
    │       ├── images/                 # Original input images
    │       ├── masks/                  # Corresponding masks
    │       └── splits_incremental/     # Progressive splits for incremental experiment
    ├── demo
    |   ├── Classification/             # Source code for SVM
    |   └── GFLIM/                      # Source code of GFLIM programs
    |       ├── GFLIM/
    |       ├── GFLIM_BoFP_precompGraph/
    |       ├── GFLIM-BoFP/
    |       └── M-GFLIM-BoFP/
    ├── include                         # LibIFT header files
    ├── src                             # LibIFT source files in C
    ├── externals
    ├── lib
    ├── pyGFLIM
    ├── Results
    └── windows
```

The folder [Datasets](Datasets/) contains the Helminth eggs datasets from the repository [intestinal-parasites-datasets](https://github.com/LIDS-UNICAMP/intestinal-parasites-datasets) along with a folder called `splits_incremental` with json files defining the training, validation and test sets for diferent percentages of training data and 50% of test data.

## Tutorial

This tutorial shows how to use all GFLIM BoFP functionalities with the default parameters. See the python scripts used in this tutorial for more parameters choices. For the other GFLIM versions, you can create similar train and extract scripts using the classes defined in [GFLIMFunctions](GFLIMFunctions.py). Be aware that this code may contain bugs. For simplicity, this tutorial will refers to GFLIM BoFP as GFLIM and will not mention any other GFLIM version.

### Compile and set a common variable to the project

Fist, compile the LibIFT project with:

```bash
cd build; cmake ..; make; cp libift.a ../lib/; make install;
```

This will create executable files in `bin`.

Since the programs here use complete paths, set to the variable `LibIFT` the path to this project: 

```bash
${LibIFT}=path_to_LibIFT`
```

### Define your training set

We will create a folder `train.csv` with the training images defined on the file [data_descriptor_perc5.json](Datasets/helinth-eggs/splits_incremental/split1/data_descriptor_perc5.json).

The file will be created on `"${LibIFT}/Results/helminth-eggs/split1/gflim_model/"`, which will also contain our model.

```bash
images_dir="${LibIFT}/Datasets/intestinal-parasites/helminth-eggs/images"; # original images directory
split_file="${LibIFT}/Datasets/intestinal-parasites/helminth-eggs/splits_incremental/split1/data_descriptor_perc5.json"; # file defining training, validation and test sets
model_dir="${LibIFT}/Results/helminth-eggs/split1/gflim_model/"; # directory with the model files
train_file="${model_dir}/train.csv"; # output file with the complete paths to the training images
python3 CreateCSVFromJson.py ${split_file} ${images_dir} training ${svm_train_file} 
```

#### Define a training set only for GFLIM training

The file `train.csv` may have too many images for GFLIM. You can select just a few images per class from `train.csv` with the following code:

```bash
gflim_train_file="${model_dir}/trainGFLIM.csv"; # file with less images for training a model using GFLIM
python3 choose_images_per_class.py ${svm_train_file} ${gflim_train_file}
```

By default, [choose_images_per_class](choose_images_per_class.py), will select one image per class from `train.csv`.

### Training a model with GFLIM

To train a GFLIM model we must to have a json file defining the archtecture parameters. We will use the file [arch.json](Datasets/archs/arch.json) at Dataset/arch. Since a superpixel segmentation is used to create the graphs, we will set `250` superpixels per image, resulting in `250` vertices per graph, and 1 layer. Other parameters are available at [GFLIMBoF-train](GFLIMBoF-train.py).

For simplicity, we will use all graph nodes as labeled samples to create seeds. You can provide your own markers through the parameter `--markers_dir`. Since the architecture file [arch.json](Datasets/archs/arch.json) defines 16 kernels per marker, and we are using the whole image, it will create 16 kernels per training image per layer.

Run the following to train a model:

```bash
arch_file="${LibIFT}/Datasets/archs/arch.json";
python3 GFLIMBoF-train.py ${gflim_train_file} ${model_dir} 250 ${arch_file} --nlayers 1
```

> [!NOTE]  
> You can replace `gflim_train_file` to `train_file` to train using all training files. Since the model is designed for training in a few images, it is not optimized to handle several files during training.

After the training, your model folder will have the following structure:

```bash
    gflim_model
    ├── img_seeds/      # Text files with the image seeds (pixel coordinates and class label)
    ├── labels0/        # Superpixel maps from the input images
    ├── labels1/        # Superpixel maps from the output graphs of the first layer
    ├── layer0/         # Graphs from the input images
    ├── layer1/         # Output graphs of the first layer
    ├── markers/        # Text files with the graph seeds after k-means. In this case, each file will contain the information of 16 vertices
    ├── model/          # Weights, bias, and labels (which class each kernels came from) for each training image and layer, and their unified version
    ├── arch.json       # A copy of the architecture file used during training
    ├── config.json     # Configuration file containing all the model parameters that does not refer to the archtecture
    ├── train.csv       # File with the complete training set
    └── trainGFLIM.csv  # File with one image per class, used to train a model with GFLIM
```

The minimal files / folder for a trained model are the files on `model` and `markers`, and the files `arch.json` and `config.json`. The folder `layer1` contains the output graphs, and the superpixel maps in `labels1` are useful to transform these graphs back to images. The information in `labels0` can also be used when stride > 0.

### Visualizing the activated regions

We can visualize the activations on the image regions using the information in `labels1` and `layer1` with the following code:

```bash
python3 GFLIM-activations.py ${model_dir} --layer 1
```

This will create a folder `activation1` in `gflim_model` containing a folder per class (in this case, `label1`to `label9`). An image `gflim_model/activation1/label2/000001_00000001/kernel-017.pgm` is the activation that the kernel 17 of layer 1 produces on the image 000001_00000001. In this case, kernel 17 was created from a vertice labeled with class 2 (indicated by the folder `label2`).

### Decoding

The following code will create saliency maps from the images in `layer1`:

```bash
python3 GFLIM-decode.py ${model_dir} --layer 3
```

The saliency maps will be created using the default `graph-based decoder` and are stored at `graph-adaptive_1`. You can use other decoders, as specified in [GFLIM-decode](GFLIM-decode.py).

### Extract features

> [!WARNING]  
> Before using the model to extract feature from other images we may want to delete the `layer` and `labels` folder in `gflim_model` directory. For a layer `X`, the same `layerX` and `labelsX` folders will contain the output graphs and superpixel maps from the extract and training processes. Therefore, there folders might acumulate files.

To use the model to extract features from other images (e.g., the whole trainig set), we can run the following:

```bash
python3 GFLIMBoF-extract.py ${train_file} ${model_dir} --nlayers 3 --layers_outputs 1
```

Here, `--layers_outputs 1` indicate that we want the output graphs and superpixel maps from the layer 1. If a deeper layer is trained (e.g., with 3 layers), we could define `--layers_outputs 1,2,3` for the outputs of all layers.

After this, we can use the scripts [GFLIM-activations](GFLIM-activations.py) or [GFLIM-decode](GFLIM-decode.py), but it will be faster use the parameters in [GFLIMBoF-extract](GFLIMBoF-extract.py) to generate these outputs too.

### Train SVM

The output graphs can be used to train an SVM:

```bash
svm_model="svm_layer1.zip";
python3 train_svm.py ${svm_train_file} ${model_dir}/layer3 ${model_dir} ${svm_model} 
```

This scripts create a vector from each graph and store the vectors in a .zip file using the `iftDataset` type from LibIFT. Then, the .zip file will be used to train the SVM. You will note the files `layer1.zip` and `svm_layer1.zip` in `gflim_model`. The former is the output graphs in `iftDataset` format, and the latter is the SVM model.

### Classify using SVM

Before the classification, you may want to delete the file `layer1.zip` and the `layerX` and `labelsX` created during training to avoid testing the SVM on training image.

The following code will create a CSV file with the validation images and use the model trained with GFLIM to extract features from them:

```bash
svm_validation_file="${model_dir}/validation.csv";
python3 CreateCSVFromJson.py ${split_file} ${images_dir} validation ${svm_validation_file}
python3 GFLIMBoF-extract.py ${svm_validation_file} ${model_dir} --nlayers 1 --layers_outputs 1
```

Now we can classify:

```bash
python3 svm_classify.py ${svm_validation_file} ${model_dir} 1 ${model_dir}/${svm_model}
```

The model folder will have the following structure:

```bash
    gflim_model
    ├── img_seeds/      
    ├── labels0/        
    ├── labels1/        
    ├── layer0/         
    ├── layer1/         
    ├── markers/        
    ├── model/       
    ├── results_validation_layer1/  
    |   ├── classification.txt              # Classification results, including recall per class, accuracy, and kappa 
    |   ├── layer1-classified-images.csv    # Which class each image was assigned by the SVM
    |   └── results.csv                     # Only recall per class, accuracy, and kappa
    ├── arch.json       
    ├── config.json    
    ├── layer1.zip          # The output graphs from GFLIM embedded in vectors and converted to `iftDataset` format.
    ├── layer1_id_image.txt # Contain the graphs' ordering on `layer1.zip`.
    ├── train.csv       
    └── trainGFLIM.csv  
```
