# Occam's LGS: A Simple Approach for Language Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-2412.01807-b31b1b.svg)](https://arxiv.org/abs/2412.01807)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://insait-institute.github.io/OccamLGS/)

This is the official implementation of "Occam's LGS: A simple approach for Language Gaussian Splatting".

## Overview

Occam's LGS is a simple, training-free approach for Language-guided 3D Gaussian Splatting that achieves state-of-the-art results with a 100x speed improvement. Our method:

- 🎯 Lifts 2D language features to 3D Gaussian Splats without complex modules or training
- 🚀 Provides 100x faster optimization compared to existing methods  
- 🧩 Works with any feature dimension without compression
- 🎨 Enables easy scene manipulation and object insertion

## Installation Guide

### System Requirements
We use the following setting to run OccamLGS:

- NVIDIA GPU with CUDA support
- PyTorch 2.2.2
- Python 3.10
- GCC 11.4.0

### Clone Repository
```bash
git clone git@github.com:JoannaCCJH/occamlgs.git --recursive
```

### Environment Setup
```bash
micromamba create -n occamlgs python=3.10
micromamba activate occamlgs
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### Project Dependencies
```bash
pip install -r requirements.txt
micromamba install -c conda-forge gxx=11.4.0
```

### Submodules
```bash
pip install -e submodules/gsplat[dev]
pip install -e submodules/simple-knn
```

## Dataset Preparation
### Input Dataset
The dataset follows a structured format where each 3D scene is organized as follows:
```
lerf_ovs/
└── scene_name/           # Name of the specific scene (e.g., teatime)
    ├── distorted/        
    ├── images/           # Contains the original, unprocessed scene images
    ├── language_features/ # Contains pre-extracted language embeddings
    ├── sparse/0/      
    │   ├── test.txt     # Testing image list
    │   ├── cameras.bin 
    │   ├── images.bin
    │   └── points3D.bin 
    ├── stereo/         
```
Notes:
- Language features are pre-extracted and stored as 512-dimensional vectors
- For detailed information about feature levels and language feature extraction methodology, please refer to the [LangSplat repository](https://github.com/minghanqin/LangSplat). 

### Output Directory Structure
The pre-trained RGB model outputs are organized as follows:
```
output/
└── dataset_name/
    └── scene_name/
        ├── point_cloud/
        │   └── iteration_30000/
        │       └── point_cloud.ply      # Point cloud at 30K iterations
        ├── cameras.json                 
        ├── cfg_args                     
        ├── chkpnt30000.pth             # Model checkpoint at 30K iterations
        └── input.ply                    

```
After running the `gaussian_feature_extractor.py` for three levels of features, three additional checkpoint files are added:

```
output/
└── dataset_name/
    └── scene_name/
        ├── point_cloud/
        │   └── iteration_30000/
        │       └── point_cloud.ply      # Point cloud at 30K iterations
        ├── cameras.json                
        ├── cfg_args                    
        ├── chkpnt30000.pth             # RGB model checkpoint
        ├── input.ply                   
        ├── chkpnt30000_langfeat_1.pth  # Language features level 1
        ├── chkpnt30000_langfeat_2.pth  # Language features level 2
        └── chkpnt30000_langfeat_3.pth  # Language features level 3

```

Note:  The script `gaussian_feature_extractor.py` generates three new semantic checkpoints, each containing a different level of language features while maintaining the same RGB model weights from the original checkpoint.

## Usage


### Prerequisites

-  A pre-trained RGB Gaussian model (use `train.py` and `render.py` commands below to train a model on your scene using gsplat renderer)
- `test.txt` file in `scene_name/sparse/0/` defining test set


#### 1. Train and Render Model
```bash
# Train gaussian model
python train.py -s $DATA_SOURCE_PATH -m $MODEL_OUTPUT_PATH --iterations 30000

# Render trained model
python render.py -m $MODEL_OUTPUT_PATH --iteration 30000
```

#### 2. Feature Extraction and Visualization
```bash
#  gaussian feature vectors
python gaussian_feature_extractor.py -m $MODEL_OUTPUT_PATH --iteration 30000 --eval --feature_level 1

# Render feature maps
python feature_map_renderer.py -m $MODEL_OUTPUT_PATH --iteration 30000 --eval --feature_level 1
```
### Example Pipeline
Check `run_lerf.sh` for a complete example using the "teatime" scene from LERF_OVS dataset

## Evaluation
### LERF
We follow the evaluation methodology established by LangSplat for our LERF assessments. For detailed information about the evaluation metrics and procedures, please refer to the LangSplat methodology.

### 3DOVS
The evaluation protocol for 3DOVS will be released in future updates. For current information about evaluation methods and metrics, please refer to the supplementary materials of our paper.

## TODO
- [x] Training and rendering code released
- [x] GSplat rasterizer code released
- [ ] Corrected room scene labels to be released
- [ ] Evaluation code to be released
- [ ] Autoencoder for any-dimensional feature to be released



## BibTeX

```bibtex
@article{cheng2024occamslgssimpleapproach,
 title={Occam's LGS: A Simple Approach for Language Gaussian Splatting}, 
 author={Jiahuan Cheng and Jan-Nico Zaech and Luc Van Gool and Danda Pani Paudel},
 year={2024},
 eprint={2412.01807}
}
