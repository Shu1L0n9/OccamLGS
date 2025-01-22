# OccamLGS

## Prerequisites

- CUDA-capable GPU
- Python 3.10
- GCC 11.4.0

## Installation

### Clone Repository
```bash
git clone git@github.com:JoannaCCJH/occamlgs.git
cd occamlgs
git submodule init
git submodule update --recursive
```

### Environment Setup
```bash
micromamba create -n occamlgs python=3.10
micromamba activate occamlgs
```

### PyTorch Installation
```bash
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


## Usage


### Prerequisites

- Trained gaussian model (use `train.py` and `render.py` commands below to train a model on your scene using gsplat renderer)
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
# Extract gaussian feature vectors
python gaussian_feature_extractor.py -m $MODEL_OUTPUT_PATH --iteration 30000 --eval --feature_level 1

# Render feature maps
python feature_map_renderer.py -m $MODEL_OUTPUT_PATH --iteration 30000 --eval --feature_level 1
```
### Example Pipeline
Check `run_lerf.sh` for a complete example using the "teatime" scene from LERF_OVS dataset

### Important Notes

- Uses gsplat renderer for training and rendering
- Feature extraction follows LangSplat methodology
- Feature levels must be defined before extraction
- Requires `test.txt` in `scene_name/sparse/0/` directory

### References

For more information about feature levels and feature extraction, refer to the LangSplat paper.
