DATASET_NAME="bench"
OUTPUT_DIR="/home/joanna_cheng/workspace/occamlgs_new/output/3DOVS"

cd ~/workspace/occamlgs_new

python train.py -s /scratch/joanna_cheng/3DOVS-preprocess-full/$DATASET_NAME -m $OUTPUT_DIR/$DATASET_NAME --iterations 30000
python render.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000

python gaussian_feature_extractor.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000 --eval --feature_level 3
python feature_map_renderer.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000 --eval --feature_level 1 --skip_train