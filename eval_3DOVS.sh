# Set the dataset name
DATASET_NAME="bench"

# Path to the preprocessed 3DOVS dataset
GT_FOLDER="/path/to/your/3DOVS-preprocess-full/$DATASET_NAME"

# Name of the folder containing extracted features
FEAT_FOLDER_NAME="ours_30000_langfeat"  # Replace with your model's feature folder

python eval/evaluate_iou_3dovs.py \
                    --dataset_name ${DATASET_NAME} \
                    --gt_folder ${GT_FOLDER} \
                    --feat_folder ${FEAT_FOLDER_NAME} \
                    --stability_thresh 0.4 \
                    --min_mask_size 0.005 \
                    --max_mask_size 0.9
