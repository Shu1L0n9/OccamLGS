dataset_path="data/lerf_ovs/"
dataset_name="teatime"
output_dir="output/lerf"

python train.py -s ${dataset_path}${dataset_name} -m $output_dir/$dataset_name --iterations 30000
python render.py -m $output_dir/$dataset_name --iteration 30000

python gaussian_feature_extractor.py -m $output_dir/$dataset_name --iteration 30000 --eval --feature_level 1
python feature_map_renderer.py -m $output_dir/$dataset_name --iteration 30000 --eval --feature_level 1