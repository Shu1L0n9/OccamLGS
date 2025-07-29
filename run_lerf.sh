dataset_path="data/lerf_ovs/"
dataset_names=("ramen" "teatime" "figurines" "waldo_kitchen")
output_dir="output/lerf"

for dataset_name in "${dataset_names[@]}"; do
    echo "Processing dataset: $dataset_name"
    # 3dgs训练模型
    python train.py -s ${dataset_path}${dataset_name} -m $output_dir/$dataset_name --iterations 30000
    # 渲染模型
    python render.py -m $output_dir/$dataset_name --iteration 30000
    # 提取特征
    python gaussian_feature_extractor.py -m $output_dir/$dataset_name --iteration 30000 --eval --feature_level 1
    # 渲染特征
    python feature_map_renderer.py -m $output_dir/$dataset_name --iteration 30000 --eval --feature_level 1
done