#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.dlpack
import matplotlib.pyplot as plt
import time


def render_set(model_path, name, views, gaussians, pipeline, background, feature_level, dataset_name, save_name, dataset_folder):
    
   
    save_path = f"/scratch/joanna_cheng/{model_path.split('/')[-1]}" #!!!! saved path
    folder_name = f"{save_name}_{feature_level}" #!!!! folder name
    render_path = os.path.join(save_path, name, folder_name, "renders")
    gts_path = os.path.join(save_path, name, folder_name, "gt")

    language_feature_save_path = os.path.join(save_path, "language_features")
    ckpt_save_path = os.path.join(save_path, "ckpt")
    
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(language_feature_save_path, exist_ok=True)
    os.makedirs(ckpt_save_path, exist_ok=True)
    
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_pkg= render(view, gaussians, pipeline, background)

        gt_language_feature, gt_mask = view.get_language_feature(language_feature_dir=f"/scratch/joanna_cheng/{dataset_folder}/{dataset_name}/language_features", feature_level=feature_level)
        activated = render_pkg["info"]["activated"]
        significance = render_pkg["info"]["significance"]
        means2D = render_pkg["info"]["means2d"]
        
        mask = activated[0] > 0
        gaussians.initialize_language_feature(gt_language_feature.permute(1, 2, 0), gt_mask.squeeze(0), mask, significance[0,mask], means2D[0, mask])
        

    gaussians.update_language_feature()
    gaussians.save_ply_rgb(f"/scratch/joanna_cheng/{dataset_name}-exp18/{dataset_name}_{feature_level}.ply")
    

    torch.save((gaussians.capture_rgb(), 0), ckpt_save_path + f"/chkpnt{str(0)}_{feature_level}.pth")
    print("Ckpt saved to: ", ckpt_save_path + f"/chkpnt{str(0)}_{feature_level}.pth")
    torch.save(gaussians._language_feature.detach(), 
               language_feature_save_path + f"/language_feature_{feature_level}.pt")
    print("Number of language features: ", gaussians._language_feature.shape[0])
    print("language_features saved to: ", language_feature_save_path + "/language_feature.pt")
    
    
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     render_pkg= render(view, gaussians, pipeline, background, include_feature=True)
    #     rendering, weighting_sum = render_pkg["render"], render_pkg["weighting_sum"]
    #     gt, mask = view.get_language_feature(language_feature_dir=f"/scratch/joanna_cheng/{dataset_folder}/{dataset_name}/language_features", feature_level=feature_level) #! modified
        
    #     _, H, W = gt.shape
    #     gt = gt.reshape(512, -1).T.cpu().numpy()
    #     rendering = rendering.reshape(512, -1).T.cpu().numpy() # (H*W, 512)
        
    #     pca = PCA(n_components=3)
    #     # rendering_features = pca.fit_transform(rendering)
    #     combined_np = np.concatenate((gt, rendering), axis=0)
    #     combined_features = pca.fit_transform(combined_np) # ((n+m)*H*W, 3)
    #     normalized_features = (combined_features - combined_features.min(axis=0)) / (combined_features.max(axis=0) - combined_features.min(axis=0))
    #     reshaped_combined_features = normalized_features.reshape(2, H, W, 3)
        
    #     reduced_rendering = reshaped_combined_features[1]
    #     reduced_gt = reshaped_combined_features[0]
        
    #     rendering = torch.tensor(reduced_rendering).permute(2, 0, 1)
    #     gt = torch.tensor(reduced_gt).permute(2, 0, 1)
    #     # rendering = torch.tensor(rendering_features).permute(1, 0).reshape(-1, H, W)
        
    #     torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
    #     torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
            
def render_test_set(model_path, name, views, gaussians, pipeline, background, feature_level, dataset_name, save_name, dataset_folder):
    
    save_path = f"/scratch/joanna_cheng/{model_path.split('/')[-1]}" #!!!! saved path
    folder_name = f"{save_name}_{feature_level}" #!!!! folder name
    render_path = os.path.join(save_path, name, folder_name, "renders")
    gts_path = os.path.join(save_path, name, folder_name, "gt")
    render_npy_path = os.path.join(save_path, name, folder_name, "renders_npy")
    gts_npy_path = os.path.join(save_path, name, folder_name, "gt_npy")
    
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_npy_path, exist_ok=True)
    os.makedirs(gts_npy_path, exist_ok=True)
    
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg= render(view, gaussians, pipeline, background, include_feature=True)
        rendering, weighting_sum = render_pkg["render"], render_pkg["weighting_sum"]
        gt, mask = view.get_language_feature(language_feature_dir=f"/scratch/joanna_cheng/{dataset_folder}/{dataset_name}/language_features", feature_level=feature_level) #! modified
        
        np.save(os.path.join(render_npy_path, view.image_name + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, view.image_name + ".npy"),gt.permute(1,2,0).cpu().numpy())
        
        _, H, W = gt.shape
        gt = gt.reshape(512, -1).T.cpu().numpy()
        rendering = rendering.reshape(512, -1).T.cpu().numpy() # (H*W, 512)
        
        
        pca = PCA(n_components=3)
        # rendering_features = pca.fit_transform(rendering)
        combined_np = np.concatenate((gt, rendering), axis=0)
        combined_features = pca.fit_transform(combined_np) # ((n+m)*H*W, 3)
        normalized_features = (combined_features - combined_features.min(axis=0)) / (combined_features.max(axis=0) - combined_features.min(axis=0))
        reshaped_combined_features = normalized_features.reshape(2, H, W, 3)
        
        reduced_rendering = reshaped_combined_features[1]
        reduced_gt = reshaped_combined_features[0]
        
        rendering = torch.tensor(reduced_rendering).permute(2, 0, 1)
        gt = torch.tensor(reduced_gt).permute(2, 0, 1)
        # rendering = torch.tensor(rendering_features).permute(1, 0).reshape(-1, H, W)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_level : int, scene_name: str, save_name: str, dataset_folder: str): #!!!! added

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.getTrainCameras(), gaussians, pipeline, background, feature_level, scene_name, save_name, dataset_folder)
        
        if not skip_test:
             render_test_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, feature_level, scene_name, save_name, dataset_folder)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scene_name", type=str)
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--dataset", default="lerf_ovs", type=str)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.feature_level, args.scene_name, args.save_name, args.dataset)