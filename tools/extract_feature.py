#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import numpy as np
import os
import pickle
import torch
import slowfast.utils.checkpoint as cu
import slowfast.utils.misc as misc
from slowfast.models import build_model
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import decord as de
from tqdm import tqdm
import skvideo.io
import pdb

# python -u tools/extract_feature.py --cfg ../SLOWFAST_8x8_R50.yaml TEST.CHECKPOINT_FILE_PATH ../SLOWFAST_8x8_R50.pkl OUTPUT_DIR /data2/ldc/JIGSAWS/ DATA.PATH_TO_DATA_DIR /data2/ldc/JIGSAWS/video_encoded

# python -u tools/extract_feature.py --cfg ../I3D_8x8_R50.yaml TEST.CHECKPOINT_FILE_PATH ../I3D_8x8_R50.pkl OUTPUT_DIR /data2/ldc/JIGSAWS/ DATA.PATH_TO_DATA_DIR /data2/ldc/JIGSAWS/video_encoded

def test(cfg):

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    
    de.bridge.set_bridge('native')
    
    gpu_id = 1
    
#     sample_rate = 1
#     resize_h = 270
#     resize_w = 360
#     augment = ['FiveCrop', 'TenCrop', 'None'][1]

    sample_rate = 1
    resize_h = 270
    resize_w = 360
    augment = ['FiveCrop', 'TenCrop', 'None'][0]
    
    crop_h = cfg.DATA.TEST_CROP_SIZE # 256
    crop_w = cfg.DATA.TEST_CROP_SIZE # 256
    
    if 'SLOWFAST' in cfg.TEST.CHECKPOINT_FILE_PATH and 'I3D' not in cfg.TEST.CHECKPOINT_FILE_PATH:
        model_type = 'slowfast'
        feature_dim = 2304
    elif 'SLOWFAST' not in cfg.TEST.CHECKPOINT_FILE_PATH and 'I3D' in cfg.TEST.CHECKPOINT_FILE_PATH:
        model_type = 'i3d'
        feature_dim = 2048
    else:
        raise Exception('Invalid Model.')

    video_dir = cfg.DATA.PATH_TO_DATA_DIR

    if augment == 'FiveCrop':
        feature_dir = os.path.join(cfg.OUTPUT_DIR, 'feature_{}_{}x{}_{}x{}_{}_5'.format(
            model_type, resize_h, resize_w, crop_h, crop_w, sample_rate))
    elif augment == 'TenCrop':
        feature_dir = os.path.join(cfg.OUTPUT_DIR, 'feature_{}_{}x{}_{}x{}_{}_10'.format(
            model_type, resize_h, resize_w, crop_h, crop_w, sample_rate))
    elif augment == 'None':
        feature_dir = os.path.join(cfg.OUTPUT_DIR, 'feature_{}_{}x{}_{}_1'.format(
            model_type, resize_h, resize_w, sample_rate))
    else:
        raise Exception('Invalid Augment.')
        
    norm_transform = transforms.Normalize(
        mean=cfg.DATA.MEAN,
        std=cfg.DATA.STD
    )
    
    if augment == 'FiveCrop':
        frame_transform = transforms.Compose([
            transforms.Resize(size=(resize_h, resize_w)),
            transforms.FiveCrop(size=(crop_h, crop_w)),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [norm_transform(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack(crops))
        ])  
    elif augment == 'TenCrop':
        frame_transform = transforms.Compose([
            transforms.Resize(size=(resize_h, resize_w)),
            transforms.TenCrop(size=(crop_h, crop_w)),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [norm_transform(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack(crops))
        ])  
    elif augment == 'None':
        frame_transform = transforms.Compose([
            transforms.Resize(size=(resize_h, resize_w)),
            transforms.ToTensor(),
            norm_transform,
            transforms.Lambda(lambda img: img.unsqueeze(0))
        ])   
    else:
        raise Exception('Invalid Augment.')

    # Build the video model and print model statistics.
    model = build_model(cfg)
    print(model)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    model.to(torch.device('cuda:{}'.format(gpu_id)))

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    video_files = os.listdir(video_dir)
    video_files.sort()

    for video_file in video_files:

        video_name = video_file[:-4]
        video_file = os.path.join(video_dir, video_file)

        feature_file = '{}.npy'.format(video_name)
        if feature_file in os.listdir(feature_dir):
            print('Skipped.')
            continue
        feature_file = os.path.join(feature_dir, feature_file)

        print(video_file)
        print(feature_file)

        video_feature = []

        vr = de.VideoReader(video_file, ctx=de.cpu(0))

        frame_num = len(vr)
        video_meta = skvideo.io.ffprobe(video_file)
        assert(frame_num == int(video_meta['video']['@nb_frames']))           
        
        sample_idxs = np.arange(0, frame_num, sample_rate)
        
        clip_size = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
        
#         cfg.DATA.NUM_FRAMES
#         cfg.DATA.SAMPLING_RATE
#         cfg.SLOWFAST.ALPHA

        frame_buffer = {}
        buffer_size = 128

        with torch.no_grad():

            for _, sample_idx in enumerate(tqdm(sample_idxs)):
                
                fast_pathway_idxs = np.arange(
                    sample_idx - clip_size // 2, 
                    sample_idx - clip_size // 2 + clip_size, 
                    cfg.DATA.SAMPLING_RATE
                )
                
                fast_pathway_idxs[fast_pathway_idxs < 0] = 0
                fast_pathway_idxs[fast_pathway_idxs > frame_num - 1] = frame_num - 1
                assert(fast_pathway_idxs.size == cfg.DATA.NUM_FRAMES)
                
                fast_pathway_frames = []
                
                for idx in fast_pathway_idxs:
                    
                    if idx not in frame_buffer:
                        frame = vr[idx].asnumpy() #(540, 960, 3)
                        frame = Image.fromarray(frame)
                        frame = frame_transform(frame)
                        frame = frame.to(torch.device('cuda:{}'.format(gpu_id)))

                        if augment == 'FiveCrop':
                            assert(frame.shape[0]==5)
                            assert(frame.shape[1]==3)
                            assert(frame.shape[2]==crop_h)
                            assert(frame.shape[3]==crop_w)
                        elif augment == 'TenCrop':
                            assert(frame.shape[0]==10)
                            assert(frame.shape[1]==3)
                            assert(frame.shape[2]==crop_h)
                            assert(frame.shape[3]==crop_w)
                        elif augment == 'None':
                            assert(frame.shape[0]==1)
                            assert(frame.shape[1]==3)
                            assert(frame.shape[2]==resize_h)
                            assert(frame.shape[3]==resize_w)
                        else:
                            raise Exception('Invalid Augment.')
                            
                        frame_buffer[idx] = frame
                        if len(frame_buffer) > buffer_size:
                            frame_buffer.pop(min(list(frame_buffer.keys())))
                    
                    fast_pathway_frames.append(frame_buffer[idx].unsqueeze(2))
                    
                fast_pathway_frames = torch.cat(fast_pathway_frames, 2)

                if model_type == 'slowfast':
                    
                    slow_pathway_idxs = fast_pathway_idxs[::cfg.SLOWFAST.ALPHA]
                    assert(slow_pathway_idxs.size == cfg.DATA.NUM_FRAMES / cfg.SLOWFAST.ALPHA)
                    slow_pathway_frames = []

                    for idx in slow_pathway_idxs:

                        if idx not in frame_buffer:
                            frame = vr[idx].asnumpy() #(540, 960, 3)
                            frame = Image.fromarray(frame)
                            frame = frame_transform(frame)
                            frame = frame.to(torch.device('cuda:{}'.format(gpu_id)))

                            if augment == 'FiveCrop':
                                assert(frame.shape[0]==5)
                                assert(frame.shape[1]==3)
                                assert(frame.shape[2]==crop_h)
                                assert(frame.shape[3]==crop_w)
                            elif augment == 'TenCrop':
                                assert(frame.shape[0]==10)
                                assert(frame.shape[1]==3)
                                assert(frame.shape[2]==crop_h)
                                assert(frame.shape[3]==crop_w)
                            elif augment == 'None':
                                assert(frame.shape[0]==1)
                                assert(frame.shape[1]==3)
                                assert(frame.shape[2]==resize_h)
                                assert(frame.shape[3]==resize_w)
                            else:
                                raise Exception('Invalid Augment.')

                            frame_buffer[idx] = frame
                            if len(frame_buffer) > buffer_size:
                                frame_buffer.pop(min(list(frame_buffer.keys())))

                        slow_pathway_frames.append(frame_buffer[idx].unsqueeze(2))
                    
                    slow_pathway_frames = torch.cat(slow_pathway_frames, 2)
                                    
                if model_type == 'slowfast':
                    frame_feature = model(
                        [slow_pathway_frames, fast_pathway_frames], 
                        extract_feature=True)
                elif model_type == 'i3d':
                    frame_feature = model(
                        [fast_pathway_frames], 
                        extract_feature=True)
                else:
                    raise Exception('Invalid Model.')
                
                
                # (Pdb) fast_pathway_frames.shape
                # torch.Size([5, 3, 32, 256, 256])

                # (Pdb) slow_pathway_frames.shape
                # torch.Size([5, 3, 8, 256, 256])

                assert(frame_feature.shape[1]==feature_dim)
                if augment == 'FiveCrop':
                    assert(frame_feature.shape[0]==5)
                elif augment == 'TenCrop':
                    assert(frame_feature.shape[0]==10)
                elif augment == 'None':
                    assert(frame_feature.shape[0]==1)
                else:
                    raise Exception('Invalid Augment.')  

                # slowfast is for 30 fps! be careful!
                # re-extract all!

                frame_feature = torch.unsqueeze(frame_feature, dim=0)
                frame_feature = frame_feature.cpu().numpy()

                video_feature.append(frame_feature)
                
        video_feature = np.concatenate(video_feature, axis=0)

        print(video_feature.shape)

        np.save(feature_file, video_feature)



def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

        
if __name__ == "__main__":
    main()


