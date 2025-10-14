# quality/temporal_flickering.py
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
import logging
from ivebench_utils import load_video_info, get_frames_from_folder, get_frames_from_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_mae(img1, img2):
    if img1.shape != img2.shape:
        logger.warning("Images don't have the same shape.")
        return 0.0
    
    mae = np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))
    return float(mae)


def mae_sequence(frames):
    maes = []
    for i in range(len(frames) - 1):
        mae = calculate_mae(frames[i], frames[i + 1])
        maes.append(mae)
    return maes


def calculate_flickering_score(frames):
    if len(frames) < 2:
        logger.warning("Need at least 2 frames to calculate flickering")
        return 0.0
    
    mae_scores = mae_sequence(frames)
    
    avg_mae = sum(mae_scores) / len(mae_scores) 
    flickering_score = (255.0 - avg_mae) / 255.0
    
    flickering_score = max(0.0, min(1.0, flickering_score))
    
    return float(flickering_score)


def temporal_flickering_single_video(video_info, target_videos_path, use_frames=True):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    
    try:
        if use_frames:
            video_name_without_ext = os.path.splitext(video_name)[0]
            target_frame_folder = os.path.join(target_videos_path, video_name_without_ext)
            frames = get_frames_from_folder(target_frame_folder)
        else:
            target_video_path = os.path.join(target_videos_path, video_name)
            frames = get_frames_from_video(target_video_path)
        
        score = calculate_flickering_score(frames)
        
        return {
            'video_id': int(video_id),  
            'video_name': str(video_name),  
            'video_results': float(score),  
            'frame_count': int(len(frames)),  
            'category': str(video_info['category']),
            'subcategory': str(video_info['subcategory'])
        }
        
    except Exception as e:
        logger.error(f"Error processing video {video_name}: {str(e)}")
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': 0.0,
            'error': str(e)
        }


def temporal_flickering(video_info_list, target_videos_path, use_frames=True):
    scores = []
    video_results = []
    
    logger.info(f"Processing {len(video_info_list)} videos for temporal flickering evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating temporal flickering"):
        result = temporal_flickering_single_video(video_info, target_videos_path, use_frames)
        video_results.append(result)
        
        if 'error' not in result:
            scores.append(result['video_results'])
            logger.debug(f"Video {result['video_name']}: flickering score = {result['video_results']:.4f}")
    
    if scores:
        avg_score = sum(scores) / len(scores)  
    else:
        avg_score = 0.0
        logger.warning("No valid video scores calculated")
    
    logger.info(f"Overall temporal flickering score: {avg_score:.4f}")
    
    return float(avg_score), video_results


def compute_temporal_flickering(json_dir, device, source_videos_path=None, target_videos_path=None, 
                               use_frames=True, **kwargs):
    try:
        video_info_list = load_video_info(json_dir, 'temporal_flickering')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for temporal flickering evaluation")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = temporal_flickering(
            video_info_list, target_videos_path, use_frames
        )
        
        logger.info(f"Temporal flickering evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        logger.error(f"Error in compute_temporal_flickering: {str(e)}")
        return 0.0, []