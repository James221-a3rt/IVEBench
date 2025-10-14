# quality/background_consistency.py
import os
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm
import logging
from ivebench_utils import load_video_info, load_frames_from_folder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_background_consistency_single_video(clip_model, preprocess, frames, device):
    if len(frames) < 2:
        logger.warning("Need at least 2 frames to calculate background consistency")
        return 0.0, 0
    
    processed_frames = []
    for frame in frames:
        processed_frame = preprocess(frame)
        processed_frames.append(processed_frame)
    
    images = torch.stack(processed_frames).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
    
    video_sim = 0.0
    cnt_per_video = 0
    first_image_feature = None
    former_image_feature = None
    
    for i in range(len(image_features)):
        image_feature = image_features[i].unsqueeze(0)
        
        if i == 0:
            first_image_feature = image_feature
        else:
            sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
            sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
            cur_sim = (sim_pre + sim_fir) / 2
            video_sim += cur_sim
            cnt_per_video += 1
        
        former_image_feature = image_feature
    
    if cnt_per_video > 0:
        sim_per_frame = video_sim / cnt_per_video
    else:
        sim_per_frame = 0.0
    
    return float(sim_per_frame), int(cnt_per_video)


def background_consistency_single_video(clip_model, preprocess, video_info, target_videos_path, device, use_frames=True):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    
    try:
        if use_frames:
            video_name_without_ext = os.path.splitext(video_name)[0]
            target_frame_folder = os.path.join(target_videos_path, video_name_without_ext)
            frames = load_frames_from_folder(target_frame_folder)
        else:
            raise NotImplementedError("Video file loading not implemented yet, please use frame folders")
        
        consistency_score, frame_count = calculate_background_consistency_single_video(
            clip_model, preprocess, frames, device
        )
        
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': float(consistency_score),
            'frame_count': len(frames),
            'processed_frame_pairs': int(frame_count),
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


def background_consistency(clip_model, preprocess, video_info_list, target_videos_path, device, use_frames=True):
    total_sim = 0.0
    total_cnt = 0
    video_results = []
    
    logger.info(f"Processing {len(video_info_list)} videos for background consistency evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating background consistency"):
        result = background_consistency_single_video(
            clip_model, preprocess, video_info, target_videos_path, device, use_frames
        )
        video_results.append(result)
        
        if 'error' not in result and 'processed_frame_pairs' in result:
            frame_pairs = result['processed_frame_pairs']
            video_sim = result['video_results'] * frame_pairs
            total_sim += video_sim
            total_cnt += frame_pairs
            logger.debug(f"Video {result['video_name']}: consistency = {result['video_results']:.4f}")
    
    if total_cnt > 0:
        overall_consistency = total_sim / total_cnt
    else:
        overall_consistency = 0.0
        logger.warning("No valid frame pairs processed")
    
    logger.info(f"Overall background consistency: {overall_consistency:.4f}")
    
    return float(overall_consistency), video_results


def load_clip_model(model_name="ViT-B/32", device="cuda"):
    try:
        clip_model, preprocess = clip.load(model_name, device=device)
        clip_model.eval()
        logger.info(f"CLIP model {model_name} loaded successfully")
        return clip_model, preprocess
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise


def compute_background_consistency(json_dir, device, source_videos_path=None, target_videos_path=None, 
                                 clip_model_name="ViT-B/32", use_frames=True, **kwargs):
    try:
        logger.info("Loading CLIP model...")
        clip_model, preprocess = load_clip_model(clip_model_name, device)
        
        video_info_list = load_video_info(json_dir, 'background_consistency')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for background consistency evaluation")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = background_consistency(
            clip_model, preprocess, video_info_list, target_videos_path, device, use_frames
        )
        
        logger.info(f"Background consistency evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        logger.error(f"Error in compute_background_consistency: {str(e)}")
        return 0.0, []