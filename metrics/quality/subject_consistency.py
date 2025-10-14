# quality/subject_consistency.py
import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import logging
from ivebench_utils import load_video_info, load_frames_from_folder, dino_transform_Image, load_dino_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def subject_consistency_single_video(model, frames, device, image_transform):
    video_sim = 0.0
    cnt = 0
    
    processed_frames = []
    for frame in frames:
        processed_frame = image_transform(frame)
        processed_frames.append(processed_frame)
    
    first_image_features = None
    former_image_features = None
    
    for i, frame_tensor in enumerate(processed_frames):
        with torch.no_grad():
            image = frame_tensor.unsqueeze(0).to(device)
            
            image_features = model(image)
            image_features = F.normalize(image_features, dim=-1, p=2)
            
            if i == 0:
                first_image_features = image_features
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
            
            former_image_features = image_features
    
    if cnt > 0:
        sim_per_frame = video_sim / cnt
    else:
        sim_per_frame = 0.0
    
    return sim_per_frame, cnt


def subject_consistency(model, video_info_list, target_videos_path, device):
    total_sim = 0.0
    total_cnt = 0
    video_results = []
    
    image_transform = dino_transform_Image(224)
    
    logger.info(f"Processing {len(video_info_list)} videos for subject consistency evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating subject consistency"):
        try:
            video_name = video_info['src_video_name']
            video_id = video_info['id']
            
            video_name_without_ext = os.path.splitext(video_name)[0]
            target_frame_folder = os.path.join(target_videos_path, video_name_without_ext)
            
            if not os.path.exists(target_frame_folder):
                logger.warning(f"Target frame folder not found: {target_frame_folder}")
                video_results.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'video_results': 0.0,
                    'error': 'Target frame folder not found'
                })
                continue
            
            frames = load_frames_from_folder(target_frame_folder)
            
            if len(frames) < 2:
                logger.warning(f"Video {video_name} has less than 2 frames, skipping")
                video_results.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'video_results': 0.0,
                    'error': 'Insufficient frames'
                })
                continue
            
            video_sim, frame_cnt = subject_consistency_single_video(
                model, frames, device, image_transform
            )
            
            total_sim += video_sim * frame_cnt
            total_cnt += frame_cnt
            
            video_results.append({
                'video_id': video_id,
                'video_name': video_name,
                'video_results': video_sim,
                'frame_count': len(frames),
                'category': video_info['category'],
                'subcategory': video_info['subcategory']
            })
            
            logger.debug(f"Video {video_name}: consistency = {video_sim:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing video {video_info.get('src_video_name', 'unknown')}: {str(e)}")
            video_results.append({
                'video_id': video_info.get('id', -1),
                'video_name': video_info.get('src_video_name', 'unknown'),
                'video_results': 0.0,
                'error': str(e)
            })
    
    if total_cnt > 0:
        overall_consistency = total_sim / total_cnt
    else:
        overall_consistency = 0.0
    
    logger.info(f"Overall subject consistency: {overall_consistency:.4f}")
    
    return overall_consistency, video_results


def compute_subject_consistency(json_dir, device, source_videos_path=None, target_videos_path=None, **kwargs):
    try:
        logger.info("Loading DINO model...")
        dino_model = load_dino_model(device)
        logger.info("DINO model loaded successfully")
        
        video_info_list = load_video_info(json_dir, 'subject_consistency')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for subject consistency evaluation")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = subject_consistency(
            dino_model, video_info_list, target_videos_path, device
        )
        
        logger.info(f"Subject consistency evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        logger.error(f"Error in compute_subject_consistency: {str(e)}")
        return 0.0, []