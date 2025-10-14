# quality/vtss.py
import os
import time
import yaml
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import logging
from ivebench_utils import load_video_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metric_paths(path_yml='path.yml', metric_name='vtss'):
    """Load model checkpoint path from path.yml"""
    try:
        if not os.path.exists(path_yml):
            logger.warning(f"Path configuration file not found: {path_yml}")
            return None
        
        with open(path_yml, 'r', encoding='utf-8') as f:
            paths_config = yaml.safe_load(f)
        
        if metric_name not in paths_config:
            logger.warning(f"Metric '{metric_name}' not found in {path_yml}")
            return None
        
        metric_config = paths_config[metric_name]
        checkpoint_path = metric_config.get('checkpoint')
        
        logger.info(f"Loaded checkpoint path for {metric_name}: {checkpoint_path}")
        
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Error loading metric paths from {path_yml}: {e}")
        return None


class VTSSCalculator:
    
    def __init__(self, device, config_path=None, checkpoint_path=None):
        self.device = device
        self.config_path = config_path or "quality/training_suitability_assessment/infer.yml"
        self.checkpoint_path = checkpoint_path
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"VTSS config file not found: {self.config_path}")
        
        self._load_model()
    
    def _load_model(self):
        try:
            with open(self.config_path, "r") as f:
                opt = yaml.safe_load(f)

            try:
                from quality.training_suitability_assessment.model import DiViDeAddEvaluator
                from quality.training_suitability_assessment.datasets import FusionDataset
            except ImportError:
                raise ImportError("VTSS modules not found. Please install vtss package or check the import path.")
            
            self.model = DiViDeAddEvaluator(**opt["model"]["args"])
            self.model.to(self.device)
            self.model.eval()

            load_path = self.checkpoint_path if self.checkpoint_path else opt["load_path"]
            
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"VTSS model weights not found: {load_path}")
            
            logger.info(f"Loading VTSS model from: {load_path}")
            state_dict = torch.load(load_path, map_location=self.device, weights_only=False)["state_dict"]
            self.model.load_state_dict(state_dict, strict=True)

            self.val_dataset = FusionDataset(opt["data"]['test-data']["args"])
            
            logger.info("VTSS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VTSS model: {e}")
            raise
    
    def process_video_from_frames(self, frame_folder_path):
        if not os.path.exists(frame_folder_path):
            raise FileNotFoundError(f"Frame folder not found: {frame_folder_path}")

        frame_files = sorted([f for f in os.listdir(frame_folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not frame_files:
            raise ValueError(f"No image files found in {frame_folder_path}")

        temp_video_path = self._create_temp_video_from_frames(frame_folder_path, frame_files)
        
        try:
            score = self.process_video(temp_video_path)
            return score
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    def _create_temp_video_from_frames(self, frame_folder_path, frame_files):
        temp_video_path = os.path.join(frame_folder_path, "temp_vtss_video.mp4")
        
        first_frame_path = os.path.join(frame_folder_path, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")
        
        height, width, _ = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 24 
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        for frame_file in frame_files:
            frame_path = os.path.join(frame_folder_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
            else:
                logger.warning(f"Could not read frame: {frame_path}")
        
        out.release()
        return temp_video_path
    
    def process_video(self, video_path):
        start_time = time.perf_counter()
        
        try:
            data = self.val_dataset.prepare_video(video_path)
            video = {}
            
            for key in ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]:
                if key in data:
                    video[key] = data[key].to(self.device).unsqueeze(0)
                    b, c, t, h, w = video[key].shape
                    video[key] = video[key].reshape(
                        b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                    ).permute(0, 2, 1, 3, 4, 5).reshape(
                        b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                    )
            
            with torch.no_grad():
                labels = self.model(video, reduce_scores=False)
                labels = [np.mean(l.cpu().numpy()) for l in labels]
            
            end_time = time.perf_counter()
            score = float(labels[0]) 
            
            logger.debug(f"VTSS processing time: {end_time - start_time:.2f}s, score: {score:.4f}")
            del video, data, labels
            torch.cuda.empty_cache()
            
            return score
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return -1.0


def vtss_single_video(vtss_calculator, video_info, target_videos_path, use_frames=True):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    
    try:
        if use_frames:
            video_name_without_ext = os.path.splitext(video_name)[0]
            target_frame_folder = os.path.join(target_videos_path, video_name_without_ext)
            
            if not os.path.exists(target_frame_folder):
                error_msg = f"Frame folder not found: {target_frame_folder}"
                logger.warning(error_msg)
                return {
                    'video_id': int(video_id),
                    'video_name': str(video_name),
                    'video_results': -1.0,
                    'category': str(video_info['category']),
                    'subcategory': str(video_info['subcategory']),
                    'error': error_msg
                }
            
            score = vtss_calculator.process_video_from_frames(target_frame_folder)
        else:
            target_video_path = os.path.join(target_videos_path, video_name)
            
            if not os.path.exists(target_video_path):
                error_msg = f"Video file not found: {target_video_path}"
                logger.warning(error_msg)
                return {
                    'video_id': int(video_id),
                    'video_name': str(video_name),
                    'video_results': -1.0,
                    'category': str(video_info['category']),
                    'subcategory': str(video_info['subcategory']),
                    'error': error_msg
                }
            
            score = vtss_calculator.process_video(target_video_path)
        
        if score == -1.0:
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'category': str(video_info['category']),
                'subcategory': str(video_info['subcategory']),
                'error': 'Video processing failed'
            }
        
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': float(score),
            'category': str(video_info['category']),
            'subcategory': str(video_info['subcategory'])
        }
        
    except Exception as e:
        error_msg = f"Error processing video {video_name}: {str(e)}"
        logger.error(error_msg)
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': -1.0,
            'category': str(video_info.get('category', '')),
            'subcategory': str(video_info.get('subcategory', '')),
            'error': error_msg
        }


def vtss_evaluation(video_info_list, target_videos_path, device, config_path=None, 
                   checkpoint_path=None, use_frames=True):
    scores = []
    video_results = []
    
    try:
        vtss_calculator = VTSSCalculator(device, config_path, checkpoint_path)
    except Exception as e:
        error_msg = f"Failed to initialize VTSS calculator: {e}"
        logger.error(error_msg)
        for video_info in video_info_list:
            video_results.append({
                'video_id': int(video_info['id']),
                'video_name': str(video_info['src_video_name']),
                'video_results': -1.0,
                'category': str(video_info.get('category', '')),
                'subcategory': str(video_info.get('subcategory', '')),
                'error': error_msg
            })
        return -1.0, video_results
    
    logger.info(f"Processing {len(video_info_list)} videos for VTSS evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating VTSS"):
        result = vtss_single_video(vtss_calculator, video_info, target_videos_path, use_frames)
        video_results.append(result)
        
        if 'error' not in result:
            scores.append(result['video_results'])
            logger.debug(f"Video {result['video_name']}: VTSS score = {result['video_results']:.4f}")
        else:
            logger.warning(f"Video {result['video_name']}: {result['error']}")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        logger.info(f"Overall VTSS score: {avg_score:.4f} (based on {len(scores)}/{len(video_info_list)} valid videos)")
    else:
        avg_score = -1.0
        logger.error("No valid VTSS scores calculated")
    
    return float(avg_score), video_results


def compute_vtss(json_dir, device, source_videos_path=None, target_videos_path=None, 
                 config_path=None, checkpoint_path=None, use_frames=True, 
                 path_yml='path.yml', **kwargs):
    """
    Compute VTSS (Video Training Suitability Score) metric
    
    Args:
        json_dir: Path to JSON file with video information
        device: Device to run evaluation on ('cuda' or 'cpu')
        source_videos_path: Path to source videos (not used in this metric)
        target_videos_path: Path to target videos to evaluate
        config_path: Config file path (uses default if not provided)
        checkpoint_path: Checkpoint file path (if None, will load from path.yml)
        use_frames: Whether to use frames or video files
        path_yml: Path to the YAML file containing model paths
        **kwargs: Additional arguments
    
    Returns:
        tuple: (overall_score, video_results)
    """
    try:
        if checkpoint_path is None:
            logger.info(f"Loading model checkpoint path from {path_yml}")
            checkpoint_path = load_metric_paths(path_yml, 'vtss')
            
            if checkpoint_path is None:
                error_msg = "Checkpoint path must be provided either as argument or in path.yml"
                logger.error(error_msg)
                video_info_list = load_video_info(json_dir, 'vtss')
                video_results = []
                for video_info in video_info_list:
                    video_results.append({
                        'video_id': int(video_info['id']),
                        'video_name': str(video_info['src_video_name']),
                        'video_results': -1.0,
                        'category': str(video_info.get('category', '')),
                        'subcategory': str(video_info.get('subcategory', '')),
                        'error': error_msg
                    })
                return -1.0, video_results
        
        video_info_list = load_video_info(json_dir, 'vtss')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for VTSS evaluation")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")

        overall_score, video_results = vtss_evaluation(
            video_info_list, target_videos_path, device, config_path, checkpoint_path, use_frames
        )
        
        if overall_score == -1.0:
            logger.error("VTSS evaluation failed.")
        else:
            logger.info(f"VTSS evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        error_msg = f"Error in compute_vtss: {str(e)}"
        logger.error(error_msg)
        return -1.0, []