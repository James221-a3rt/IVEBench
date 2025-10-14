import os
import logging
import yaml
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from ivebench_utils import load_video_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from compliance.videoclipxl_utils.modeling import VideoCLIP_XL
    from compliance.videoclipxl_utils.text_encoder import text_encoder
    VIDEOCLIP_AVAILABLE = True
except ImportError:
    logger.warning("VideoCLIP-XL modules not available. Please ensure modeling and utils modules are in the Python path.")
    VIDEOCLIP_AVAILABLE = False


def load_metric_paths(path_yml='path.yml', metric_name='overall_semantic_consistency'):
    """Load model path from path.yml"""
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
        model_path = metric_config.get('model_path')
        
        logger.info(f"Loaded model path for {metric_name}: {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error loading metric paths from {path_yml}: {e}")
        return None


class VideoCLIPEvaluator:
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        
        self._load_model()
    
    def _load_model(self):
        if not VIDEOCLIP_AVAILABLE:
            error_msg = "VideoCLIP-XL modules not available"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading VideoCLIP-XL model from {self.model_path}")
            
            self.model = VideoCLIP_XL()
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("VideoCLIP-XL model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load VideoCLIP-XL model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_frames_from_folder(self, folder_path, fnum=8):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        frame_files = []
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                frame_files.append(os.path.join(folder_path, file))
        
        frame_files.sort()
        
        if len(frame_files) == 0:
            raise ValueError(f"No image files found in {folder_path}")
        
        step = max(1, len(frame_files) // fnum)
        selected_files = frame_files[::step][:fnum]
        
        frames = []
        for file_path in selected_files:
            img = Image.open(file_path).convert('RGB')
            frame = np.array(img)
            frames.append(frame)
        
        return frames
    
    def load_frames_from_video(self, video_path, fnum=8):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")

        step = max(1, total_frames // fnum)
        
        frames = []
        frame_indices = [i * step for i in range(fnum)]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if len(frames) >= fnum:
                break
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        return frames
    
    def normalize(self, data):
        return (data / 255.0 - self.v_mean) / self.v_std
    
    def frames_preprocessing(self, video_path, fnum=8):
        if os.path.isdir(video_path):
            frames = self.load_frames_from_folder(video_path, fnum)
        elif os.path.isfile(video_path):
            frames = self.load_frames_from_video(video_path, fnum)
        else:
            raise ValueError(f"Invalid video path: {video_path}")
        
        vid_tube = []
        for fr in frames:
            fr = cv2.resize(fr, (224, 224))
            fr = np.expand_dims(self.normalize(fr), axis=(0, 1))
            vid_tube.append(fr) 
        
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube)
        
        return vid_tube
    
    def compute_similarity(self, video_path, text):
        with torch.no_grad():
            video_input = self.frames_preprocessing(video_path).float().to(self.device)
            video_features = self.model.vision_model.get_vid_features(video_input).float()
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            
            text_input = text_encoder.tokenize([text], truncate=True).to(self.device)
            text_features = self.model.text_model.encode_text(text_input).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = torch.dot(text_features[0], video_features[0]).item()
            
            return float(similarity)


def overall_semantic_consistency_single_video(evaluator, video_info, target_videos_path, use_frames=True):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    target_prompt = video_info.get('target_prompt', video_info.get('edit_prompt', video_info.get('prompt', '')))
    
    if not target_prompt:
        logger.warning(f"No target_prompt found for video {video_name}")
        target_prompt = "A video"  
    
    try:
        if use_frames:
            video_name_without_ext = os.path.splitext(video_name)[0]
            target_frame_folder = os.path.join(target_videos_path, video_name_without_ext)
            video_path = target_frame_folder
        else:
            video_path = os.path.join(target_videos_path, video_name)
        
        if not os.path.exists(video_path):
            error_msg = f'Path not found: {video_path}'
            logger.warning(error_msg)
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'category': str(video_info['category']),
                'subcategory': str(video_info['subcategory']),
                'error': error_msg
            }
        
        similarity = evaluator.compute_similarity(video_path, target_prompt)
        
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': float(similarity),
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


def overall_semantic_consistency_evaluation(video_info_list, target_videos_path, model_path, device="cuda", use_frames=True):
    scores = []
    video_results = []
    
    try:
        evaluator = VideoCLIPEvaluator(model_path, device)
    except Exception as e:
        error_msg = f"Failed to initialize VideoCLIP evaluator: {e}"
        logger.error(error_msg)
        # Return -1 for all videos if evaluator fails to initialize
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
    
    logger.info(f"Processing {len(video_info_list)} videos for overall semantic consistency evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating overall semantic consistency"):
        result = overall_semantic_consistency_single_video(evaluator, video_info, target_videos_path, use_frames)
        video_results.append(result)
        
        if 'error' not in result:
            scores.append(result['video_results'])
            logger.debug(f"Video {result['video_name']}: semantic consistency score = {result['video_results']:.4f}")
        else:
            logger.warning(f"Video {result['video_name']}: {result['error']}")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        logger.info(f"Overall semantic consistency score: {avg_score:.4f} (based on {len(scores)}/{len(video_info_list)} valid videos)")
    else:
        avg_score = -1.0
        logger.error("No valid overall semantic consistency scores calculated")
    
    return float(avg_score), video_results


def compute_overall_semantic_consistency(json_dir, device, source_videos_path=None, target_videos_path=None, 
                                       model_path=None, use_frames=True, path_yml='path.yml', **kwargs):
    """
    Compute overall semantic consistency metric using VideoCLIP-XL
    
    Args:
        json_dir: Path to JSON file with video information
        device: Device to run evaluation on ('cuda' or 'cpu')
        source_videos_path: Path to source videos (not used in this metric)
        target_videos_path: Path to target videos
        model_path: Path to VideoCLIP-XL model (if None, will load from path.yml)
        use_frames: Whether to use frames or video files
        path_yml: Path to the YAML file containing model paths
        **kwargs: Additional arguments
    
    Returns:
        tuple: (overall_score, video_results)
    """
    try:
        if not VIDEOCLIP_AVAILABLE:
            error_msg = "VideoCLIP-XL modules not available. Please ensure modeling and utils modules are in the Python path."
            logger.error(error_msg)
            return -1.0, []
        
        # Load model path from path.yml if not provided
        if model_path is None:
            logger.info(f"Loading model path from {path_yml}")
            model_path = load_metric_paths(path_yml, 'overall_semantic_consistency')
            
            if model_path is None:
                error_msg = "Model path must be provided either as argument or in path.yml"
                logger.error(error_msg)
                video_info_list = load_video_info(json_dir, 'overall_semantic_consistency')
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
        
        video_info_list = load_video_info(json_dir, 'overall_semantic_consistency')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for overall semantic consistency evaluation")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = overall_semantic_consistency_evaluation(
            video_info_list, target_videos_path, model_path, device, use_frames
        )
        
        if overall_score == -1.0:
            logger.error("Overall semantic consistency evaluation failed.")
        else:
            logger.info(f"Overall semantic consistency evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        error_msg = f"Error in compute_overall_semantic_consistency: {str(e)}"
        logger.error(error_msg)
        return -1.0, []