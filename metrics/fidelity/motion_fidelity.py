import os
import logging
import yaml
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
from ivebench_utils import load_video_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from fidelity.cotracker.predictor import CoTrackerPredictor
    COTRACKER_AVAILABLE = True
except ImportError:
    logger.warning("CoTracker not available. Please install cotracker package.")
    COTRACKER_AVAILABLE = False


def load_metric_paths(path_yml='path.yml', metric_name='motion_fidelity'):
    """Load checkpoint path from path.yml"""
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


class MotionFidelityEvaluator:
    
    def __init__(self, checkpoint_path, device="cuda", grid_size=10, max_frames=None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.grid_size = grid_size
        self.max_frames = max_frames
        self.model = None
        
        if not COTRACKER_AVAILABLE:
            error_msg = "CoTracker not available. Please install cotracker package."
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info("Loading CoTracker model...")
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                logger.info(f"Loading CoTracker from checkpoint: {self.checkpoint_path}")
                window_len = 60  # offline model
                self.model = CoTrackerPredictor(
                    checkpoint=self.checkpoint_path,
                    v2=False,
                    offline=True,
                    window_len=window_len,
                )
            else:
                logger.info("Loading default CoTracker model from torch hub...")
                self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
            
            self.model = self.model.to(self.device)
            logger.info("CoTracker model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load CoTracker model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def read_frames_from_folder(self, folder_path, image_extensions=None):
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_files = []
        for ext in image_extensions:
            pattern = str(folder_path / f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = str(folder_path / f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")
        
        image_files.sort()
        
        if self.max_frames is not None:
            image_files = image_files[:self.max_frames]
        
        logger.debug(f"Reading {len(image_files)} frames from {folder_path}")
        
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            raise ValueError(f"Cannot read image: {image_files[0]}")
        
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        height, width = first_frame.shape[:2]
        
        frames = np.zeros((len(image_files), height, width, 3), dtype=np.uint8)
        frames[0] = first_frame
        
        for i, image_file in enumerate(image_files[1:], 1):
            frame = cv2.imread(image_file)
            if frame is None:
                logger.warning(f"Cannot read image {image_file}, skipping")
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame.shape[:2] != (height, width):
                logger.debug(f"Resizing inconsistent frame {image_file}")
                frame = cv2.resize(frame, (width, height))
            
            frames[i] = frame
        
        return frames
    
    def read_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
            
            if self.max_frames is not None and frame_count >= self.max_frames:
                break
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        return np.array(frames)
    
    def load_video_data(self, video_path):
        video_path = Path(video_path)
        
        if video_path.is_file():
            frames = self.read_video_file(str(video_path))
        elif video_path.is_dir():
            frames = self.read_frames_from_folder(video_path)
        else:
            raise ValueError(f"Invalid path: {video_path}")
        
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
        return video_tensor
    
    def interpolate_track(self, track, visibility, target_length):
        valid_indices = np.where(visibility > 0.5)[0]
        
        if len(valid_indices) < 2:
            return np.zeros((target_length, 2)), np.zeros(target_length)
        
        valid_track = track[valid_indices]
        
        original_indices = np.linspace(0, 1, len(valid_indices))
        target_indices = np.linspace(0, 1, target_length)
        
        interp_x = interp1d(original_indices, valid_track[:, 0], kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(original_indices, valid_track[:, 1], kind='linear',
                           bounds_error=False, fill_value='extrapolate')
        
        interpolated_track = np.column_stack([interp_x(target_indices), interp_y(target_indices)])
        
        interp_vis = interp1d(original_indices, np.ones(len(valid_indices)), kind='linear',
                             bounds_error=False, fill_value=0.5)
        interpolated_visibility = interp_vis(target_indices)
        
        return interpolated_track, interpolated_visibility
    
    def compute_frame_by_frame_similarity(self, track1, track2, vis1, vis2):
        T = len(track1)
        
        position_distances = np.linalg.norm(track1 - track2, axis=1)
        
        if T > 1:
            velocity1 = np.diff(track1, axis=0)
            velocity2 = np.diff(track2, axis=0)
            velocity_distances = np.linalg.norm(velocity1 - velocity2, axis=1)
            velocity_distances = np.concatenate([[velocity_distances[0]], velocity_distances])
        else:
            velocity_distances = np.zeros(T)
        
        visibility_weights = np.minimum(vis1, vis2)
        
        track1_span = np.max(track1, axis=0) - np.min(track1, axis=0)
        track2_span = np.max(track2, axis=0) - np.min(track2, axis=0)
        normalization_factor = np.mean([np.linalg.norm(track1_span), np.linalg.norm(track2_span)])
        
        if normalization_factor < 1e-6:
            normalization_factor = 1.0
        
        position_distances = position_distances / normalization_factor
        velocity_distances = velocity_distances / normalization_factor
        
        position_similarities = 1.0 / (1.0 + position_distances)
        velocity_similarities = 1.0 / (1.0 + velocity_distances)
        
        frame_similarities = (0.7 * position_similarities + 0.3 * velocity_similarities)
        
        weighted_similarities = frame_similarities * visibility_weights
        
        if np.sum(visibility_weights) > 0:
            overall_similarity = np.sum(weighted_similarities) / np.sum(visibility_weights)
        else:
            overall_similarity = 0.0
        
        return overall_similarity
    
    def synchronize_videos(self, tracks1, visibility1, tracks2, visibility2):
        T1, N1 = tracks1.shape[:2]
        T2, N2 = tracks2.shape[:2]
        
        target_length = min(T1, T2)
        
        synced_tracks1 = np.zeros((target_length, N1, 2))
        synced_vis1 = np.zeros((target_length, N1))
        synced_tracks2 = np.zeros((target_length, N2, 2))
        synced_vis2 = np.zeros((target_length, N2))
        
        for i in range(N1):
            synced_tracks1[:, i, :], synced_vis1[:, i] = self.interpolate_track(
                tracks1[:, i, :], visibility1[:, i], target_length)
        
        for i in range(N2):
            synced_tracks2[:, i, :], synced_vis2[:, i] = self.interpolate_track(
                tracks2[:, i, :], visibility2[:, i], target_length)
        
        return synced_tracks1, synced_vis1, synced_tracks2, synced_vis2
    
    def compute_motion_similarity(self, source_video_path, target_video_path):
        if self.model is None:
            raise RuntimeError("CoTracker model not loaded")
        
        video1 = self.load_video_data(source_video_path).to(self.device)
        video2 = self.load_video_data(target_video_path).to(self.device)
        
        with torch.no_grad():
            pred_tracks1, pred_visibility1 = self.model(
                video1,
                grid_size=self.grid_size,
                grid_query_frame=0,
                backward_tracking=False,
            )
            
            pred_tracks2, pred_visibility2 = self.model(
                video2,
                grid_size=self.grid_size,
                grid_query_frame=0,
                backward_tracking=False,
            )
        
        similarity_score = self._compute_similarity_from_tracks(
            pred_tracks1, pred_visibility1, pred_tracks2, pred_visibility2)
        
        return float(similarity_score)
    
    def _compute_similarity_from_tracks(self, tracks1, visibility1, tracks2, visibility2):
        tracks1 = tracks1.squeeze(0).cpu().numpy()
        tracks2 = tracks2.squeeze(0).cpu().numpy()
        visibility1 = visibility1.squeeze(0).cpu().numpy()
        visibility2 = visibility2.squeeze(0).cpu().numpy()
        
        tracks1, visibility1, tracks2, visibility2 = self.synchronize_videos(
            tracks1, visibility1, tracks2, visibility2)
        
        min_track_length = 5
        min_visibility = 0.3
        
        valid_indices1 = []
        valid_indices2 = []
        
        for i in range(tracks1.shape[1]):
            avg_vis = np.mean(visibility1[:, i])
            valid_frames = np.sum(visibility1[:, i] > 0.5)
            if avg_vis > min_visibility and valid_frames >= min_track_length:
                valid_indices1.append(i)
        
        for i in range(tracks2.shape[1]):
            avg_vis = np.mean(visibility2[:, i])
            valid_frames = np.sum(visibility2[:, i] > 0.5)
            if avg_vis > min_visibility and valid_frames >= min_track_length:
                valid_indices2.append(i)
        
        if len(valid_indices1) == 0 or len(valid_indices2) == 0:
            return 0.0
        
        similarity_matrix = np.zeros((len(valid_indices1), len(valid_indices2)))
        
        for i, idx1 in enumerate(valid_indices1):
            for j, idx2 in enumerate(valid_indices2):
                track1 = tracks1[:, idx1, :]
                track2 = tracks2[:, idx2, :]
                vis1 = visibility1[:, idx1]
                vis2 = visibility2[:, idx2]
                
                similarity = self.compute_frame_by_frame_similarity(track1, track2, vis1, vis2)
                similarity_matrix[i, j] = similarity
        
        row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
        
        similarity_threshold = 0.3
        valid_similarities = []
        
        for i, j in zip(row_indices, col_indices):
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                valid_similarities.append(similarity)
        
        if valid_similarities:
            return np.mean(valid_similarities)
        else:
            return 0.0


def motion_fidelity_single_video(evaluator, video_info, source_videos_path, target_videos_path, use_frames=True):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    category = str(video_info.get("category", ""))
    subcategory = str(video_info.get("subcategory", ""))

    try:
        if category in ["subject_motion_editing", "camera_motion_editing"] or subcategory == "event effect":
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'category': category,
                'subcategory': subcategory
            }

        if use_frames:
            video_name_without_ext = os.path.splitext(video_name)[0]
            source_video_path = os.path.join(source_videos_path, video_name_without_ext)
            target_video_path = os.path.join(target_videos_path, video_name_without_ext)
        else:
            source_video_path = os.path.join(source_videos_path, video_name)
            target_video_path = os.path.join(target_videos_path, video_name)
        
        if not os.path.exists(source_video_path):
            error_msg = f'Source path not found: {source_video_path}'
            logger.warning(error_msg)
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'category': category,
                'subcategory': subcategory,
                'error': error_msg
            }
        
        if not os.path.exists(target_video_path):
            error_msg = f'Target path not found: {target_video_path}'
            logger.warning(error_msg)
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'category': category,
                'subcategory': subcategory,
                'error': error_msg
            }
        
        similarity = evaluator.compute_motion_similarity(source_video_path, target_video_path)
        
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': float(similarity),
            'category': category,
            'subcategory': subcategory
        }
        
    except Exception as e:
        error_msg = f"Error processing video {video_name}: {str(e)}"
        logger.error(error_msg)
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': -1.0,
            'category': category,
            'subcategory': subcategory,
            'error': error_msg
        }


def motion_fidelity_evaluation(video_info_list, source_videos_path, target_videos_path, 
                             checkpoint_path, device="cuda", use_frames=True, grid_size=10, max_frames=None):
    scores = []
    video_results = []
    
    try:
        evaluator = MotionFidelityEvaluator(checkpoint_path, device, grid_size, max_frames)
    except Exception as e:
        error_msg = f"Failed to initialize motion fidelity evaluator: {e}"
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
    
    logger.info(f"Processing {len(video_info_list)} videos for motion fidelity evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating motion fidelity"):
        result = motion_fidelity_single_video(evaluator, video_info, source_videos_path, target_videos_path, use_frames)
        video_results.append(result)
        
        if 'error' not in result and result['video_results'] != -1.0:
            scores.append(result['video_results'])
            logger.debug(f"Video {result['video_name']}: motion fidelity score = {result['video_results']:.4f}")
        else:
            if 'error' in result:
                logger.warning(f"Video {result['video_name']}: {result['error']}")
            else:
                logger.warning(f"Video {result['video_name']}: skipped (category/subcategory exclusion or processing failed)")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        logger.info(f"Overall motion fidelity score: {avg_score:.4f} (based on {len(scores)}/{len(video_info_list)} valid videos)")
    else:
        avg_score = -1.0
        logger.error("No valid motion fidelity scores calculated")
    
    return float(avg_score), video_results

def compute_motion_fidelity(json_dir, device, source_videos_path=None, target_videos_path=None, 
                          checkpoint_path=None, use_frames=True, grid_size=10, max_frames=None, 
                          path_yml='path.yml', **kwargs):
    """
    Compute motion fidelity metric using CoTracker
    
    Args:
        json_dir: Path to JSON file with video information
        device: Device to run evaluation on ('cuda' or 'cpu')
        source_videos_path: Path to source videos
        target_videos_path: Path to target videos
        checkpoint_path: Path to CoTracker checkpoint (if None, will load from path.yml)
        use_frames: Whether to use frames or video files
        grid_size: Grid size for CoTracker
        max_frames: Maximum number of frames to process
        path_yml: Path to the YAML file containing model paths
        **kwargs: Additional arguments
    
    Returns:
        tuple: (overall_score, video_results)
    """
    try:
        if not COTRACKER_AVAILABLE:
            error_msg = "CoTracker not available. Please install cotracker package."
            logger.error(error_msg)
            return -1.0, []

        if checkpoint_path is None:
            logger.info(f"Loading checkpoint path from {path_yml}")
            checkpoint_path = load_metric_paths(path_yml, 'motion_fidelity')
            
            if checkpoint_path is None:
                error_msg = "Checkpoint path must be provided either as argument or in path.yml"
                logger.error(error_msg)
                video_info_list = load_video_info(json_dir, 'motion_fidelity')
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
        
        video_info_list = load_video_info(json_dir, 'motion_fidelity')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if source_videos_path is None:
            raise ValueError("source_videos_path is required for motion fidelity evaluation")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for motion fidelity evaluation")
        
        if not os.path.exists(source_videos_path):
            raise FileNotFoundError(f"Source videos path not found: {source_videos_path}")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = motion_fidelity_evaluation(
            video_info_list, source_videos_path, target_videos_path, 
            checkpoint_path, device, use_frames, grid_size, max_frames
        )
        
        if overall_score == -1.0:
            logger.error("Motion fidelity evaluation failed.")
        else:
            logger.info(f"Motion fidelity evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        error_msg = f"Error in compute_motion_fidelity: {str(e)}"
        logger.error(error_msg)
        return -1.0, []