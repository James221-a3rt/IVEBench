import os
import re
import logging
import yaml
import torch
import numpy as np
from PIL import Image
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
from ivebench_utils import load_video_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import compliance.groundingdino.datasets.transforms as T
    from compliance.groundingdino.models import build_model
    from compliance.groundingdino.util.slconfig import SLConfig
    from compliance.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from compliance.groundingdino.util.vl_utils import create_positive_map_from_span
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    logger.warning("GroundingDINO not available. Please install groundingdino package.")
    GROUNDING_DINO_AVAILABLE = False

temp_dir = "./tmp/quantity_accuracy_frames"


def load_metric_paths(path_yml='path.yml', metric_name='quantity_accuracy'):
    """Load config and checkpoint paths from path.yml"""
    try:
        if not os.path.exists(path_yml):
            logger.warning(f"Path configuration file not found: {path_yml}")
            return None, None
        
        with open(path_yml, 'r', encoding='utf-8') as f:
            paths_config = yaml.safe_load(f)
        
        if metric_name not in paths_config:
            logger.warning(f"Metric '{metric_name}' not found in {path_yml}")
            return None, None
        
        metric_config = paths_config[metric_name]
        config_file = metric_config.get('config')
        checkpoint_path = metric_config.get('checkpoint')
        
        logger.info(f"Loaded paths for {metric_name}: config={config_file}, checkpoint={checkpoint_path}")
        
        return config_file, checkpoint_path
        
    except Exception as e:
        logger.error(f"Error loading metric paths from {path_yml}: {e}")
        return None, None


class QuantityAccuracyEvaluator:
    def __init__(self, config_file, checkpoint_path, device="cuda", box_threshold=0.3, text_threshold=0.25):
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        if not GROUNDING_DINO_AVAILABLE:
            error_msg = "GroundingDINO not available. Please install groundingdino package."
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info("Loading GroundingDINO model...")
            
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            
            args = SLConfig.fromfile(self.config_file)
            args.device = self.device
            self.model = build_model(args)
            
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            load_res = self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            logger.debug(f"Model load result: {load_res}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("GroundingDINO model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load GroundingDINO model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def parse_edit_prompt(self, edit_prompt):
        if not edit_prompt:
            return None, None
        
        patterns = [
            r"increase\s+the\s+number\s+of\s+([\w\s]+?)\s+to\s+(\d+)",
            r"decrease\s+the\s+number\s+of\s+([\w\s]+?)\s+to\s+(\d+)",
            r"change\s+the\s+number\s+of\s+([\w\s]+?)\s+to\s+(\d+)",
            r"set\s+the\s+number\s+of\s+([\w\s]+?)\s+to\s+(\d+)",
            r"make\s+(\d+)\s+([\w\s]+?)",
            r"add\s+([\w\s]+?)\s+to\s+(\d+)",
            r"remove\s+([\w\s]+?)\s+to\s+(\d+)",
        ]
        
        patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        edit_prompt_lower = edit_prompt.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, edit_prompt_lower)
            if match:
                object_name = match.group(1)
                target_count = int(match.group(2))
                
                if object_name.endswith('s') and target_count == 1:
                    if object_name.endswith('ies'):
                        object_name = object_name[:-3] + 'y'
                    elif object_name.endswith('es'):
                        object_name = object_name[:-2]
                    else:
                        object_name = object_name[:-1]
                elif not object_name.endswith('s') and target_count > 1:
                    if object_name.endswith('y'):
                        object_name = object_name[:-1] + 'ies'
                    elif object_name.endswith(('s', 'sh', 'ch', 'x', 'z')):
                        object_name = object_name + 'es'
                    else:
                        object_name = object_name + 's'
                
                return target_count, object_name
        
        logger.warning(f"Could not parse edit prompt: {edit_prompt}")
        return None, None
    
    def load_image(self, image_path):
        try:
            image_pil = Image.open(image_path).convert("RGB")
            
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            image, _ = transform(image_pil, None)
            return image_pil, image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None, None
    
    def get_grounding_output(self, image, caption):
        if self.model is None:
            raise RuntimeError("GroundingDINO model not loaded")
        
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        
        logits = outputs["pred_logits"].sigmoid()[0] 
        boxes = outputs["pred_boxes"][0]  
        
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)
        
        pred_phrases = []
        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
        
        return boxes_filt, pred_phrases
    
    def count_objects_in_frames(self, video_path, object_name, sample_frames=5):
        if self.model is None:
            raise RuntimeError("GroundingDINO model not loaded")
        
        frames = self._get_video_frames(video_path, sample_frames)
        if not frames:
            return 0.0, []
        
        frame_counts = []
        
        for frame_path in frames:
            image_pil, image = self.load_image(frame_path)
            if image is None:
                continue
            
            detection_text = f"a {object_name}"
            
            boxes, phrases = self.get_grounding_output(image, detection_text)
            
            count = len([phrase for phrase in phrases if object_name.lower() in phrase.lower()])
            frame_counts.append(count)
            
            logger.debug(f"Frame {frame_path}: detected {count} {object_name}(s)")
        
        if frame_counts:
            average_count = np.mean(frame_counts)
        else:
            average_count = 0.0
        
        return float(average_count), frame_counts
    
    def _get_video_frames(self, video_path, sample_frames=5):
        video_path = Path(video_path)
        
        if video_path.is_dir():
            image_files = []
            for ext in self.image_extensions:
                pattern = str(video_path / f"*{ext}")
                image_files.extend(glob.glob(pattern))
                pattern = str(video_path / f"*{ext.upper()}")
                image_files.extend(glob.glob(pattern))
            
            image_files.sort()
            
            if len(image_files) == 0:
                logger.warning(f"No image files found in {video_path}")
                return []
            
            if len(image_files) <= sample_frames:
                return image_files
            else:
                step = len(image_files) // sample_frames
                return image_files[::step][:sample_frames]
                
        elif video_path.is_file():
            return self._extract_frames_from_video(str(video_path), sample_frames)
        else:
            logger.error(f"Invalid video path: {video_path}")
            return []
    
    def _extract_frames_from_video(self, video_path, sample_frames=5):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return []

            step = max(1, total_frames // sample_frames)
            frame_indices = [i * step for i in range(sample_frames)]
            
            os.makedirs(temp_dir, exist_ok=True)
            
            extracted_frames = []
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    extracted_frames.append(frame_path)
            
            cap.release()
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return []
    
    def compute_quantity_accuracy(self, video_path, edit_prompt, tolerance=0.5):
        target_count, object_name = self.parse_edit_prompt(edit_prompt)
        
        if target_count is None or object_name is None:
            logger.warning(f"Cannot parse edit prompt: {edit_prompt}")
            return 0
        
        detected_count, frame_counts = self.count_objects_in_frames(video_path, object_name)
        
        error = abs(detected_count - target_count)
        is_correct = error <= tolerance
        
        score = 1 if is_correct else 0
        
        logger.debug(f"Target: {target_count} {object_name}, Detected: {detected_count:.1f}, "
                    f"Error: {error:.1f}, Correct: {is_correct}, Score: {score}")
        
        return score


def is_quantity_editing_task(video_info):
    category = video_info.get('category', '').strip().lower()
    return category == "quantity_modification"


def quantity_accuracy_single_video(evaluator, video_info, target_videos_path, use_frames=True):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    category = video_info.get('category', '')
    subcategory = video_info.get('subcategory', '')
    edit_prompt = video_info.get('edit_prompt', video_info.get('target_prompt', ''))
    
    if not is_quantity_editing_task(video_info):
        logger.debug(f"Video {video_name} is not a quantity editing task, returning -1")
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': -1, 
            'edit_prompt': str(edit_prompt),
            'category': str(category),
            'subcategory': str(subcategory),
            'note': 'Not a quantity editing task'
        }
    
    if not edit_prompt:
        logger.warning(f"No edit_prompt found for quantity editing video {video_name}")
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': -2, 
            'edit_prompt': '',
            'category': str(category),
            'subcategory': str(subcategory),
            'error': 'No edit_prompt found for quantity editing task'
        }
    
    try:
        if use_frames:
            video_name_without_ext = os.path.splitext(video_name)[0]
            target_video_path = os.path.join(target_videos_path, video_name_without_ext)
        else:
            target_video_path = os.path.join(target_videos_path, video_name)
        
        if not os.path.exists(target_video_path):
            error_msg = f'Target path not found: {target_video_path}'
            logger.warning(error_msg)
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -3, 
                'edit_prompt': str(edit_prompt),
                'category': str(category),
                'subcategory': str(subcategory),
                'error': error_msg
            }
        
        accuracy = evaluator.compute_quantity_accuracy(target_video_path, edit_prompt)
        
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': int(accuracy), 
            'edit_prompt': str(edit_prompt),
            'category': str(category),
            'subcategory': str(subcategory)
        }
        
    except Exception as e:
        error_msg = f"Error processing video {video_name}: {str(e)}"
        logger.error(error_msg)
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': 0, 
            'edit_prompt': str(edit_prompt),
            'category': str(category),
            'subcategory': str(subcategory),
            'error': error_msg
        }


def quantity_accuracy_evaluation(video_info_list, target_videos_path, config_file, checkpoint_path, 
                               device="cuda", use_frames=True, box_threshold=0.3, text_threshold=0.25):
    scores = []
    video_results = []
    valid_task_count = 0  # 重命名，表示真正有效评估的任务数
    correct_count = 0
    
    try:
        evaluator = QuantityAccuracyEvaluator(config_file, checkpoint_path, device, box_threshold, text_threshold)
    except Exception as e:
        error_msg = f"Failed to initialize GroundingDINO evaluator: {e}"
        logger.error(error_msg)
        # Return results with errors for all quantity editing tasks
        for video_info in video_info_list:
            if is_quantity_editing_task(video_info):
                video_results.append({
                    'video_id': int(video_info['id']),
                    'video_name': str(video_info['src_video_name']),
                    'video_results': 0,
                    'edit_prompt': str(video_info.get('edit_prompt', video_info.get('target_prompt', ''))),
                    'category': str(video_info.get('category', '')),
                    'subcategory': str(video_info.get('subcategory', '')),
                    'error': error_msg
                })
            else:
                video_results.append({
                    'video_id': int(video_info['id']),
                    'video_name': str(video_info['src_video_name']),
                    'video_results': -1,
                    'edit_prompt': str(video_info.get('edit_prompt', video_info.get('target_prompt', ''))),
                    'category': str(video_info.get('category', '')),
                    'subcategory': str(video_info.get('subcategory', '')),
                    'note': 'Not a quantity editing task'
                })
        return 0.0, video_results
    
    logger.info(f"Processing {len(video_info_list)} videos for quantity accuracy evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating quantity accuracy"):
        result = quantity_accuracy_single_video(evaluator, video_info, target_videos_path, use_frames)
        video_results.append(result)
        
        # 只计算真正有效的评估结果（video_results为0或1，且没有error）
        if result['video_results'] in [0, 1] and 'error' not in result:
            valid_task_count += 1
            scores.append(result['video_results'])
            if result['video_results'] == 1:
                correct_count += 1
            logger.debug(f"Video {result['video_name']}: quantity accuracy = {result['video_results']}")
        elif result['video_results'] == -1:
            logger.debug(f"Video {result['video_name']}: not a quantity editing task")
        else:
            # video_results 为 -2, -3 或有error的情况
            if 'error' in result:
                logger.warning(f"Video {result['video_name']}: {result['error']}")
            else:
                logger.warning(f"Video {result['video_name']}: skipped (result code: {result['video_results']})")
    
    if valid_task_count > 0:
        accuracy_rate = correct_count / valid_task_count
        logger.info(f"Valid quantity editing tasks: {valid_task_count}, Correct: {correct_count}, "
                   f"Accuracy rate: {accuracy_rate:.4f}")
    else:
        accuracy_rate = 0.0
        logger.warning("No valid quantity editing task evaluations")
    
    return float(accuracy_rate), video_results


def compute_quantity_accuracy(json_dir, device, source_videos_path=None, target_videos_path=None, 
                            config_file=None, checkpoint_path=None, use_frames=True, 
                            box_threshold=0.3, text_threshold=0.25, path_yml='path.yml', **kwargs):
    """
    Compute quantity accuracy metric using GroundingDINO
    
    Args:
        json_dir: Path to JSON file with video information
        device: Device to run evaluation on ('cuda' or 'cpu')
        source_videos_path: Path to source videos (not used in this metric)
        target_videos_path: Path to target videos
        config_file: Path to GroundingDINO config file (if None, will load from path.yml)
        checkpoint_path: Path to GroundingDINO checkpoint (if None, will load from path.yml)
        use_frames: Whether to use frames or video files
        box_threshold: Box threshold for detection
        text_threshold: Text threshold for detection
        path_yml: Path to the YAML file containing model paths
        **kwargs: Additional arguments
    
    Returns:
        tuple: (accuracy_rate, video_results)
    """
    try:
        if not GROUNDING_DINO_AVAILABLE:
            error_msg = "GroundingDINO not available. Please install groundingdino package."
            logger.error(error_msg)
            return 0.0, []
        
        # Load config and checkpoint paths from path.yml if not provided
        if config_file is None or checkpoint_path is None:
            logger.info(f"Loading model paths from {path_yml}")
            yml_config, yml_checkpoint = load_metric_paths(path_yml, 'quantity_accuracy')
            
            if config_file is None:
                config_file = yml_config
            if checkpoint_path is None:
                checkpoint_path = yml_checkpoint
            
            if config_file is None or checkpoint_path is None:
                error_msg = "Config file and checkpoint path must be provided either as arguments or in path.yml"
                logger.error(error_msg)
                return 0.0, []
        
        video_info_list = load_video_info(json_dir, 'quantity_accuracy')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for quantity accuracy evaluation")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = quantity_accuracy_evaluation(
            video_info_list, target_videos_path, config_file, checkpoint_path,
            device, use_frames, box_threshold, text_threshold
        )
        
        logger.info(f"Quantity accuracy evaluation completed. Overall accuracy rate: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        error_msg = f"Error in compute_quantity_accuracy: {str(e)}"
        logger.error(error_msg)
        return 0.0, []