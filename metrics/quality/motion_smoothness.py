import os
import cv2
import glob
import torch
import numpy as np
import logging
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf
from ivebench_utils import load_video_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from quality.amt.utils.utils import (
        img2tensor, tensor2img,
        check_dim_and_resize
    )
    from quality.amt.utils.build_utils import build_from_cfg
    from quality.amt.utils.utils import InputPadder
    AMT_AVAILABLE = True
except ImportError as e:
    logger.error(f"AMT modules not available: {e}")
    AMT_AVAILABLE = False


def load_metric_paths(path_yml='path.yml', metric_name='motion_smoothness'):
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
        config_path = metric_config.get('config')
        checkpoint_path = metric_config.get('checkpoint')
        
        logger.info(f"Loaded paths for {metric_name}:")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        
        return config_path, checkpoint_path
        
    except Exception as e:
        logger.error(f"Error loading metric paths from {path_yml}: {e}")
        return None, None


class FrameProcess:
    def __init__(self):
        pass

    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != [], f"No frames extracted from {video_path}"
        return frame_list 

    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
                'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
                'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) 
                      if os.path.splitext(p)[1][1:] in exts])
        
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
        
        assert frame_list != [], f"No frames found in {img_folder}"
        return frame_list

    def extract_frame(self, frame_list, start_from=0):
        extract = []
        for i in range(start_from, len(frame_list), 2):
            extract.append(frame_list[i])
        return extract


class MotionSmoothness:
    def __init__(self, config=None, ckpt=None, device="cuda"):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.model = None
        self.initialization()
        
        if not AMT_AVAILABLE:
            error_msg = "AMT modules are not available. Cannot initialize motion smoothness evaluator."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if not config or not ckpt:
            error_msg = "Config and checkpoint paths are required for AMT model."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.load_model()

    def load_model(self):
        try:
            cfg_path = self.config
            ckpt_path = self.ckpt
            
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Config file not found: {cfg_path}")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            
            network_cfg = OmegaConf.load(cfg_path).network
            network_name = network_cfg.name
            logger.info(f'Loading [{network_name}] from [{ckpt_path}]...')
            self.model = build_from_cfg(network_cfg)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt['state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("AMT model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load AMT model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def initialization(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(0).total_memory
            logger.info("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            self.anchor_resolution = 8192*8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        if torch.cuda.is_available():
            self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        else:
            self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1)
        self.fp = FrameProcess()

    def motion_score(self, video_path):
        if self.model is None:
            raise RuntimeError("AMT model is not loaded. Cannot compute motion score.")
        
        iters = int(self.niters)
        
        if video_path.endswith('.mp4'):
            frames = self.fp.get_frames(video_path)
        elif os.path.isdir(video_path):
            frames = self.fp.get_frames_from_img_folder(video_path)
        else:
            raise NotImplementedError(f"Unsupported input type: {video_path}")
        
        frame_list = self.fp.extract_frame(frames, start_from=0)
        inputs = [img2tensor(frame).to(self.device) for frame in frame_list]
        
        assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"
        
        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        
        if scale < 1:
            logger.debug(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
        
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        for i in range(iters):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(out) for out in outputs]
        vfi_score = self.vfi_score(frames, outputs)
        norm = (255.0 - vfi_score) / 255.0
        return float(norm)

    def vfi_score(self, ori_frames, interpolate_frames):
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        scores = []
        for i in range(len(interpolate)):
            scores.append(self.get_diff(ori[i], interpolate[i]))
        return np.mean(np.array(scores))

    def get_diff(self, img1, img2):
        img = cv2.absdiff(img1, img2)
        return np.mean(img)


def motion_smoothness_single_video(motion_evaluator, video_info, target_videos_path, use_frames=True):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    
    try:
        if use_frames:
            video_name_without_ext = os.path.splitext(video_name)[0]
            target_frame_folder = os.path.join(target_videos_path, video_name_without_ext)
            video_path = target_frame_folder
        else:
            video_path = os.path.join(target_videos_path, video_name)
        
        if not os.path.exists(video_path):
            error_msg = f"Video path not found: {video_path}"
            logger.warning(error_msg)
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'category': str(video_info['category']),
                'subcategory': str(video_info['subcategory']),
                'error': error_msg
            }
        
        score = motion_evaluator.motion_score(video_path)
        
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


def motion_smoothness_evaluation(video_info_list, target_videos_path, config=None, ckpt=None, device="cuda", use_frames=True):
    scores = []
    video_results = []
    
    try:
        motion_evaluator = MotionSmoothness(config, ckpt, device)
    except Exception as e:
        error_msg = f"Failed to initialize motion smoothness evaluator: {e}"
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
    
    logger.info(f"Processing {len(video_info_list)} videos for motion smoothness evaluation")
    
    for video_info in tqdm(video_info_list, desc="Evaluating motion smoothness"):
        result = motion_smoothness_single_video(motion_evaluator, video_info, target_videos_path, use_frames)
        video_results.append(result)
        
        if 'error' not in result:
            scores.append(result['video_results'])
            logger.debug(f"Video {result['video_name']}: motion smoothness score = {result['video_results']:.4f}")
        else:
            logger.warning(f"Video {result['video_name']}: {result['error']}")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        logger.info(f"Overall motion smoothness score: {avg_score:.4f} (based on {len(scores)}/{len(video_info_list)} valid videos)")
    else:
        avg_score = -1.0
        logger.error("No valid motion smoothness scores calculated")
    
    return float(avg_score), video_results


def compute_motion_smoothness(json_dir, device, source_videos_path=None, target_videos_path=None, 
                            config=None, ckpt=None, use_frames=True, path_yml='path.yml', **kwargs):
    """
    Compute motion smoothness metric
    
    Args:
        json_dir: Path to JSON file with video information
        device: Device to run evaluation on ('cuda' or 'cpu')
        source_videos_path: Path to source videos (not used in this metric)
        target_videos_path: Path to target videos to evaluate
        config: Config file path (if None, will load from path.yml)
        ckpt: Checkpoint file path (if None, will load from path.yml)
        use_frames: Whether to use frames or video files
        path_yml: Path to the YAML file containing model paths
        **kwargs: Additional arguments
    
    Returns:
        tuple: (overall_score, video_results)
    """
    try:
        if config is None or ckpt is None:
            logger.info(f"Loading model paths from {path_yml}")
            loaded_config, loaded_ckpt = load_metric_paths(path_yml, 'motion_smoothness')
            
            if config is None:
                config = loaded_config
            if ckpt is None:
                ckpt = loaded_ckpt
        
        if config is None or ckpt is None:
            error_msg = "Config and checkpoint paths must be provided either as arguments or in path.yml"
            logger.error(error_msg)
            video_info_list = load_video_info(json_dir, 'motion_smoothness')
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
        
        video_info_list = load_video_info(json_dir, 'motion_smoothness')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for motion smoothness evaluation")
        
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = motion_smoothness_evaluation(
            video_info_list, target_videos_path, config, ckpt, device, use_frames
        )
        
        if overall_score == -1.0:
            logger.error("Motion smoothness evaluation failed.")
        else:
            logger.info(f"Motion smoothness evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        error_msg = f"Error in compute_motion_smoothness: {str(e)}"
        logger.error(error_msg)
        return -1.0, []