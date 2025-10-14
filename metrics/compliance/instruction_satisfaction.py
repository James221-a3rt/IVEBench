# compliance/instruction_satisfaction.py
import os
import tempfile
import subprocess
import glob
import gc
import re
import shutil
import logging
import yaml
import cv2
import torch
from tqdm import tqdm
from ivebench_utils import load_video_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metric_paths(path_yml='path.yml', metric_name='instruction_satisfaction'):
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


class QwenVLEvaluator:   
    def __init__(self, model_path, device="auto"):
        self.model_path = model_path
        self.device = device
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from compliance.qwen_vl_utils import process_vision_info
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            logger.info(f"Loading Qwen2.5-VL model from {self.model_path}")
            
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all GPUs")
            logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}")

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype="auto", 
                device_map="auto"
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.process_vision_info = process_vision_info
            
            logger.info("Qwen2.5-VL model loaded successfully")
            
            if hasattr(self.model, 'hf_device_map'):
                logger.info(f"Model device map: {self.model.hf_device_map}")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("Please install transformers and qwen_vl_utils packages")
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL model: {e}")
            raise

    def release_model(self):
        logger.info("Releasing model resources...")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if hasattr(self, 'process_vision_info'):
            del self.process_vision_info
        
        gc.collect()
        torch.cuda.empty_cache()

    def frames_to_video(self, frames_dir, output_path, fps=25):
        exts = [".jpg", ".png"]
        used_ext = None
        for ext in exts:
            if glob.glob(os.path.join(frames_dir, f"*{ext}")):
                used_ext = ext
                break
        if used_ext is None:
            raise ValueError(f"can not find jpg/png files in {frames_dir}")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, f"%05d{ext}"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def compress_video(self, input_path, output_path, target_size_mb=1, max_frames=20, max_side=426, output_fps=5):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 1
        cap.release()

        sample_fps = min(max_frames / duration, fps)

        scale_factor = min(max_side / max(width, height), 1.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        new_width -= new_width % 2
        new_height -= new_height % 2

        final_frame_count = min(max_frames, int(frame_count * (sample_fps / fps)))
        target_bitrate = (target_size_mb * 8 * 1024 * 1024) // (duration * max(1, final_frame_count / max_frames))

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vf", f"scale={new_width}:{new_height},fps={sample_fps}",
            "-r", str(output_fps),
            "-c:v", "libx264",
            "-preset", "fast",
            "-b:v", str(target_bitrate),
            "-maxrate", str(target_bitrate),
            "-bufsize", str(target_bitrate),
            "-an",
            output_path,
        ]

        subprocess.run(cmd, check=True)
        return output_path

    def process_video_frames(self, frames_dir, temp_dir):
        tmp_video = os.path.join(temp_dir, "tmp.mp4")
        compressed_video = os.path.join(temp_dir, "compressed.mp4")
        
        self.frames_to_video(frames_dir, tmp_video, fps=25)
        
        self.compress_video(tmp_video, compressed_video, target_size_mb=1, max_frames=20, max_side=426)
        
        return compressed_video

    def evaluate_video(self, source_frames_dir, target_frames_dir, edit_prompt):
        temp_dir = None
        
        try:
            temp_dir = tempfile.mkdtemp()

            source_video_path = self.process_video_frames(source_frames_dir, temp_dir)
            os.rename(source_video_path, os.path.join(temp_dir, "source.mp4"))
            source_video_path = os.path.join(temp_dir, "source.mp4")

            target_video_path = self.process_video_frames(target_frames_dir, temp_dir)
            os.rename(target_video_path, os.path.join(temp_dir, "target.mp4"))
            target_video_path = os.path.join(temp_dir, "target.mp4")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": source_video_path},
                        {"type": "text", "text": "The video above is the first video."},
                        {"type": "video", "video": target_video_path},
                        {"type": "text", "text": f"Given that the first video is the source video (original video) and the second video is the target video (edited video), the edit prompt is '{edit_prompt}'. Does the target video match the expected result of the source video after applying the edit prompt? Please provide a rating from 1 to 5, where higher values mean a better match. 1 means completely unrelated, 2 means possibly matches, 3 means somewhat matches, 4 means mostly matches, and 5 means perfectly matches. Respond in the format: [score number] [explanation]. Example: [1] [XXX]"},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=1280)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            score = self._parse_score(output_text[0] if output_text else "")

            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

            return score, output_text[0] if output_text else ""
            
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Could not delete temp directory {temp_dir}: {e}")

    def _parse_score(self, output_text):
        patterns = [
            r'\[([1-5])\]', 
            r'([1-5])(?:\s*分|\s*\/5|\s*out\s*of\s*5)', 
            r'评分\s*[：:]\s*([1-5])', 
            r'给出\s*([1-5])', 
            r'(\d+(?:\.\d+)?)\s*[\/分]',  
            r'([1-5])', 
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output_text)
            if matches:
                try:
                    score = float(matches[0])
                    if 1 <= score <= 5:
                        return score
                except ValueError:
                    continue
        
        logger.warning(f"Could not parse score from output: {output_text}")
        return -1.0  # Changed from 2.5 to -1.0


def instruction_satisfaction_single_video(evaluator, video_info, source_videos_path, target_videos_path):
    video_name = video_info['src_video_name']
    video_id = video_info['id']
    edit_prompt = video_info.get('edit_prompt', video_info.get('prompt', ''))
    
    if not edit_prompt:
        logger.warning(f"No edit_prompt found for video {video_name}")
        edit_prompt = "Edit this video" 
    
    try:
        video_name_without_ext = os.path.splitext(video_name)[0]
        source_frame_folder = os.path.join(source_videos_path, video_name_without_ext)
        target_frame_folder = os.path.join(target_videos_path, video_name_without_ext)
        
        if not os.path.exists(source_frame_folder):
            error_msg = f"Source frame folder not found: {source_frame_folder}"
            logger.warning(error_msg)
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'edit_prompt': str(edit_prompt),
                'category': str(video_info.get('category', '')),
                'subcategory': str(video_info.get('subcategory', '')),
                'error': error_msg
            }
        
        if not os.path.exists(target_frame_folder):
            error_msg = f"Target frame folder not found: {target_frame_folder}"
            logger.warning(error_msg)
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'edit_prompt': str(edit_prompt),
                'category': str(video_info.get('category', '')),
                'subcategory': str(video_info.get('subcategory', '')),
                'error': error_msg
            }
        
        score, model_output = evaluator.evaluate_video(
            source_frame_folder, target_frame_folder, edit_prompt
        )
        
        if score == -1.0:
            return {
                'video_id': int(video_id),
                'video_name': str(video_name),
                'video_results': -1.0,
                'compliance_output': str(model_output),
                'edit_prompt': str(edit_prompt),
                'category': str(video_info.get('category', '')),
                'subcategory': str(video_info.get('subcategory', '')),
                'error': 'Failed to parse score from model output'
            }
        
        cleaned_output = model_output.replace('\n', ' ').replace('\r', ' ').strip()
        logger.info(f"Video {video_name}: instruction satisfaction score = {score:.4f}")
        logger.debug(f"Model output: {cleaned_output}")
        
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': float(score),
            'compliance_output': str(cleaned_output),
            'edit_prompt': str(edit_prompt),
            'category': str(video_info.get('category', '')),
            'subcategory': str(video_info.get('subcategory', ''))
        }
        
    except Exception as e:
        error_msg = f"Error processing video {video_name}: {str(e)}"
        logger.error(error_msg)
        return {
            'video_id': int(video_id),
            'video_name': str(video_name),
            'video_results': -1.0,
            'edit_prompt': str(edit_prompt),
            'category': str(video_info.get('category', '')),
            'subcategory': str(video_info.get('subcategory', '')),
            'error': error_msg
        }


def instruction_satisfaction_evaluation(video_info_list, source_videos_path, target_videos_path, model_path, device="auto"):
    scores = []
    video_results = []
    evaluator = None
    
    try:
        evaluator = QwenVLEvaluator(model_path, device)
    except Exception as e:
        error_msg = f"Failed to initialize Qwen-VL evaluator: {e}"
        logger.error(error_msg)
        for video_info in video_info_list:
            video_results.append({
                'video_id': int(video_info['id']),
                'video_name': str(video_info['src_video_name']),
                'video_results': -1.0,
                'edit_prompt': str(video_info.get('edit_prompt', video_info.get('prompt', ''))),
                'category': str(video_info.get('category', '')),
                'subcategory': str(video_info.get('subcategory', '')),
                'error': error_msg
            })
        return -1.0, video_results
    
    try:
        logger.info(f"Processing {len(video_info_list)} videos for instruction satisfaction evaluation")
        
        for video_info in tqdm(video_info_list, desc="Evaluating instruction satisfaction"):
            result = instruction_satisfaction_single_video(
                evaluator, video_info, source_videos_path, target_videos_path
            )
            video_results.append(result)
            
            if 'error' not in result:
                scores.append(result['video_results'])
                logger.debug(f"Video {result['video_name']}: instruction satisfaction score = {result['video_results']:.4f}")
            else:
                logger.warning(f"Video {result['video_name']}: {result['error']}")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info(f"Overall instruction satisfaction score: {avg_score:.4f} (based on {len(scores)}/{len(video_info_list)} valid videos)")
        else:
            avg_score = -1.0
            logger.error("No valid instruction satisfaction scores calculated")
        
        return float(avg_score), video_results
        
    finally:
        if evaluator is not None:
            evaluator.release_model()


def compute_instruction_satisfaction(json_dir, device, source_videos_path=None, target_videos_path=None, 
                                   model_path=None, path_yml='path.yml', **kwargs):

    try:
        # Load model path from path.yml if not provided
        if model_path is None:
            logger.info(f"Loading model path from {path_yml}")
            model_path = load_metric_paths(path_yml, 'instruction_satisfaction')
            
            if model_path is None:
                error_msg = "Model path must be provided either as argument or in path.yml"
                logger.error(error_msg)
                video_info_list = load_video_info(json_dir, 'instruction_satisfaction')
                video_results = []
                for video_info in video_info_list:
                    video_results.append({
                        'video_id': int(video_info['id']),
                        'video_name': str(video_info['src_video_name']),
                        'video_results': -1.0,
                        'edit_prompt': str(video_info.get('edit_prompt', video_info.get('prompt', ''))),
                        'category': str(video_info.get('category', '')),
                        'subcategory': str(video_info.get('subcategory', '')),
                        'error': error_msg
                    })
                return -1.0, video_results
        
        video_info_list = load_video_info(json_dir, 'instruction_satisfaction')
        logger.info(f"Loaded {len(video_info_list)} video entries")
        
        if source_videos_path is None:
            raise ValueError("source_videos_path is required for instruction satisfaction evaluation")
        if target_videos_path is None:
            raise ValueError("target_videos_path is required for instruction satisfaction evaluation")
        
        if not os.path.exists(source_videos_path):
            raise FileNotFoundError(f"Source videos path not found: {source_videos_path}")
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        
        overall_score, video_results = instruction_satisfaction_evaluation(
            video_info_list, source_videos_path, target_videos_path, model_path, device
        )
        
        if overall_score == -1.0:
            logger.error("Instruction satisfaction evaluation failed.")
        else:
            logger.info(f"Instruction satisfaction evaluation completed. Overall score: {overall_score:.4f}")
        
        return overall_score, video_results
        
    except Exception as e:
        error_msg = f"Error in compute_instruction_satisfaction: {str(e)}"
        logger.error(error_msg)
        return -1.0, []