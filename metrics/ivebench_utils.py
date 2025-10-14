import os
import cv2
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_video_info(json_path, metric):
    video_info = load_json(json_path)
    video_list = []
    
    for item in video_info:
        video_list.append({
            'id': item['id'],
            'src_video_name': item['src_video_name'],
            'category': item['category'],
            'subcategory': item['subcategory'],
            'source_prompt': item['source_prompt'],
            'edit_prompt': item['edit_prompt'],
            'target_prompt': item['target_prompt']
        })
    
    return video_list

def load_frames_from_folder(frame_folder_path):
    if not os.path.exists(frame_folder_path):
        raise FileNotFoundError(f"Frame folder not found: {frame_folder_path}")
    
    frame_files = sorted([f for f in os.listdir(frame_folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not frame_files:
        raise ValueError(f"No image files found in {frame_folder_path}")
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder_path, frame_file)
        try:
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        except Exception as e:
            logger.warning(f"Could not load frame {frame_path}: {e}")
    
    if not frames:
        raise ValueError(f"No valid frames loaded from {frame_folder_path}")
    
    return frames

def get_frames_from_folder(frame_folder_path):
    if not os.path.exists(frame_folder_path):
        raise FileNotFoundError(f"Frame folder not found: {frame_folder_path}")
    
    frame_files = sorted([f for f in os.listdir(frame_folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not frame_files:
        raise ValueError(f"No image files found in {frame_folder_path}")
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
        else:
            logger.warning(f"Could not load frame: {frame_path}")
    
    if not frames:
        raise ValueError(f"No valid frames loaded from {frame_folder_path}")
    
    return frames

def get_frames_from_video(video_path):
    frames = []
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    while video.isOpened():
        success, frame = video.read()
        if success:
            frames.append(frame)
        else:
            break
    
    video.release()
    
    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    return frames

def dino_transform_Image(size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def load_dino_model(device):
    import torch
    
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    model.eval()
    model.to(device)
    
    return model