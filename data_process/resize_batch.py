import os
import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch center-crop, resize, and optionally frame-sample video frames"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="input frame folders",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="output frame folders",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        required=True,
        metavar=("W", "H"),
        help="Target size (width height)",
    )
    parser.add_argument(
        "--max_frame",
        type=int,
        default=None,
        help="Maximum number of frames per video; if exceeded, uniform sampling will be applied",
    )
    return parser.parse_args()

def center_crop_and_resize(img, target_width, target_height):
    h, w = img.shape[:2]
    scale = max(target_width / w, target_height / h)  # Scale based on target size
    new_w, new_h = round(w * scale), round(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    # Center crop
    start_x = (new_w - target_width) // 2
    start_y = (new_h - target_height) // 2
    return img_resized[start_y:start_y + target_height, start_x:start_x + target_width]


def uniform_sample_indices(total_frames, max_frames):
    if total_frames <= max_frames:
        return np.arange(total_frames)
    indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
    return indices


def process_video_frames_folder(
    input_folder,
    output_folder,
    target_width,
    target_height,
    max_frame=None,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
):
    for video_name in sorted(os.listdir(input_folder)):
        video_in_dir = os.path.join(input_folder, video_name)
        if not os.path.isdir(video_in_dir):
            continue
        video_out_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(video_in_dir) if f.lower().endswith(exts)])

        if not images:
            print(f"Skipping empty folder: {video_in_dir}")
            continue

        print(f"Processing {video_in_dir}")

        # Check whether to sample
        sampling_applied = False
        if max_frame is not None and len(images) > max_frame:
            indices = uniform_sample_indices(len(images), max_frame)
            selected_images = [images[i] for i in indices]
            sampling_applied = True
            print(f"  Sampling {len(selected_images)} frames (down from {len(images)})")
        else:
            selected_images = images

        for idx, fname in enumerate(selected_images):
            in_path = os.path.join(video_in_dir, fname)

            if sampling_applied:
                out_name = f"{idx:05d}.png"
            else:
                out_name = fname

            out_path = os.path.join(video_out_dir, out_name)

            img = cv2.imread(in_path)
            if img is None:
                print(f"  Failed to read: {in_path}")
                continue
            cropped = center_crop_and_resize(img, target_width, target_height)
            cv2.imwrite(out_path, cropped)

        print(f"  Output saved to {video_out_dir}\n")


if __name__ == "__main__":
    args = parse_args()
    process_video_frames_folder(
        args.input_path,
        args.output_path,
        args.size[0],
        args.size[1],
        args.max_frame,
    )