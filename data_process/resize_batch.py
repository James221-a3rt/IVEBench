import os
import cv2
import argparse

def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch center-crop and resize video frames")
    parser.add_argument(
        '--input_path',
        required=True,
        help='Root directory of input frame folders'
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help='Root directory for output frame folders'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=(854, 480),
        metavar=('W', 'H'),
        help='Target size (width height)'
    )
    return parser.parse_args()

def center_crop_and_resize(img, target_width, target_height):
    """Scale the image so that the shorter side fits the target, then center crop."""
    h, w = img.shape[:2]
    scale = max(target_width / w, target_height / h)  # Scale based on target size
    new_w, new_h = round(w * scale), round(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    # Center crop
    start_x = (new_w - target_width) // 2
    start_y = (new_h - target_height) // 2
    return img_resized[start_y:start_y + target_height, start_x:start_x + target_width]


def process_video_frames_folder(
    input_folder, output_folder, target_width, target_height,
    exts=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
):
    """Batch process all frame folders: resize and center crop frames."""
    for video_name in sorted(os.listdir(input_folder)):
        video_in_dir = os.path.join(input_folder, video_name)
        if not os.path.isdir(video_in_dir):
            continue
        video_out_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        images = sorted([
            f for f in os.listdir(video_in_dir)
            if f.lower().endswith(exts)
        ])

        if not images:
            print(f"Skipping empty folder: {video_in_dir}")
            continue

        print(f"Processing {video_in_dir}, total {len(images)} frames")
        for fname in images:
            in_path = os.path.join(video_in_dir, fname)
            out_path = os.path.join(video_out_dir, fname)

            img = cv2.imread(in_path)
            if img is None:
                print(f"Failed to read: {in_path}")
                continue
            cropped = center_crop_and_resize(img, target_width, target_height)
            cv2.imwrite(out_path, cropped)
        print(f"Output saved to {video_out_dir}")


if __name__ == "__main__":
    args = parse_args()
    process_video_frames_folder(args.input_path, args.output_path, args.size[0], args.size[1])