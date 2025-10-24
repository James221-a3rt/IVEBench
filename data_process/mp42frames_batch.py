import os
import subprocess
import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract video frames using FFmpeg")
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the source video folder"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to the output folder for extracted frames"
    )
    return parser.parse_args()


def extract_frames_with_ffmpeg(input_folder, output_folder):
    """Extract frames only from .mp4 videos."""
    os.makedirs(output_folder, exist_ok=True)
    videos = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]
    
    if not videos:
        print("No .mp4 video files found.")
        return

    for filename in videos:
        name, _ = os.path.splitext(filename)
        video_path = os.path.join(input_folder, filename)
        output_dir = os.path.join(output_folder, name)
        os.makedirs(output_dir, exist_ok=True)

        # FFmpeg command to extract frames
        output_pattern = os.path.join(output_dir, "%05d.png")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vsync", "0",
            "-q:v", "1",
            "-pix_fmt", "rgb24",
            "-start_number", "0",
            output_pattern,
            "-hide_banner",
            "-loglevel", "error"
        ]

        print(f"Processing: {filename}")
        subprocess.run(cmd, check=True)
        print(f"Extracted frames saved to: {output_dir}")

    print("All video frames have been extracted!")


if __name__ == "__main__":
    args = parse_args()
    extract_frames_with_ffmpeg(
        input_folder=args.input_path,
        output_folder=args.output_path
    )