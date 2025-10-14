import torch
import os
from ivebench import VEBench
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='IVEBench - Video Editing Benchmark', 
                                   formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "--output_path",
        type=str,
        default='',
        help="Output path to save the evaluation results",
    )
    
    parser.add_argument(
        "--source_videos_path",
        type=str,
        default='',
        help="Folder that contains the source video frames",
    )
    
    parser.add_argument(
        "--target_videos_path", 
        type=str,
        default='',
        help="Folder that contains the edited video frames",
    )
    
    parser.add_argument(
        "--info_json_path",
        type=str,
        default='',
        help="Path to the JSON file containing video information and prompts",
    )
    
    parser.add_argument(
        "--metric",
        nargs='+',
        default=None,
        help="List of evaluation metrics, usage: --metric <metric_1> <metric_2>",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="ivebench_eval",
        help="Name prefix for output files",
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    print(f'Arguments: {args}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_VEBench = VEBench(device, args.output_path)
    
    print(f'Starting IVEBench evaluation on device: {device}')
    
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    eval_name = f'{args.name}_{current_time}'
    
    my_VEBench.evaluate(
        source_videos_path=args.source_videos_path,
        target_videos_path=args.target_videos_path,
        info_json_path=args.info_json_path,
        name=eval_name,
        metric_list=args.metric
    )
    
    print('Evaluation completed successfully!')


if __name__ == "__main__":
    main()