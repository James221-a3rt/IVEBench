import os
import json
import csv
import datetime
import importlib
import numpy as np
import logging
from pathlib import Path

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{timestamp}_ivebench.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
                        logging.StreamHandler()  
                    ])

def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_types(item) for item in obj)
        else:
            return obj

def save_json(data, path):
    
    converted_data = convert_types(data)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class VEBench(object):
    def __init__(self, device, output_path):
        self.device = device
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.metric_folder_map = {
            "subject_consistency": "quality",
            "temporal_flickering": "quality",
            "background_consistency": "quality",
            "motion_smoothness": "quality",
            "vtss": "quality",
            "overall_semantic_consistency": "compliance",
            "instruction_satisfaction": "compliance",
            "phrase_semantic_consistency": "compliance",
            "quantity_accuracy": "compliance",
            "semantic_fidelity": "fidelity",
            "motion_fidelity": "fidelity",
            "content_fidelity": "fidelity"
        }
        
        self.logger.info(f"VEBench initialized with device: {device}")
        self.logger.info(f"Output path: {output_path}")

    def build_full_metric_list(self):
        return [
            "subject_consistency",
            "temporal_flickering",
            "background_consistency",
            "motion_smoothness",
            "vtss",
            "overall_semantic_consistency",
            "instruction_satisfaction",
            "phrase_semantic_consistency",
            "quantity_accuracy",
            "semantic_fidelity",
            "motion_fidelity",
            "content_fidelity"
        ]

    def load_video_info(self, info_json_path):
        with open(info_json_path, 'r', encoding='utf-8') as f:
            video_info = json.load(f)
        return video_info

    def save_results_to_csv(self, results_dict, output_csv_path):
        if not results_dict:
            self.logger.warning("No results to save")
            return

        video_data = {}
        all_metrics = set()
        
        for metric, (avg_score, detailed_results) in results_dict.items():
            all_metrics.add(metric)
            self.logger.info(f"Processing metric: {metric} with {len(detailed_results)} results")
            
            for i, result in enumerate(detailed_results):
                video_key = result.get('video_name') or str(result.get('video_id', f'unknown_{i}'))
                
                if video_key not in video_data:
                    video_data[video_key] = {}
                    for key, value in result.items():
                        if key not in ['video_results', 'metric', 'avg_score', 'error']:
                            video_data[video_key][key] = value
                
                score_value = result.get('video_results', 0.0)
                video_data[video_key][f'{metric}_score'] = score_value
                
                if 'error' in result:
                    video_data[video_key][f'{metric}_error'] = result['error']
        
        if not video_data:
            self.logger.warning("No video data to save")
            return
        
        self.logger.info(f"Total unique videos found: {len(video_data)}")
        self.logger.info(f"Metrics processed: {sorted(all_metrics)}")
        
        basic_columns = set()
        score_columns = set()
        error_columns = set()
        
        for video_info in video_data.values():
            for key in video_info.keys():
                if key.endswith('_score'):
                    score_columns.add(key)
                elif key.endswith('_error'):
                    error_columns.add(key)
                else:
                    basic_columns.add(key)
        
        basic_columns = sorted(list(basic_columns))
        score_columns = sorted(list(score_columns))
        error_columns = sorted(list(error_columns))
        fieldnames = basic_columns + score_columns + error_columns
        
        self.logger.debug(f"CSV columns: {fieldnames}")
        
        csv_rows = []
        for video_key, video_info in video_data.items():
            row = {}
            for col in fieldnames:
                if col.endswith('_score'):
                    row[col] = video_info.get(col, 0.0)
                else:
                    row[col] = video_info.get(col, '')
            csv_rows.append(row)
        
        if 'video_id' in basic_columns:
            csv_rows.sort(key=lambda x: int(x.get('video_id', 0)) if str(x.get('video_id', 0)).isdigit() else 0)
        
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            self.logger.info(f'Results saved to CSV: {output_csv_path}')
            self.logger.info(f'Total videos: {len(csv_rows)}, Metrics: {len(all_metrics)}')
            
            self._print_metric_statistics(csv_rows, all_metrics)
            
        except Exception as e:
            self.logger.error(f"Error saving CSV file: {e}")

    def _print_metric_statistics(self, csv_rows, all_metrics):
        self.logger.info("=== Metric Statistics ===")
        
        for metric in sorted(all_metrics):
            score_col = f'{metric}_score'
            if score_col in csv_rows[0] if csv_rows else False:
                scores = [float(row[score_col]) for row in csv_rows if float(row[score_col]) != -1.0]
                total_count = len([row for row in csv_rows])
                invalid_count = total_count - len(scores)
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    min_score = min(scores)
                    max_score = max(scores)
                    self.logger.info(f'{metric}: {len(scores)}/{total_count} valid videos evaluated '
                                f'({invalid_count} skipped/failed), '
                                f'avg={avg_score:.4f}, min={min_score:.4f}, max={max_score:.4f}')
                else:
                    self.logger.warning(f'{metric}: No valid scores found - all {total_count} videos skipped/failed')

    def save_results_to_json(self, results_dict, output_json_path):
        try:
            save_json(results_dict, output_json_path)
            self.logger.info(f"Detailed results saved to JSON: {output_json_path}")
        except Exception as e:
            self.logger.error(f"Error saving JSON results: {e}")

    def evaluate(self, source_videos_path, target_videos_path, info_json_path, 
                name, metric_list=None, save_json_results=True, **kwargs):
        results_dict = {}
        
        if metric_list is None:
            metric_list = self.build_full_metric_list()
        
        if not os.path.exists(source_videos_path):
            raise FileNotFoundError(f"Source videos path not found: {source_videos_path}")
        if not os.path.exists(target_videos_path):
            raise FileNotFoundError(f"Target videos path not found: {target_videos_path}")
        if not os.path.exists(info_json_path):
            raise FileNotFoundError(f"Info JSON file not found: {info_json_path}")
        
        priority_metrics = ["content_fidelity", "instruction_satisfaction"]
        ordered_metric_list = []

        for priority_metric in priority_metrics:
            if priority_metric in metric_list:
                ordered_metric_list.append(priority_metric)
        
        for metric in metric_list:
            if metric not in priority_metrics:
                ordered_metric_list.append(metric)
        
        self.logger.info(f"Starting evaluation with metrics (prioritized): {ordered_metric_list}")
        
        for metric in ordered_metric_list:
            try:
                folder_name = self.metric_folder_map.get(metric, "quality")  

                metric_module = importlib.import_module(f'{folder_name}.{metric}')
                evaluate_func = getattr(metric_module, f'compute_{metric}')
                
                self.logger.info(f"Evaluating metric: {metric} (from {folder_name} folder)")
                
                results = evaluate_func(
                    json_dir=info_json_path,
                    device=self.device,
                    source_videos_path=source_videos_path,
                    target_videos_path=target_videos_path,
                    **kwargs
                )
                
                results_dict[metric] = results
                self.logger.info(f"Completed metric: {metric}, Average score: {results[0]:.4f}")
                
            except Exception as e:
                self.logger.error(f'Error in metric {metric}: {e}')
                results_dict[metric] = (0.0, [])
        
        output_csv = os.path.join(self.output_path, f'{name}_eval_results.csv')
        
        self.save_results_to_csv(results_dict, output_csv)
        
        return results_dict