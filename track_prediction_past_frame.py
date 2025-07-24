#!/usr/bin/env python3

import argparse
import cv2
import json
import numpy as np
import os
from track_red import detect_red_circle, calculate_head_movement, remap_position_from_movement

def process_video_past_frame_prediction(video_path, output_video_path=None, output_json_path=None, fps_override=None, show_video=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = float(fps_override) if fps_override else detected_fps
    
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_data = []
    prev_red_pos = None
    frame_idx = 0
    prediction_errors = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        vis_frame = frame.copy()
        red_circle = detect_red_circle(frame) 
        
        if red_circle is not None:
            curr_red_pos = (float(red_circle[0]), float(red_circle[1]))
            curr_radius = int(red_circle[2])
        else:
            curr_red_pos = None
            curr_radius = None
        
        head_movement = None
        recalculated_pos = None
        prediction_error = None
        
        if frame_idx == 0:
            head_movement = {"horizontal": {"radians": 0.0, "degrees": 0.0}, "vertical": {"radians": 0.0, "degrees": 0.0}}
        else:
            if curr_red_pos is not None and prev_red_pos is not None:
                head_movement = calculate_head_movement(prev_red_pos, curr_red_pos, width, height)
                
                if head_movement is not None and not np.isnan(head_movement['horizontal']['radians']):
                    recalculated_pos = remap_position_from_movement(prev_red_pos, head_movement, width, height)
                    
                    if recalculated_pos is not None:
                        error_x = recalculated_pos[0] - curr_red_pos[0]
                        error_y = recalculated_pos[1] - curr_red_pos[1]
                        prediction_error = {
                            "error_x": float(error_x),
                            "error_y": float(error_y),
                            "distance": float(np.sqrt(error_x**2 + error_y**2)),
                            "recalculated_pos": [float(recalculated_pos[0]), float(recalculated_pos[1])],
                            "actual_pos": [float(curr_red_pos[0]), float(curr_red_pos[1])]
                        }
                        prediction_errors.append(prediction_error["distance"])
            else:
                head_movement = {"horizontal": {"radians": float('nan'), "degrees": float('nan')}, "vertical": {"radians": float('nan'), "degrees": float('nan')}}
        
        if curr_red_pos is not None:
            display_radius = max(3, curr_radius if curr_radius is not None else 3)
            cv2.circle(vis_frame, (int(curr_red_pos[0]), int(curr_red_pos[1])), display_radius, (0, 0, 255), 2)
            cv2.circle(vis_frame, (int(curr_red_pos[0]), int(curr_red_pos[1])), 2, (0, 0, 255), -1)
            cv2.putText(vis_frame, "Actual", (int(curr_red_pos[0]) + 10, int(curr_red_pos[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if recalculated_pos is not None:
            cv2.circle(vis_frame, (int(recalculated_pos[0]), int(recalculated_pos[1])), 5, (0, 255, 0), 2)
            cv2.circle(vis_frame, (int(recalculated_pos[0]), int(recalculated_pos[1])), 2, (0, 255, 0), -1)
            cv2.putText(vis_frame, "Calculated", (int(recalculated_pos[0]) + 10, int(recalculated_pos[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if curr_red_pos is not None:
                cv2.line(vis_frame, (int(recalculated_pos[0]), int(recalculated_pos[1])), (int(curr_red_pos[0]), int(curr_red_pos[1])), (255, 255, 0), 1)
        
        cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames-1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if prediction_error is not None:
            cv2.putText(vis_frame, f"Error: {prediction_error['distance']:.2f} px", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if head_movement is not None and not np.isnan(head_movement["horizontal"]["radians"]):
            cv2.putText(vis_frame, f"H: {head_movement['horizontal']['degrees']:.2f}° V: {head_movement['vertical']['degrees']:.2f}°", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frame_info = {
            "frame_index": frame_idx,
            "timestamp": frame_idx / fps,
            "red_circle": {"detected": curr_red_pos is not None, "position": curr_red_pos, "radius": curr_radius},
            "head_movement": head_movement,
            "prediction": {"recalculated_position": recalculated_pos, "error": prediction_error}
        }
        frame_data.append(frame_info)
        
        if video_writer is not None:
            video_writer.write(vis_frame)
        
        if show_video:
            cv2.imshow('Past Frame Prediction', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if curr_red_pos is not None:
            prev_red_pos = curr_red_pos
        
        frame_idx += 1
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    detected_frames = sum(1 for f in frame_data if f["red_circle"]["detected"])
    valid_predictions = len(prediction_errors)
    
    analysis_results = {
        "total_frames": len(frame_data),
        "detected_frames": detected_frames,
        "valid_predictions": valid_predictions,
        "prediction_accuracy": {}
    }
    
    if prediction_errors:
        analysis_results["prediction_accuracy"] = {
            "mean_error_px": float(np.mean(prediction_errors)),
            "median_error_px": float(np.median(prediction_errors)),
            "std_error_px": float(np.std(prediction_errors)),
            "min_error_px": float(np.min(prediction_errors)),
            "max_error_px": float(np.max(prediction_errors))
        }
    
    if output_json_path:
        output_data = {
            "metadata": {"video_path": video_path, "analysis_type": "past_frame_prediction"},
            "analysis_results": analysis_results,
            "frames": frame_data
        }
        
        def convert_nan(obj):
            if isinstance(obj, dict):
                return {k: convert_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan(item) for item in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            else:
                return obj
        
        output_data = convert_nan(output_data)
        
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Past frame prediction tracking')
    parser.add_argument('video_path', help='Input video file')
    parser.add_argument('--output_video', '-ov', help='Output video file')
    parser.add_argument('--output_json', '-oj', help='Output JSON file')
    parser.add_argument('--fps', '-f', type=float, help='Override video FPS')
    parser.add_argument('--show', '-s', action='store_true', help='Show video')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    if args.output_video is None:
        args.output_video = f"{video_name}_past_prediction.mp4"
    if args.output_json is None:
        args.output_json = f"{video_name}_past_analysis.json"
    
    process_video_past_frame_prediction(args.video_path, args.output_video, args.output_json, args.fps, args.show)

if __name__ == "__main__":
    exit(main())