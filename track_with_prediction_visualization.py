#!/usr/bin/env python3

import argparse
import cv2
import json
import numpy as np
import os
from track_red import detect_red_circle, calculate_head_movement, remap_position_from_movement

def process_video_with_prediction_visualization(video_path, output_video_path=None, output_json_path=None, fps_override=None, show_video=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = float(fps_override) if fps_override is not None else detected_fps
    
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"Warning: Could not open video writer for {output_video_path}")
            video_writer = None
    
    frame_data = []
    all_frames = []
    frame_idx = 0
    
    print("\nReading all frames...")
    print("-" * 80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame.copy())
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Read {frame_idx} frames...")
    
    cap.release()
    total_frames = len(all_frames)
    print(f"Total frames read: {total_frames}")
    
    prediction_errors = []
    
    print("\nProcessing frames with predictions...")
    print("-" * 80)
    for frame_idx in range(total_frames):
        frame = all_frames[frame_idx]
        vis_frame = frame.copy()
        
        red_circle = detect_red_circle(frame)
        
        if red_circle is not None:
            curr_red_pos = (float(red_circle[0]), float(red_circle[1]))
            curr_radius = int(red_circle[2])
        else:
            curr_red_pos = None
            curr_radius = None
        
        head_movement = None
        predicted_pos_for_next_frame = None
        prediction_error = None
        actual_next_pos = None
        
        if frame_idx == 0:
            head_movement = {
                "horizontal": {"radians": 0.0, "degrees": 0.0},
                "vertical": {"radians": 0.0, "degrees": 0.0}
            }
        else:
            prev_frame = all_frames[frame_idx - 1]
            prev_red_circle = detect_red_circle(prev_frame)
            
            if prev_red_circle is not None:
                prev_red_pos = (float(prev_red_circle[0]), float(prev_red_circle[1]))
            else:
                prev_red_pos = None
            
            if curr_red_pos is not None and prev_red_pos is not None:
                head_movement = calculate_head_movement(
                    prev_red_pos, curr_red_pos, width, height
                )
                
                predicted_pos_for_next_frame = remap_position_from_movement(
                    curr_red_pos, head_movement, width, height
                )
            else:
                head_movement = {
                    "horizontal": {"radians": float('nan'), "degrees": float('nan')},
                    "vertical": {"radians": float('nan'), "degrees": float('nan')}
                }
        
        if frame_idx < total_frames - 1:
            next_frame = all_frames[frame_idx + 1]
            next_red_circle = detect_red_circle(next_frame)
            
            if next_red_circle is not None:
                actual_next_pos = (float(next_red_circle[0]), float(next_red_circle[1]))
                
                if predicted_pos_for_next_frame is not None and actual_next_pos is not None:
                    error_x = predicted_pos_for_next_frame[0] - actual_next_pos[0]
                    error_y = predicted_pos_for_next_frame[1] - actual_next_pos[1]
                    prediction_error = {
                        "error_x": float(error_x),
                        "error_y": float(error_y),
                        "distance": float(np.sqrt(error_x**2 + error_y**2)),
                        "predicted_pos": [float(predicted_pos_for_next_frame[0]), float(predicted_pos_for_next_frame[1])],
                        "actual_next_pos": [float(actual_next_pos[0]), float(actual_next_pos[1])]
                    }
                    prediction_errors.append(prediction_error["distance"])
        
        if curr_red_pos is not None:
            display_radius = max(3, curr_radius if curr_radius is not None else 3)
            cv2.circle(vis_frame, (int(curr_red_pos[0]), int(curr_red_pos[1])), 
                      display_radius, (0, 0, 255), 2)
            cv2.circle(vis_frame, (int(curr_red_pos[0]), int(curr_red_pos[1])), 
                      2, (0, 0, 255), -1)
            
            cv2.putText(vis_frame, "ACTUAL", 
                       (int(curr_red_pos[0]) + 10, int(curr_red_pos[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if predicted_pos_for_next_frame is not None:
            cv2.circle(vis_frame, (int(predicted_pos_for_next_frame[0]), int(predicted_pos_for_next_frame[1])), 
                      5, (0, 255, 0), 2)
            cv2.circle(vis_frame, (int(predicted_pos_for_next_frame[0]), int(predicted_pos_for_next_frame[1])), 
                      2, (0, 255, 0), -1)
            
            cv2.putText(vis_frame, "PREDICTED NEXT", 
                       (int(predicted_pos_for_next_frame[0]) + 10, int(predicted_pos_for_next_frame[1]) + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if curr_red_pos is not None:
                cv2.line(vis_frame, (int(curr_red_pos[0]), int(curr_red_pos[1])),
                        (int(predicted_pos_for_next_frame[0]), int(predicted_pos_for_next_frame[1])), (255, 255, 0), 1)
        
        info_y = 30
        cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames-1}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        
        if prediction_error is not None:
            cv2.putText(vis_frame, f"Prediction Error: {prediction_error['distance']:.2f} px", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 25
        
        if head_movement is not None and not np.isnan(head_movement["horizontal"]["radians"]):
            cv2.putText(vis_frame, f"H: {head_movement['horizontal']['degrees']:.2f}° V: {head_movement['vertical']['degrees']:.2f}°", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frame_info = {
            "frame_index": frame_idx,
            "timestamp": frame_idx / fps,
            "red_circle": {
                "detected": curr_red_pos is not None,
                "position": curr_red_pos,  # (x, y) or None
                "radius": curr_radius
            },
            "head_movement": head_movement,
            "prediction": {
                "predicted_position": predicted_pos_for_next_frame,
                "error": prediction_error,
                "actual_next_position": actual_next_pos
            },
            "previous_frame": {
                "index": frame_idx - 1 if frame_idx > 0 else None,
                "red_position": None
            }
        }

        frame_data.append(frame_info)
        
        if video_writer is not None:
            video_writer.write(vis_frame)
        
        if show_video:
            cv2.imshow('Red Circle Tracking with Predictions', vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        if curr_red_pos is not None:
            prev_red_pos = curr_red_pos
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
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
            "max_error_px": float(np.max(prediction_errors)),
            "total_predictions": len(prediction_errors)
        }
    
    print(f"\nProcessing complete!")
    print(f"Total frames: {len(frame_data)}")
    print(f"Frames with red circle detected: {detected_frames} ({detected_frames/len(frame_data)*100:.1f}%)")
    print(f"Valid predictions made: {valid_predictions}")
    
    if prediction_errors:
        print(f"\nPrediction Accuracy:")
        print(f"Mean error: {analysis_results['prediction_accuracy']['mean_error_px']:.2f} pixels")
        print(f"Median error: {analysis_results['prediction_accuracy']['median_error_px']:.2f} pixels")
        print(f"Std deviation: {analysis_results['prediction_accuracy']['std_error_px']:.2f} pixels")
        print(f"Range: {analysis_results['prediction_accuracy']['min_error_px']:.2f} - {analysis_results['prediction_accuracy']['max_error_px']:.2f} pixels")
    
    # Save JSON results if requested
    if output_json_path:
        save_analysis_results(frame_data, analysis_results, video_path, output_json_path)
    
    return {
        "frame_data": frame_data,
        "analysis": analysis_results
    }


def save_analysis_results(frame_data, analysis_results, video_path, output_path):
    """Save analysis results to JSON file."""
    
    output_data = {
        "metadata": {
            "video_path": video_path,
            "target_color_rgb": [255, 28, 48],
            "fov_degrees": 104.0,
            "analysis_type": "red_circle_tracking_with_prediction_validation"
        },
        "analysis_results": analysis_results,
        "frames": frame_data
    }
    
    # Convert NaN values to null for JSON compatibility
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
    
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Analysis results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving analysis results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Track red circles with prediction visualization')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output_video', '-ov', help='Output video file path with visualizations (optional)')
    parser.add_argument('--output_json', '-oj', help='Output JSON file path for analysis data (optional)')
    parser.add_argument('--fps', '-f', type=float, 
                      help='Override video FPS (use if video metadata is incorrect)')
    parser.add_argument('--show', '-s', action='store_true',
                      help='Display video in real-time (press q to quit)')
    
    args = parser.parse_args()
    
    # Validate input video file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Set default output paths if not specified
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    if args.output_video is None:
        args.output_video = f"{video_name}_prediction_visualization.mp4"
    
    if args.output_json is None:
        args.output_json = f"{video_name}_prediction_analysis.json"
    
    # Process video
    results = process_video_with_prediction_visualization(
        args.video_path, 
        args.output_video, 
        args.output_json, 
        args.fps,
        args.show
    )
    
    if results is None:
        return 1
    
    print(f"\nPrediction visualization completed successfully!")
    if args.output_video:
        print(f"Output video: {args.output_video}")
    if args.output_json:
        print(f"Analysis data: {args.output_json}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 