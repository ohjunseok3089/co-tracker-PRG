#!/usr/bin/env python3

import argparse
import cv2
import json
import numpy as np
import os
from track_red import detect_red_circle, calculate_head_movement

def process_video(video_path, output_path=None, fps_override=None):
    """
    Process video to track red circles and calculate head movements.
    
    Args:
        video_path: Path to input video file
        output_path: Path to output JSON file (optional)
        fps_override: Manual FPS value to use instead of video metadata (optional)
    
    Returns:
        List of frame data with head movement calculations
    """
    print(f"Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use override FPS if provided, otherwise use detected FPS
    if fps_override is not None:
        fps = float(fps_override)
        print(f"Video info: {total_frames} frames, {detected_fps:.2f} FPS (detected) -> {fps:.2f} FPS (override), {width}x{height}")
    else:
        fps = detected_fps
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
    # Validate FPS
    if fps <= 0 or fps > 1000:
        print(f"Warning: Invalid FPS detected ({fps}), defaulting to 30 FPS")
        fps = 30.0
    
    # Initialize tracking variables
    frame_data = []
    prev_red_pos = None
    frame_idx = 0
    
    print("\nProcessing frames...")
    print("Frame | Red Circle Position | Head Movement (Horizontal | Vertical)")
    print("-" * 80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect red circle in current frame
        red_circle = detect_red_circle(frame)
        
        if red_circle is not None:
            curr_red_pos = (red_circle[0], red_circle[1])  # (x, y)
            curr_radius = red_circle[2]
        else:
            curr_red_pos = None
            curr_radius = None
        
        # Calculate head movement
        if frame_idx == 0:
            # First frame - no movement
            head_movement = {
                "horizontal": {"radians": 0.0, "degrees": 0.0},
                "vertical": {"radians": 0.0, "degrees": 0.0}
            }
        else:
            # Calculate movement from previous frame
            if curr_red_pos is not None and prev_red_pos is not None:
                head_movement = calculate_head_movement(
                    prev_red_pos, curr_red_pos, width, height
                )
            else:
                # Red circle not trackable - mark as NaN
                head_movement = {
                    "horizontal": {"radians": float('nan'), "degrees": float('nan')},
                    "vertical": {"radians": float('nan'), "degrees": float('nan')}
                }

        # Prepare frame data
        frame_info = {
            "frame_index": frame_idx,
            "timestamp": frame_idx / fps,
            "red_circle": {
                "detected": curr_red_pos is not None,
                "position": curr_red_pos,  # (x, y) or None
                "radius": curr_radius
            },
            "head_movement": head_movement,
            "previous_frame": {
                "index": frame_idx - 1 if frame_idx > 0 else None,
                "red_position": prev_red_pos
            }
        }

        frame_data.append(frame_info)

        # Print current frame info
        red_pos_str = f"({curr_red_pos[0]:3d}, {curr_red_pos[1]:3d})" if curr_red_pos else "Not found"
        if head_movement and not np.isnan(head_movement["horizontal"]["radians"]):
            h_rad = head_movement["horizontal"]["radians"]
            h_deg = head_movement["horizontal"]["degrees"]
            v_rad = head_movement["vertical"]["radians"]
            v_deg = head_movement["vertical"]["degrees"]
            movement_str = f"H:({h_rad:6.3f},{h_deg:6.2f}°) V:({v_rad:6.3f},{v_deg:6.2f}°)"
        else:
            movement_str = "H:(NaN,NaN°) V:(NaN,NaN°)"
        print(f"{frame_idx:5d} | {red_pos_str:15s} | {movement_str}")
        
        # Update previous position
        # Important: Only update if current detection is valid, otherwise keep previous
        if curr_red_pos is not None:
            prev_red_pos = curr_red_pos
        
        frame_idx += 1
        
        # Progress indicator for long videos
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
    
    cap.release()
    
    # Summary statistics
    detected_frames = sum(1 for f in frame_data if f["red_circle"]["detected"])
    valid_movements = sum(1 for f in frame_data if not np.isnan(f["head_movement"]["horizontal"]["radians"]))
    
    print(f"\nProcessing complete!")
    print(f"Total frames: {len(frame_data)}")
    print(f"Frames with red circle detected: {detected_frames} ({detected_frames/len(frame_data)*100:.1f}%)")
    print(f"Valid head movements calculated: {valid_movements} ({valid_movements/len(frame_data)*100:.1f}%)")
    
    # Calculate movement statistics
    h_movements_rad = [f["head_movement"]["horizontal"]["radians"] for f in frame_data if not np.isnan(f["head_movement"]["horizontal"]["radians"])]
    v_movements_rad = [f["head_movement"]["vertical"]["radians"] for f in frame_data if not np.isnan(f["head_movement"]["vertical"]["radians"])]
    
    if h_movements_rad:
        h_min_rad = float(np.min(h_movements_rad))
        h_max_rad = float(np.max(h_movements_rad))
        print(f"Horizontal movement range: {h_min_rad:.3f} to {h_max_rad:.3f} radians")
        print(f"                          {np.degrees(h_min_rad):.2f}° to {np.degrees(h_max_rad):.2f}°")
    
    if v_movements_rad:
        v_min_rad = float(np.min(v_movements_rad))
        v_max_rad = float(np.max(v_movements_rad))
        print(f"Vertical movement range:   {v_min_rad:.3f} to {v_max_rad:.3f} radians")
        print(f"                          {np.degrees(v_min_rad):.2f}° to {np.degrees(v_max_rad):.2f}°")
    
    # Save to JSON file
    if output_path:
        save_results(frame_data, video_path, output_path)
    
    return frame_data


def save_results(frame_data, video_path, output_path):
    """Save results to JSON file with metadata."""
    
    # Calculate movement statistics
    h_movements_rad = [f["head_movement"]["horizontal"]["radians"] for f in frame_data if not np.isnan(f["head_movement"]["horizontal"]["radians"])]
    v_movements_rad = [f["head_movement"]["vertical"]["radians"] for f in frame_data if not np.isnan(f["head_movement"]["vertical"]["radians"])]
    h_movements_deg = [f["head_movement"]["horizontal"]["degrees"] for f in frame_data if not np.isnan(f["head_movement"]["horizontal"]["degrees"])]
    v_movements_deg = [f["head_movement"]["vertical"]["degrees"] for f in frame_data if not np.isnan(f["head_movement"]["vertical"]["degrees"])]
    
    # Calculate statistics for horizontal movement
    h_stats_rad = {}
    h_stats_deg = {}
    if h_movements_rad:
        h_stats_rad = {
            "mean": float(np.mean(h_movements_rad)),
            "median": float(np.median(h_movements_rad)),
            "std": float(np.std(h_movements_rad)),
            "min": float(np.min(h_movements_rad)),
            "max": float(np.max(h_movements_rad))
        }
        h_stats_deg = {
            "mean": float(np.mean(h_movements_deg)),
            "median": float(np.median(h_movements_deg)),
            "std": float(np.std(h_movements_deg)),
            "min": float(np.min(h_movements_deg)),
            "max": float(np.max(h_movements_deg))
        }
    
    # Calculate statistics for vertical movement
    v_stats_rad = {}
    v_stats_deg = {}
    if v_movements_rad:
        v_stats_rad = {
            "mean": float(np.mean(v_movements_rad)),
            "median": float(np.median(v_movements_rad)),
            "std": float(np.std(v_movements_rad)),
            "min": float(np.min(v_movements_rad)),
            "max": float(np.max(v_movements_rad))
        }
        v_stats_deg = {
            "mean": float(np.mean(v_movements_deg)),
            "median": float(np.median(v_movements_deg)),
            "std": float(np.std(v_movements_deg)),
            "min": float(np.min(v_movements_deg)),
            "max": float(np.max(v_movements_deg))
        }
    
    # Prepare output data with metadata
    output_data = {
        "metadata": {
            "video_path": video_path,
            "total_frames": len(frame_data),
            "target_color_rgb": [255, 28, 48],
            "fov_degrees": 104.0,
            "processing_info": {
                "detected_frames": sum(1 for f in frame_data if f["red_circle"]["detected"]),
                "valid_movements": sum(1 for f in frame_data if not np.isnan(f["head_movement"]["horizontal"]["radians"]))
            },
            "movement_statistics": {
                "horizontal": {
                    "radians": h_stats_rad,
                    "degrees": h_stats_deg,
                    "description": "Positive = right turn, Negative = left turn"
                },
                "vertical": {
                    "radians": v_stats_rad,
                    "degrees": v_stats_deg,
                    "description": "Positive = tilt down, Negative = tilt up"
                }
            }
        },
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
        print(f"Results saved to: {output_path}")
        
        # Print statistics summary
        print(f"\nMovement Statistics Summary:")
        if h_stats_deg:
            print(f"Horizontal (Yaw): mean={h_stats_deg['mean']:.2f}°, std={h_stats_deg['std']:.2f}°, median={h_stats_deg['median']:.2f}°")
            print(f"                  range=[{h_stats_deg['min']:.2f}° to {h_stats_deg['max']:.2f}°]")
        if v_stats_deg:
            print(f"Vertical (Pitch): mean={v_stats_deg['mean']:.2f}°, std={v_stats_deg['std']:.2f}°, median={v_stats_deg['median']:.2f}°")
            print(f"                  range=[{v_stats_deg['min']:.2f}° to {v_stats_deg['max']:.2f}°]")
            
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Track red circles and calculate head movements from video')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    parser.add_argument('--target_radius', '-r', type=int, default=3, 
                      help='Expected radius of red circles (default: 3)')
    parser.add_argument('--fps', '-f', type=float, 
                      help='Override video FPS (use if video metadata is incorrect)')
    
    args = parser.parse_args()
    
    # Validate input video file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Set default output path if not specified
    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        args.output = f"{video_name}_head_movement.json"
    
    # Process video
    results = process_video(args.video_path, args.output, args.fps)
    
    if results is None:
        return 1
    
    print(f"\nHead movement tracking completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 