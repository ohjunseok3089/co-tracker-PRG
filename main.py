# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
from PIL import Image
import cv2
import imageio
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerOnlinePredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
FRAMES_INTERVAL = 10

def extract_video_info(video_path):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    num_frames = reader.get_length()
    reader.close()
    return fps, num_frames

def extract_frames(video, seconds, fps, start_frame, num_frames):
    frames_to_extract = int(fps * seconds)
    end_frame = start_frame + frames_to_extract
    if end_frame > num_frames:
        end_frame = num_frames
    video = video[start_frame:end_frame]
    return video, end_frame
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--mask_path",
        default=None,
        help="path to a mask",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    print("Loading model...")
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    print("Model loaded.")

    print(f"Moving model to device: {DEFAULT_DEVICE}")
    model = model.to(DEFAULT_DEVICE)
    print("Model moved to device.")

    window_frames = []
    if args.mask_path is not None:
        segm_mask = np.array(Image.open(args.mask_path))
        print(f"Original segm_mask shape: {segm_mask.shape}")
        if segm_mask.ndim == 4:
            segm_mask_gray = segm_mask[..., 0, 0] if segm_mask.shape[3] == 1 else segm_mask[..., 0]
        elif segm_mask.ndim == 3 and segm_mask.shape[2] == 3:
            segm_mask_gray = cv2.cvtColor(segm_mask, cv2.COLOR_RGB2GRAY)
        elif segm_mask.ndim == 3 and segm_mask.shape[2] == 4:
            segm_mask_gray = segm_mask[..., 0]
        else:
            segm_mask_gray = segm_mask

        segm_mask_model = cv2.resize(segm_mask_gray, (512, 384))  
        print(f"Model input mask shape: {segm_mask_model.shape}")

        segm_mask = segm_mask_model
        print("Mask processed.")
    else:
        segm_mask = None
        print("No mask provided.")
        
    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        print(f"Preparing video chunk for _process_step (is_first_step={is_first_step})")
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        print("Calling model in _process_step...")
        result = model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            # segm_mask=torch.from_numpy(segm_mask)[None, None],
        )
        print("Model call in _process_step completed.")
        return result

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    fps, num_frames = extract_video_info(args.video_path)
    full_vid = read_video_from_path(args.video_path)
    start_frame = 0
    while True:
        video, end_frame = extract_frames(full_vid, FRAMES_INTERVAL, fps, start_frame, num_frames)
        if end_frame >= num_frames:
            break
        for i, frame in enumerate(video):
            if i % model.step == 0 and i != 0:
                print(f"Processing frames from {start_frame} to {end_frame}")
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                    grid_size=args.grid_size,
                    grid_query_frame=args.grid_query_frame,
                )
                is_first_step = False
            window_frames.append(frame)
        # Processing the final video frames in case video length is not a multiple of model.step
        pred_tracks, pred_visibility = _process_step(
            window_frames[-(i % model.step) - model.step - 1 :],
            is_first_step,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
        )

        print("Tracks are computed")

        # save a video with predicted tracks
        seq_name = args.video_path.split("/")[-1]
        video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        vis.visualize(
            video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=f"{seq_name}_{start_frame}_{end_frame}.mp4"
        )
        start_frame = end_frame
        print(f"Processed frames from {start_frame} to {end_frame}")
    print(f"Processed all frames from {start_frame} to {end_frame}")