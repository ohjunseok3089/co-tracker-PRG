import os
import torch
import argparse
import imageio.v3 as iio

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
import imageio
import numpy as np

from cotracker.predictor import CoTrackerPredictor
from cotracker.predictor import CoTrackerOnlinePredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
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
    args = parser.parse_args()
    
    grid_size = 30
    grid_query_frame = 0
    
    if args.video_path is None:
        video_path = "../egocom/720p/EGOCOM/720p/5min_parts/vid_001__day_1__con_1__person_1_part1.MP4"
        args.video_path = video_path
        
    video = read_video_from_path(args.video_path)
    # Get FPS using imageio
    reader = imageio.get_reader(args.video_path)
    fps = reader.get_meta_data()['fps']
    reader.close()

    frames_per_minute = int(fps * 60)
    video = video[:frames_per_minute]
    # video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    if torch.cuda.is_available():
        model = model.cuda()
        # video = video.cuda()
        
    window_frames = []
    
    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
        
    for i, frame in enumerate(
        video
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
            )
            is_first_step = False
        window_frames.append(frame)
    
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
    )
    pred_tracks, pred_visibility = model(video, grid_size=30)
    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=grid_query_frame
    )