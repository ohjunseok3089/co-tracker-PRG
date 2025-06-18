import os
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
import imageio

from cotracker.predictor import CoTrackerPredictor


if __name__ == "__main__":
    video = read_video_from_path("../egocom/720p/EGOCOM/720p/5min_parts/vid_001__day_1__con_1__person_1_part1.MP4")
    # Get FPS using imageio
    reader = imageio.get_reader('./assets/vid_001__day_1__con_1__person_1_part1.MP4')
    fps = reader.get_meta_data()['fps']
    reader.close()

    frames_per_minute = int(fps * 60)
    video = video[:frames_per_minute]
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/scaled_offline.pth'
        )
    )
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
        
    pred_tracks, pred_visibility = model(video, grid_size=30)
    vis = Visualizer(save_dir='./videos', pad_value=100)
    vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser');