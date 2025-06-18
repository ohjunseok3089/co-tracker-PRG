import os
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML

video = read_video_from_path("./data/vid_096__day_2__con_6__person_1.MP4")