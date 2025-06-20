# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
import imageio.v3 as iio
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


class MultiGPUCoTrackerProcessor:
    def __init__(self, checkpoint=None, gpu_ids=None, batch_size=1):
        self.gpu_ids = gpu_ids if gpu_ids else list(range(torch.cuda.device_count()))
        self.num_gpus = len(self.gpu_ids)
        self.batch_size = batch_size
        self.models = {}
        self.checkpoint = checkpoint
        
        print(f"Initializing MultiGPU processor with {self.num_gpus} GPUs: {self.gpu_ids}")
        
        # Initialize models on each GPU
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize CoTracker models on each GPU."""
        for gpu_id in self.gpu_ids:
            device = torch.device(f"cuda:{gpu_id}")
            
            if self.checkpoint is not None:
                model = CoTrackerOnlinePredictor(checkpoint=self.checkpoint)
            else:
                model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
            
            model = model.to(device)
            model.eval()
            self.models[gpu_id] = model
            print(f"Model initialized on GPU {gpu_id}")
    
    def process_chunk_on_gpu(self, gpu_id, video_chunks, args):
        """Process video chunks on a specific GPU."""
        device = torch.device(f"cuda:{gpu_id}")
        model = self.models[gpu_id]
        results = []
        
        print(f"GPU {gpu_id}: Processing {len(video_chunks)} chunks")
        
        for chunk_idx, (chunk_frames, is_first_step) in enumerate(video_chunks):
            try:
                # Move frames to GPU
                video_chunk = torch.tensor(
                    np.stack(chunk_frames), device=device
                ).float().permute(0, 3, 1, 2)[None]
                
                # Process chunk
                with torch.no_grad():
                    pred_tracks, pred_visibility = model(
                        video_chunk,
                        is_first_step=is_first_step,
                        grid_size=args.grid_size,
                        grid_query_frame=args.grid_query_frame,
                    )
                
                # Move results to CPU to save GPU memory
                results.append({
                    'chunk_idx': chunk_idx,
                    'tracks': pred_tracks.cpu(),
                    'visibility': pred_visibility.cpu(),
                    'frames': chunk_frames
                })
                
                print(f"GPU {gpu_id}: Completed chunk {chunk_idx + 1}/{len(video_chunks)}")
                
            except Exception as e:
                print(f"Error processing chunk {chunk_idx} on GPU {gpu_id}: {e}")
                continue
        
        return gpu_id, results
    
    def process_video_parallel(self, video_path, args):
        """Process video using multiple GPUs in parallel."""
        print("Loading video frames...")
        
        # Load all frames
        all_frames = []
        for frame in iio.imiter(video_path, plugin="FFMPEG"):
            all_frames.append(frame)
        
        total_frames = len(all_frames)
        print(f"Loaded {total_frames} frames")
        
        # Get model step size
        model_step = self.models[self.gpu_ids[0]].step
        
        # Create chunks for processing
        chunks = []
        is_first_step = True
        
        for i in range(0, total_frames, model_step):
            end_idx = min(i + model_step * 2, total_frames)
            chunk_frames = all_frames[i:end_idx]
            
            if len(chunk_frames) >= model_step:
                chunks.append((chunk_frames, is_first_step))
                is_first_step = False
        
        print(f"Created {len(chunks)} chunks for processing")
        
        # Distribute chunks across GPUs
        chunks_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, chunk in enumerate(chunks):
            gpu_idx = i % self.num_gpus
            chunks_per_gpu[gpu_idx].append(chunk)
        
        # Process chunks in parallel
        print("Starting parallel processing...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            for i, gpu_id in enumerate(self.gpu_ids):
                if chunks_per_gpu[i]:  # Only submit if there are chunks to process
                    future = executor.submit(
                        self.process_chunk_on_gpu, 
                        gpu_id, 
                        chunks_per_gpu[i], 
                        args
                    )
                    futures.append(future)
            
            # Collect results
            all_results = {}
            for future in as_completed(futures):
                gpu_id, results = future.result()
                all_results[gpu_id] = results
        
        processing_time = time.time() - start_time
        print(f"Parallel processing completed in {processing_time:.2f} seconds")
        
        # Merge results from all GPUs
        return self._merge_results(all_results, all_frames)
    
    def _merge_results(self, all_results, all_frames):
        """Merge results from all GPUs in the correct order."""
        print("Merging results from all GPUs...")
        
        # Flatten and sort results by chunk index
        sorted_results = []
        for gpu_results in all_results.values():
            sorted_results.extend(gpu_results)
        
        sorted_results.sort(key=lambda x: x['chunk_idx'])
        
        if not sorted_results:
            raise ValueError("No results to merge")
        
        # Use the last chunk's results as final output
        final_result = sorted_results[-1]
        
        return final_result['tracks'], final_result['visibility'], all_frames


def setup_distributed(rank, world_size):
    """Setup distributed processing."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed processing."""
    dist.destroy_process_group()


def get_device_setup(args):
    """Setup device configuration for single or multi-GPU."""
    if args.cpu:
        return torch.device("cpu"), False, None
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu"), False, None
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    if args.gpu_ids:
        # Use specific GPU IDs
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        if max(gpu_ids) >= num_gpus:
            raise ValueError(f"GPU ID {max(gpu_ids)} not available. Only {num_gpus} GPUs found.")
        device = torch.device(f"cuda:{gpu_ids[0]}")
        use_multi_gpu = len(gpu_ids) > 1
        print(f"Using GPU(s): {gpu_ids}")
    else:
        # Use all available GPUs
        gpu_ids = list(range(num_gpus))
        device = torch.device("cuda:0")
        use_multi_gpu = num_gpus > 1
        print(f"Using all {num_gpus} GPU(s)")
    
    return device, use_multi_gpu, gpu_ids


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
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
    # Multi-GPU arguments
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2'). If not specified, uses all available GPUs",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for multi-GPU processing",
    )
    parser.add_argument(
        "--use_advanced_multigpu",
        action="store_true",
        help="Use advanced multi-GPU processing with chunk distribution",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    # Setup device configuration
    device, use_multi_gpu, gpu_ids = get_device_setup(args)

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
    else:
        segm_mask = None

    # Choose processing method based on arguments
    if use_multi_gpu and args.use_advanced_multigpu:
        print("Using advanced multi-GPU processing")
        processor = MultiGPUCoTrackerProcessor(
            checkpoint=args.checkpoint,
            gpu_ids=gpu_ids,
            batch_size=args.batch_size
        )
        
        pred_tracks, pred_visibility, window_frames = processor.process_video_parallel(
            args.video_path, args
        )
        
    else:
        print("Using standard processing (single GPU or DataParallel)")
        
        # Load model
        if args.checkpoint is not None:
            model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        
        model = model.to(device)
        
        # Enable DataParallel for multi-GPU
        if use_multi_gpu:
            print("Enabling multi-GPU processing with DataParallel")
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        
        # Standard processing logic
        def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
            model_step = model.module.step if hasattr(model, 'module') else model.step
            
            video_chunk = (
                torch.tensor(
                    np.stack(window_frames[-model_step * 2 :]), device=device
                )
                .float()
                .permute(0, 3, 1, 2)[None]
            )
            
            return model(
                video_chunk,
                is_first_step=is_first_step,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
            )

        # Process video
        window_frames = []
        is_first_step = True
        model_step = model.module.step if hasattr(model, 'module') else model.step
        
        for i, frame in enumerate(iio.imiter(args.video_path, plugin="FFMPEG")):
            if i % model_step == 0 and i != 0:
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                    grid_size=args.grid_size,
                    grid_query_frame=args.grid_query_frame,
                )
                is_first_step = False
            window_frames.append(frame)
        
        # Process final frames
        pred_tracks, pred_visibility = _process_step(
            window_frames[-(i % model_step) - model_step - 1 :],
            is_first_step,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
        )

    print("Tracks are computed")

    # Save video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=device).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./processed_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=seq_name
    ) 