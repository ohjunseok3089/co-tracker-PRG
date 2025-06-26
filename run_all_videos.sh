#!/bin/bash

# Set the base directory path
BASE_DIR="/mas/robots/prg-egocom/egocom-pre-process/egocom_asset/"
MASK_DIR="$BASE_DIR/masked_values"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# Check if the mask directory exists
if [ ! -d "$MASK_DIR" ]; then
    echo "Error: Directory $MASK_DIR does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p processed_videos

echo "Starting batch processing of videos in parallel (4 GPUs)..."
echo "Base directory: $BASE_DIR"
echo "Mask directory: $MASK_DIR"
echo "================================"

# Gather all .mp4 files into an array
mapfile -t video_files < <(find "$BASE_DIR" -maxdepth 1 -name '*.mp4' | sort)
num_videos=${#video_files[@]}

if [ "$num_videos" -eq 0 ]; then
    echo "No .mp4 files found in $BASE_DIR"
    exit 1
fi

# Split video files into 4 groups for 4 GPUs
num_gpus=4

declare -a groups
for ((i=0; i<num_gpus; i++)); do
    groups[$i]=""
done

for ((i=0; i<num_videos; i++)); do
    gpu_index=$((i % num_gpus))
    groups[$gpu_index]="${groups[$gpu_index]} ${video_files[$i]}"
done

# Function to process a group of videos (to be run in each screen session)
process_group() {
    local gpu_id=$1
    shift
    local videos=("$@")
    local count=0
    local failed=0
    for video_file in "${videos[@]}"; do
        video_basename=$(basename "$video_file" .mp4)
        mask_file="$MASK_DIR/${video_basename}.png"
        echo "[GPU $gpu_id] Processing video: $video_basename"
        echo "[GPU $gpu_id] Video path: $video_file"
        echo "[GPU $gpu_id] Mask path: $mask_file"
        if [ ! -f "$mask_file" ]; then
            echo "[GPU $gpu_id] Warning: Mask file $mask_file not found for video $video_basename"
            echo "[GPU $gpu_id] Skipping this video..."
            echo "--------------------------------"
            ((failed++))
            continue
        fi
        echo "[GPU $gpu_id] Running CoTracker on $video_basename..."
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
            --video_path "$video_file" \
            --mask_path "$mask_file" \
            --grid_size 30 \
            --grid_query_frame 0
        if [ $? -eq 0 ]; then
            echo "[GPU $gpu_id] ✓ Successfully processed $video_basename"
            ((count++))
        else
            echo "[GPU $gpu_id] ✗ Failed to process $video_basename"
            ((failed++))
        fi
        echo "--------------------------------"
    done
    echo "[GPU $gpu_id] Done. Successfully processed: $count, Failed: $failed"
}

# Launch 4 parallel screen sessions, one for each GPU
for ((gpu=0; gpu<num_gpus; gpu++)); do
    group_videos=(${groups[$gpu]})
    if [ ${#group_videos[@]} -eq 0 ]; then
        continue
    fi
    screen -dmS cotracker_gpu$gpu bash -c "$(declare -f process_group); process_group $gpu ${group_videos[@]}"
    echo "Launched screen session 'cotracker_gpu$gpu' for GPU $gpu with ${#group_videos[@]} videos."
done

echo "================================"
echo "Batch processing launched in 4 parallel screen sessions!"
echo "Use 'screen -ls' to see running sessions. Attach with 'screen -r cotracker_gpuX' (X=0,1,2,3)."
echo "Output videos saved in: ./processed_videos/" 