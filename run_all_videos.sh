#!/bin/bash

# Set the base directory path
BASE_DIR="/mas/robots/prg-egocom/EGOCOM/720p/5min_parts/dataset/"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p processed_videos

echo "Starting batch processing of videos in parallel (4 GPUs)..."
echo "Base directory: $BASE_DIR"
echo "================================"

# Gather all video files into an array (both .mp4 and .MP4)
mapfile -t video_files < <(find "$BASE_DIR" -maxdepth 1 \( -name '*.mp4' -o -name '*.MP4' \) | sort)
num_videos=${#video_files[@]}

if [ "$num_videos" -eq 0 ]; then
    echo "No video files (.mp4 or .MP4) found in $BASE_DIR"
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
        echo "[GPU $gpu_id] Processing video: $video_basename"
        echo "[GPU $gpu_id] Video path: $video_file"

        echo "[GPU $gpu_id] Running CoTracker on $video_basename..."
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
            --video_path "$video_file" \
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

# Launch 4 parallel screen sessions, one for each GPU, using temp scripts for robust argument passing and logging
for ((gpu=0; gpu<num_gpus; gpu++)); do
    group_videos=(${groups[$gpu]})
    if [ ${#group_videos[@]} -eq 0 ]; then
        continue
    fi
    temp_script="cotracker_gpu${gpu}_run.sh"
    echo "#!/bin/bash" > $temp_script
    echo "count=0" >> $temp_script
    echo "failed=0" >> $temp_script
    echo "" >> $temp_script
    for video_file in "${group_videos[@]}"; do
        # Properly escape the video file path for the script
        escaped_video_file=$(printf '%q' "$video_file")
        echo "video_file=$escaped_video_file" >> $temp_script
        echo "video_basename=\$(basename \"\$video_file\")" >> $temp_script
        echo "video_basename=\${video_basename%.*}" >> $temp_script
        echo "echo \"[GPU $gpu] Processing video: \$video_basename\"" >> $temp_script
        echo "echo \"[GPU $gpu] Video path: \$video_file\"" >> $temp_script
        echo "echo \"[GPU $gpu] Running CoTracker on \$video_basename...\"" >> $temp_script
        echo "CUDA_VISIBLE_DEVICES=$gpu python main.py --video_path \"\$video_file\" --grid_size 30 --grid_query_frame 0" >> $temp_script
        echo "exit_code=\$?" >> $temp_script
        echo "if [ \$exit_code -eq 0 ]; then" >> $temp_script
        echo "  echo \"[GPU $gpu] ✓ Successfully processed \$video_basename\"" >> $temp_script
        echo "  ((count++))" >> $temp_script
        echo "else" >> $temp_script
        echo "  echo \"[GPU $gpu] ✗ Failed to process \$video_basename (exit code: \$exit_code)\"" >> $temp_script
        echo "  if [ \$exit_code -eq 1 ]; then" >> $temp_script
        echo "    echo \"[GPU $gpu] → Video appears to be corrupted, continuing with next video...\"" >> $temp_script
        echo "  fi" >> $temp_script
        echo "  ((failed++))" >> $temp_script
        echo "fi" >> $temp_script
        echo "echo \"--------------------------------\"" >> $temp_script
    done
    echo "echo \"[GPU $gpu] Done. Successfully processed: \$count, Failed: \$failed\"" >> $temp_script
    chmod +x $temp_script
    screen -dmS cotracker_gpu$gpu bash -c "./$temp_script &> cotracker_gpu${gpu}.log"
    echo "Launched screen session 'cotracker_gpu$gpu' for GPU $gpu with ${#group_videos[@]} videos. Log: cotracker_gpu${gpu}.log"
done

echo "================================"
echo "Batch processing launched in 4 parallel screen sessions!"
echo "Use 'screen -ls' to see running sessions. Attach with 'screen -r cotracker_gpuX' (X=0,1,2,3)."
echo "Output videos saved in: ./processed_videos/" 
