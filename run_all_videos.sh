#!/bin/bash

# Set the base directory path
BASE_DIR="assets/egocom_asset"
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

echo "Starting batch processing of videos..."
echo "Base directory: $BASE_DIR"
echo "Mask directory: $MASK_DIR"
echo "================================"

# Counter for processed videos
count=0
failed=0

# Process each .mp4 file in the base directory
for video_file in "$BASE_DIR"/*.mp4; do
    # Check if any .mp4 files exist
    if [ ! -f "$video_file" ]; then
        echo "No .mp4 files found in $BASE_DIR"
        exit 1
    fi
    
    # Extract the base filename without extension
    video_basename=$(basename "$video_file" .mp4)
    
    # Construct the corresponding mask file path
    mask_file="$MASK_DIR/${video_basename}.png"
    
    echo "Processing video: $video_basename"
    echo "Video path: $video_file"
    echo "Mask path: $mask_file"
    
    # Check if the corresponding mask file exists
    if [ ! -f "$mask_file" ]; then
        echo "Warning: Mask file $mask_file not found for video $video_basename"
        echo "Skipping this video..."
        echo "--------------------------------"
        ((failed++))
        continue
    fi
    
    # Run the main.py script with the video and mask
    echo "Running CoTracker on $video_basename..."
    python main.py \
        --video_path "$video_file" \
        --mask_path "$mask_file" \
        --grid_size 30 \
        --grid_query_frame 0
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $video_basename"
        ((count++))
    else
        echo "✗ Failed to process $video_basename"
        ((failed++))
    fi
    
    echo "--------------------------------"
done

echo "================================"
echo "Batch processing completed!"
echo "Successfully processed: $count videos"
echo "Failed: $failed videos"
echo "Output videos saved in: ./processed_videos/" 