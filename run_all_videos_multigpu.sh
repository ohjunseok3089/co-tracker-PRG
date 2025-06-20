#!/bin/bash

# Advanced Multi-GPU Video Processing Script for CoTracker
# Uses main_advanced_multigpu.py for optimal GPU utilization

# Configuration
BASE_DIR="assets/egocom_asset"
MASK_DIR="$BASE_DIR/masked_values"
OUTPUT_DIR="processed_videos"
LOG_DIR="processing_logs"

# Multi-GPU settings
DEFAULT_GPU_IDS="0,1,2,3"  # Adjust based on your available GPUs
BATCH_SIZE=2
GRID_SIZE=30
GRID_QUERY_FRAME=0

# Advanced processing options
USE_ADVANCED_MULTIGPU=true
PARALLEL_VIDEOS=1  # Number of videos to process simultaneously
MAX_RETRIES=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to get available GPUs
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        print_status $BLUE "Detected $gpu_count GPU(s)"
        
        if [ $gpu_count -eq 0 ]; then
            print_status $YELLOW "No GPUs detected, will use CPU"
            USE_CPU=true
        else
            # Generate GPU IDs string based on available GPUs
            available_gpus=$(seq -s, 0 $((gpu_count-1)))
            print_status $BLUE "Available GPUs: $available_gpus"
        fi
    else
        print_status $YELLOW "nvidia-smi not found, assuming CPU mode"
        USE_CPU=true
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status $BLUE "Checking prerequisites..."
    
    # Check if Python script exists
    if [ ! -f "main_advanced_multigpu.py" ]; then
        print_status $RED "Error: main_advanced_multigpu.py not found!"
        exit 1
    fi
    
    # Check directories
    if [ ! -d "$BASE_DIR" ]; then
        print_status $RED "Error: Directory $BASE_DIR does not exist!"
        exit 1
    fi
    
    if [ ! -d "$MASK_DIR" ]; then
        print_status $RED "Error: Directory $MASK_DIR does not exist!"
        exit 1
    fi
    
    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    
    print_status $GREEN "Prerequisites check passed"
}

# Function to process a single video
process_video() {
    local video_file=$1
    local mask_file=$2
    local video_basename=$3
    local gpu_ids=$4
    local log_file="$LOG_DIR/${video_basename}_$(date +%Y%m%d_%H%M%S).log"
    
    print_status $BLUE "Processing: $video_basename"
    echo "Video: $video_file" >> "$log_file"
    echo "Mask: $mask_file" >> "$log_file"
    echo "GPU IDs: $gpu_ids" >> "$log_file"
    echo "Started: $(date)" >> "$log_file"
    
    # Build command
    cmd="python main_advanced_multigpu.py"
    cmd="$cmd --video_path \"$video_file\""
    cmd="$cmd --mask_path \"$mask_file\""
    cmd="$cmd --grid_size $GRID_SIZE"
    cmd="$cmd --grid_query_frame $GRID_QUERY_FRAME"
    cmd="$cmd --batch_size $BATCH_SIZE"
    
    if [ "$USE_CPU" != "true" ]; then
        cmd="$cmd --gpu_ids $gpu_ids"
    else
        cmd="$cmd --cpu"
    fi
    
    if [ "$USE_ADVANCED_MULTIGPU" = "true" ] && [ "$USE_CPU" != "true" ]; then
        cmd="$cmd --use_advanced_multigpu"
    fi
    
    echo "Command: $cmd" >> "$log_file"
    
    # Execute with timeout and logging
    local success=false
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ] && [ "$success" = "false" ]; do
        print_status $YELLOW "Attempt $attempt/$MAX_RETRIES for $video_basename"
        echo "Attempt $attempt started: $(date)" >> "$log_file"
        
        if timeout 3600 bash -c "$cmd" >> "$log_file" 2>&1; then
            success=true
            print_status $GREEN "✓ Successfully processed $video_basename"
            echo "Completed successfully: $(date)" >> "$log_file"
        else
            print_status $RED "✗ Attempt $attempt failed for $video_basename"
            echo "Attempt $attempt failed: $(date)" >> "$log_file"
            ((attempt++))
            
            if [ $attempt -le $MAX_RETRIES ]; then
                print_status $YELLOW "Retrying in 10 seconds..."
                sleep 10
            fi
        fi
    done
    
    if [ "$success" = "true" ]; then
        return 0
    else
        print_status $RED "✗ Failed to process $video_basename after $MAX_RETRIES attempts"
        return 1
    fi
}

# Function to process videos in parallel
process_videos_parallel() {
    local video_list=("$@")
    local total_videos=${#video_list[@]}
    local processed=0
    local failed=0
    local active_jobs=0
    local job_pids=()
    
    print_status $BLUE "Starting parallel processing of $total_videos videos"
    print_status $BLUE "Parallel jobs: $PARALLEL_VIDEOS"
    
    for video_info in "${video_list[@]}"; do
        # Wait if we have too many active jobs
        while [ $active_jobs -ge $PARALLEL_VIDEOS ]; do
            # Check completed jobs
            for i in "${!job_pids[@]}"; do
                local pid=${job_pids[i]}
                if ! kill -0 $pid 2>/dev/null; then
                    wait $pid
                    local exit_code=$?
                    unset job_pids[i]
                    ((active_jobs--))
                    
                    if [ $exit_code -eq 0 ]; then
                        ((processed++))
                    else
                        ((failed++))
                    fi
                fi
            done
            sleep 1
        done
        
        # Parse video info
        IFS='|' read -r video_file mask_file video_basename gpu_ids <<< "$video_info"
        
        # Start processing in background
        process_video "$video_file" "$mask_file" "$video_basename" "$gpu_ids" &
        local job_pid=$!
        job_pids+=($job_pid)
        ((active_jobs++))
        
        print_status $BLUE "Started job for $video_basename (PID: $job_pid)"
    done
    
    # Wait for remaining jobs
    print_status $BLUE "Waiting for remaining jobs to complete..."
    for pid in "${job_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            wait $pid
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                ((processed++))
            else
                ((failed++))
            fi
        fi
    done
    
    return $((processed << 8 | failed))
}

# Function to distribute GPUs across videos
distribute_gpus() {
    local video_files=("$@")
    local gpu_list=()
    local video_list=()
    
    if [ "$USE_CPU" != "true" ]; then
        IFS=',' read -ra gpu_array <<< "$available_gpus"
        local gpu_count=${#gpu_array[@]}
        
        # Create GPU distribution strategy
        if [ $gpu_count -ge $PARALLEL_VIDEOS ]; then
            # Assign different GPUs to each parallel job
            local gpus_per_job=$((gpu_count / PARALLEL_VIDEOS))
            for ((i=0; i<PARALLEL_VIDEOS; i++)); do
                local start_gpu=$((i * gpus_per_job))
                local end_gpu=$(( (i+1) * gpus_per_job - 1))
                if [ $i -eq $((PARALLEL_VIDEOS-1)) ]; then
                    end_gpu=$((gpu_count-1))  # Last job gets remaining GPUs
                fi
                
                local job_gpus=""
                for ((j=start_gpu; j<=end_gpu; j++)); do
                    if [ -n "$job_gpus" ]; then
                        job_gpus="$job_gpus,${gpu_array[j]}"
                    else
                        job_gpus="${gpu_array[j]}"
                    fi
                done
                gpu_list+=("$job_gpus")
            done
        else
            # Share GPUs across jobs
            for ((i=0; i<PARALLEL_VIDEOS; i++)); do
                gpu_list+=("$available_gpus")
            done
        fi
    else
        # CPU mode
        for ((i=0; i<PARALLEL_VIDEOS; i++)); do
            gpu_list+=("cpu")
        done
    fi
    
    # Create video processing list
    local gpu_idx=0
    for video_file in "${video_files[@]}"; do
        local video_basename=$(basename "$video_file" .mp4)
        local mask_file="$MASK_DIR/${video_basename}.png"
        
        if [ -f "$mask_file" ]; then
            local assigned_gpus=${gpu_list[$gpu_idx]}
            video_list+=("$video_file|$mask_file|$video_basename|$assigned_gpus")
            gpu_idx=$(( (gpu_idx + 1) % ${#gpu_list[@]} ))
        else
            print_status $YELLOW "Warning: Mask file not found for $video_basename, skipping"
        fi
    done
    
    printf '%s\n' "${video_list[@]}"
}

# Main function
main() {
    print_status $BLUE "=== CoTracker Advanced Multi-GPU Batch Processor ==="
    print_status $BLUE "$(date)"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu-ids)
                DEFAULT_GPU_IDS="$2"
                shift 2
                ;;
            --parallel-videos)
                PARALLEL_VIDEOS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --grid-size)
                GRID_SIZE="$2"
                shift 2
                ;;
            --cpu)
                USE_CPU=true
                shift
                ;;
            --no-advanced)
                USE_ADVANCED_MULTIGPU=false
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --gpu-ids GPUS        Comma-separated GPU IDs (default: 0,1,2,3)"
                echo "  --parallel-videos N   Number of videos to process in parallel (default: 1)"
                echo "  --batch-size N        Batch size for processing (default: 2)"
                echo "  --grid-size N         Grid size for tracking (default: 30)"
                echo "  --cpu                 Force CPU usage"
                echo "  --no-advanced         Disable advanced multi-GPU processing"
                echo "  --help                Show this help message"
                exit 0
                ;;
            *)
                print_status $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Setup
    detect_gpus
    check_prerequisites
    
    if [ "$USE_CPU" != "true" ]; then
        available_gpus=${DEFAULT_GPU_IDS}
    fi
    
    # Find all video files
    mapfile -t video_files < <(find "$BASE_DIR" -name "*.mp4" -type f)
    
    if [ ${#video_files[@]} -eq 0 ]; then
        print_status $RED "No .mp4 files found in $BASE_DIR"
        exit 1
    fi
    
    print_status $BLUE "Found ${#video_files[@]} video files"
    
    # Distribute GPUs and create processing list
    mapfile -t video_list < <(distribute_gpus "${video_files[@]}")
    
    if [ ${#video_list[@]} -eq 0 ]; then
        print_status $RED "No valid video-mask pairs found"
        exit 1
    fi
    
    print_status $BLUE "Processing ${#video_list[@]} valid video-mask pairs"
    
    # Show configuration
    print_status $BLUE "Configuration:"
    print_status $BLUE "  - Base directory: $BASE_DIR"
    print_status $BLUE "  - Output directory: $OUTPUT_DIR"
    print_status $BLUE "  - Parallel videos: $PARALLEL_VIDEOS"
    print_status $BLUE "  - Batch size: $BATCH_SIZE"
    print_status $BLUE "  - Grid size: $GRID_SIZE"
    print_status $BLUE "  - Advanced multi-GPU: $USE_ADVANCED_MULTIGPU"
    if [ "$USE_CPU" != "true" ]; then
        print_status $BLUE "  - GPU IDs: $available_gpus"
    else
        print_status $BLUE "  - Using CPU"
    fi
    
    # Start processing
    print_status $GREEN "Starting batch processing..."
    local start_time=$(date +%s)
    
    process_videos_parallel "${video_list[@]}"
    local result=$?
    local processed=$((result >> 8))
    local failed=$((result & 255))
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Final report
    print_status $GREEN "=== Processing Complete ==="
    print_status $GREEN "Total time: ${duration}s"
    print_status $GREEN "Successfully processed: $processed videos"
    if [ $failed -gt 0 ]; then
        print_status $RED "Failed: $failed videos"
    fi
    print_status $BLUE "Output videos saved in: $OUTPUT_DIR"
    print_status $BLUE "Processing logs saved in: $LOG_DIR"
    
    # Show GPU utilization summary if available
    if command -v nvidia-smi &> /dev/null && [ "$USE_CPU" != "true" ]; then
        print_status $BLUE "Final GPU status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    fi
}

# Run main function with all arguments
main "$@" 