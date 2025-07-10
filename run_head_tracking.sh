#!/bin/bash

# Red Circle Head Movement Tracking Script
# Detects RGB(255,28,48) circles and calculates head movements

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VIDEO_PATH=""
OUTPUT_PATH=""
TARGET_RADIUS=3
FPS_OVERRIDE=""

show_help() {
    cat << EOF
Usage: $0 [OPTIONS] <video_path>

Detect red circles and calculate head movements from video files.

ARGUMENTS:
    video_path          Path to input video file

OPTIONS:
    -o, --output PATH   Output JSON file path (default: auto-generated)
    -r, --radius NUM    Expected radius of red circles (default: 3)
    -f, --fps NUM       Override video FPS if metadata is incorrect
    -h, --help          Show this help message

EXAMPLES:
    # Basic usage - auto-generate output filename
    $0 /path/to/video.mp4
    
    # Specify custom output path
    $0 -o results.json /path/to/video.mp4
    
    # Override FPS for videos with incorrect metadata
    $0 -f 30 /path/to/video.mp4
    
    # Custom circle radius and output path
    $0 -r 4 -o head_tracking.json /path/to/video.mp4

OUTPUT FORMAT:
    The script outputs JSON with frame-by-frame data including:
    - Red circle position (x, y coordinates and radius)
    - Head movement in both horizontal and vertical directions:
      * Horizontal: Yaw movement (left/right head turning)
      * Vertical: Pitch movement (up/down head tilting)
    - Timestamps and frame indices
    - Detection and movement statistics

HEAD MOVEMENT CALCULATION:
    - Horizontal (Yaw): left/right head rotation based on red dot x-movement
    - Vertical (Pitch): up/down head tilting based on red dot y-movement
    - Inverse relationship: red dot moves right = person turned left
    - Field of view: 104° horizontal, calculated vertical based on aspect ratio
    - Output in both radians and degrees

REQUIREMENTS:
    - Python 3.x with OpenCV (cv2), numpy
    - Video file with RGB(255,28,48) colored circles to track
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -r|--radius)
            TARGET_RADIUS="$2"
            shift 2
            ;;
        -f|--fps)
            FPS_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$VIDEO_PATH" ]; then
                VIDEO_PATH="$1"
            else
                echo "Error: Multiple video paths specified"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if video path was provided
if [ -z "$VIDEO_PATH" ]; then
    echo "Error: Video path is required"
    show_help
    exit 1
fi

# Check if video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

# Construct python command
PYTHON_CMD="python3 \"$SCRIPT_DIR/track_head_movement.py\""

# Add arguments
PYTHON_CMD="$PYTHON_CMD \"$VIDEO_PATH\""

if [ ! -z "$OUTPUT_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD -o \"$OUTPUT_PATH\""
fi

if [ ! -z "$TARGET_RADIUS" ]; then
    PYTHON_CMD="$PYTHON_CMD -r $TARGET_RADIUS"
fi

if [ ! -z "$FPS_OVERRIDE" ]; then
    PYTHON_CMD="$PYTHON_CMD -f $FPS_OVERRIDE"
fi

# Display command information
echo "Head Movement Tracking - Red Circle Detection"
echo "=============================================="
echo "Video: $VIDEO_PATH"
echo "Target radius: $TARGET_RADIUS pixels"
if [ ! -z "$OUTPUT_PATH" ]; then
    echo "Output: $OUTPUT_PATH"
else
    echo "Output: Auto-generated filename"
fi
if [ ! -z "$FPS_OVERRIDE" ]; then
    echo "FPS override: $FPS_OVERRIDE"
fi
echo ""

# Run the python script
echo "Executing: $PYTHON_CMD"
echo ""
eval $PYTHON_CMD 