import cv2
import os

def extract_first_frame(video_path, output_dir):
    try:
        video_filename = os.path.basename(video_path)
        frame_filename = os.path.splitext(video_filename)[0] + ".jpg"
        output_filepath = os.path.join(output_dir, frame_filename)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        ret, frame = cap.read()

        if ret:
            cv2.imwrite(output_filepath, frame)
            print(f"Successfully saved frame for '{video_filename}' to '{output_filepath}'")
        else:
            print(f"Error: Could not read the first frame from '{video_filename}'")

        cap.release()

    except Exception as e:
        print(f"An error occurred while processing {video_path}: {e}")


def main():
    video_directory = r"C:\Users\ohjun\Documents\Github\egocom_asset"
    
    output_directory_name = "extracted_frames"
    output_directory_path = os.path.join(video_directory, output_directory_name)

    if not os.path.isdir(video_directory):
        print(f"Error: The specified directory does not exist: {video_directory}")
        return

    try:
        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)
            print(f"Created output directory: {output_directory_path}")
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return


    print("\nStarting frame extraction process...")
    for filename in os.listdir(video_directory):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_video_path = os.path.join(video_directory, filename)
            extract_first_frame(full_video_path, output_directory_path)

    print("\nFrame extraction complete.")


if __name__ == "__main__":
    main()