import cv2
import numpy as np

def detect_green_margin_intrusion(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Target color range around RGB(4, 215, 18)
    target_bgr = np.array([[[18, 215, 4]]], dtype=np.uint8)  # RGB to BGR conversion
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    target_h, target_s, target_v = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
    
    # Define tolerance ranges for target color
    h_range = 15
    s_range = 15
    v_range = 30
    
    h_min_target = max(0, target_h - h_range)
    h_max_target = min(179, target_h + h_range)
    s_min_target = max(0, target_s - s_range)
    s_max_target = min(255, target_s + s_range)
    v_min_target = max(0, target_v - v_range)
    v_max_target = min(255, target_v + v_range)
    
    lower_target = np.array([h_min_target, s_min_target, v_min_target], dtype=np.uint8)
    upper_target = np.array([h_max_target, s_max_target, v_max_target], dtype=np.uint8)
    target_mask = cv2.inRange(hsv_image, lower_target, upper_target)
    
    lower_general = np.array([35, 40, 40], dtype=np.uint8)
    upper_general = np.array([85, 255, 255], dtype=np.uint8)
    general_mask = cv2.inRange(hsv_image, lower_general, upper_general)
    
    # Target green + general green
    combined_mask = cv2.bitwise_or(target_mask, general_mask)
    
    # Remove noise
    kernel = np.ones((2,2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Detect content area (non-white regions)
    lower_non_white = np.array([0, 0, 0])
    upper_non_white = np.array([160, 160, 160])  # Stricter threshold for content detection
    content_mask = cv2.inRange(image, lower_non_white, upper_non_white)
    
    # Remove noise 
    kernel_content = np.ones((15,15), np.uint8)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel_content)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel_content)
    
    kernel_small = np.ones((5,5), np.uint8)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Find bounding box of content area
    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    content_x, content_y, content_w, content_h = cv2.boundingRect(largest_contour)
    
    # Add padding to content bounding box to prevent false positives at boundaries
    padding = 25  # 25 pixels padding in each direction
    content_x = max(0, content_x - padding)
    content_y = max(0, content_y - padding)
    content_w = min(w - content_x, content_w + 2 * padding)
    content_h = min(h - content_y, content_h + 2 * padding)
    
    # Define margin areas (outside padded content bounding box)
    margin_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Top margin
    if content_y > 0:
        margin_mask[0:content_y, :] = 255
    
    # Bottom margin
    if content_y + content_h < h:
        margin_mask[content_y + content_h:h, :] = 255
    
    # Left margin
    if content_x > 0:
        margin_mask[:, 0:content_x] = 255
    
    # Right margin
    if content_x + content_w < w:
        margin_mask[:, content_x + content_w:w] = 255
    
    # Define white margin areas (margin ∩ white pixels)
    lower_white = np.array([120, 120, 120])  # Generous white threshold
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    
    real_white_margin = cv2.bitwise_and(margin_mask, white_mask)
    
    # Find green pixels in actual white margins
    green_in_real_margin = cv2.bitwise_and(combined_mask, real_white_margin)
    
    # Calculate green pixel counts
    green_pixels_in_margin = cv2.countNonZero(green_in_real_margin)
    total_green_pixels = cv2.countNonZero(combined_mask)
    
    if green_pixels_in_margin == 0:
        video_center_x = content_x + content_w // 2
        video_center_y = content_y + content_h // 2
        return {
            "has_green_in_margin": False,
            "green_pixels_in_margin": 0,
            "total_green_pixels": total_green_pixels,
            "closest_distance": None,
            "closest_position": None,
            "content_bbox": (content_x, content_y, content_w, content_h),
            "video_center": (video_center_x, video_center_y),
            "angle_to_target": None
        }
    
    # Find closest green pixel to center
    green_positions = np.column_stack(np.where(green_in_real_margin > 0))
    
    # Convert (row, col) to (x, y) and calculate distances to center
    distances = []
    positions = []
    
    for row, col in green_positions:
        x, y = int(col), int(row)  # OpenCV coordinate conversion
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        distances.append(distance)
        positions.append((x, y))
    
    # Find closest pixel
    min_distance_idx = np.argmin(distances)
    closest_distance = distances[min_distance_idx]
    closest_position = positions[min_distance_idx]
    
    # Calculate viewing angle delta to look at the intrusion point
    video_fov_degrees = 104.0  # Video field of view in degrees
    video_center_x = content_x + content_w // 2
    video_center_y = content_y + content_h // 2
    
    # Calculate horizontal and vertical offsets from video center
    horizontal_offset = closest_position[0] - video_center_x
    vertical_offset = closest_position[1] - video_center_y
    
    # Convert pixel offsets to angles
    # Horizontal: positive = right, negative = left
    horizontal_angle_degrees = (horizontal_offset / (content_w / 2)) * (video_fov_degrees / 2)
    
    # Vertical: assume same FOV ratio (might need adjustment based on aspect ratio)
    vertical_fov_degrees = video_fov_degrees * (content_h / content_w)
    vertical_angle_degrees = (vertical_offset / (content_h / 2)) * (vertical_fov_degrees / 2)
    
    # Convert to radians
    horizontal_angle_radians = np.radians(horizontal_angle_degrees)
    vertical_angle_radians = np.radians(vertical_angle_degrees)
    
    # Calculate total angular distance (for reference)
    total_angle_degrees = np.sqrt(horizontal_angle_degrees**2 + vertical_angle_degrees**2)
    total_angle_radians = np.radians(total_angle_degrees)
    
    # Create visualization
    result_image = image.copy()
    
    # Show content area (yellow border)
    cv2.rectangle(result_image, (content_x, content_y), (content_x + content_w, content_y + content_h), (0, 255, 255), 3)
    
    # Show margin areas (light blue)
    result_image[real_white_margin > 0] = [255, 200, 150]
    
    # Highlight green pixels (red)
    result_image[green_in_real_margin > 0] = [0, 0, 255]
    
    # Mark image center point (white cross)
    cv2.drawMarker(result_image, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
    
    # Mark video center point (cyan cross)
    cv2.drawMarker(result_image, (video_center_x, video_center_y), (255, 255, 0), cv2.MARKER_CROSS, 15, 2)
    
    # Mark closest pixel (yellow circle)
    cv2.circle(result_image, closest_position, 8, (0, 255, 255), -1)
    
    # Draw distance line from video center to intrusion point (green)
    cv2.line(result_image, (video_center_x, video_center_y), closest_position, (0, 255, 0), 2)
    
    cv2.imwrite('debug_final_real_margin_result.jpg', result_image)
    
    result = {
        "has_green_in_margin": True,
        "green_pixels_in_margin": green_pixels_in_margin,
        "total_green_pixels": total_green_pixels,
        "closest_distance": closest_distance,
        "closest_position": closest_position,
        "content_bbox": (content_x, content_y, content_w, content_h),
        "video_center": (video_center_x, video_center_y),
        "angle_to_target": {
            "horizontal_degrees": horizontal_angle_degrees,
            "vertical_degrees": vertical_angle_degrees,
            "horizontal_radians": horizontal_angle_radians,
            "vertical_radians": vertical_angle_radians,
            "total_degrees": total_angle_degrees,
            "total_radians": total_angle_radians
        }
    }
    
    return result


def get_camera_adjustment_angles(image_path: str, video_fov_degrees: float = 104.0):
    result = detect_green_margin_intrusion(image_path)
    
    if result and result["has_green_in_margin"] and result["angle_to_target"]:
        return (
            result["angle_to_target"]["horizontal_radians"], 
            result["angle_to_target"]["vertical_radians"]
        )
    
    return (None, None)


if __name__ == "__main__":
    # Test the angle calculation
    result = detect_green_margin_intrusion("image.png")
    if result:
        print(f"Green intrusion detected: {'Yes' if result['has_green_in_margin'] else 'No'}")
        if result['has_green_in_margin']:
            angles = result['angle_to_target']
            print(f"Intruding pixels: {result['green_pixels_in_margin']:,}")
            print(f"Video center: {result['video_center']}")
            print(f"Target position: {result['closest_position']}")
            print(f"Camera adjustment needed:")
            print(f"  Horizontal: {angles['horizontal_degrees']:.2f}° ({angles['horizontal_radians']:.4f} rad)")
            print(f"  Vertical: {angles['vertical_degrees']:.2f}° ({angles['vertical_radians']:.4f} rad)")
            print(f"  Total angle: {angles['total_degrees']:.2f}° ({angles['total_radians']:.4f} rad)")
        else:
            print("No green pixels found in white margins.")
    else:
        print("Error: Image processing failed")

