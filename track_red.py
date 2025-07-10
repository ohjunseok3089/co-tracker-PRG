import cv2
import numpy as np

def detect_red_circle(image, target_radius: int = 3):
    """
    Simplified red circle detection for video processing.
    Returns the center and radius of the closest red circle to image center, or None if not found.
    
    Args:
        image: OpenCV image (BGR format)
        target_radius: Expected radius of circles to detect
    
    Returns:
        Tuple of (center_x, center_y, radius) or None if no circle found
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Convert target RGB (255,28,48) to BGR for OpenCV
    target_bgr = np.array([[[48, 28, 255]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    target_h, target_s, target_v = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
    
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define tolerance ranges for the specific red color RGB(255,28,48)
    h_range = 10
    s_range = 30
    v_range = 40
    
    h_min = max(0, target_h - h_range)
    h_max = min(179, target_h + h_range)
    s_min = max(0, target_s - s_range)
    s_max = min(255, target_s + s_range)
    v_min = max(0, target_v - v_range)
    v_max = min(255, target_v + v_range)
    
    # Create mask for target red color
    lower_red = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_red = np.array([h_max, s_max, v_max], dtype=np.uint8)
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    # Handle red wrap-around in HSV
    if target_h < 10:
        lower_red2 = np.array([max(0, target_h + 170), s_min, v_min], dtype=np.uint8)
        upper_red2 = np.array([179, s_max, v_max], dtype=np.uint8)
        red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask, red_mask2)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # Try multiple HoughCircles parameter sets
    param_sets = [
        {"param1": 30, "param2": 8, "minRadius": 1, "maxRadius": target_radius + 3},
        {"param1": 40, "param2": 10, "minRadius": max(1, target_radius - 1), "maxRadius": target_radius + 2},
        {"param1": 20, "param2": 5, "minRadius": 1, "maxRadius": target_radius + 5},
    ]
    
    circles = None
    for params in param_sets:
        circles = cv2.HoughCircles(
            red_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=target_radius * 2,
            param1=params['param1'],
            param2=params['param2'],
            minRadius=params['minRadius'],
            maxRadius=params['maxRadius']
        )
        if circles is not None:
            break
    
    # Alternative contour-based detection if HoughCircles fails
    if circles is None:
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        alternative_circles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.3:  # Roughly circular
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                if 1 <= radius <= target_radius + 5:
                    alternative_circles.append([center[0], center[1], radius])
        
        if alternative_circles:
            circles = np.array([[alternative_circles]], dtype=np.float32)
    
    if circles is None:
        return None
    
    # Find the circle with highest red ratio closest to center
    circles_rounded = np.around(circles).astype(np.uint16)
    circles_array = circles_rounded[0, :]
    
    best_circle = None
    best_score = 0
    
    for circle in circles_array:
        circle_center = (int(circle[0]), int(circle[1]))
        circle_radius = int(circle[2])
        
        # Verify the circle contains the target color
        mask_circle = np.zeros(red_mask.shape, dtype=np.uint8)
        cv2.circle(mask_circle, circle_center, circle_radius, (255,), -1)
        
        overlap = cv2.bitwise_and(red_mask, mask_circle)
        red_pixels_in_circle = cv2.countNonZero(overlap)
        total_pixels_in_circle = cv2.countNonZero(mask_circle)
        
        red_ratio = red_pixels_in_circle / max(1, total_pixels_in_circle)
        
        # Only accept circles with good red color ratio
        if red_ratio > 0.5:  # Slightly more lenient than before
            distance_from_center = np.sqrt((circle_center[0] - center_x)**2 + (circle_center[1] - center_y)**2)
            # Score combines red ratio and proximity to center (prefer closer circles)
            score = red_ratio * (1.0 / (1.0 + distance_from_center / 100.0))
            
            if score > best_score:
                best_score = score
                best_circle = (circle_center[0], circle_center[1], circle_radius)
    
    return best_circle


def calculate_head_movement(prev_red_pos, curr_red_pos, image_width, image_height, video_fov_degrees=104.0):
    """
    Calculate head movement based on red circle position change.
    
    Args:
        prev_red_pos: (x, y) of previous red circle position, or None
        curr_red_pos: (x, y) of current red circle position, or None  
        image_width: Width of the image
        image_height: Height of the image
        video_fov_degrees: Horizontal field of view in degrees
    
    Returns:
        Dict with horizontal and vertical movement data:
        {
            "horizontal": {"radians": float, "degrees": float},  # Yaw (left/right)
            "vertical": {"radians": float, "degrees": float}     # Pitch (up/down)
        }
        or None if calculation not possible
    """
    if prev_red_pos is None or curr_red_pos is None:
        return None
    
    # Calculate horizontal movement (positive = moved right, negative = moved left)
    horizontal_pixel_change = curr_red_pos[0] - prev_red_pos[0]
    
    # Calculate vertical movement (positive = moved down, negative = moved up)
    vertical_pixel_change = curr_red_pos[1] - prev_red_pos[1]
    
    # Convert pixel changes to angles
    # Horizontal FOV is given, calculate vertical FOV based on aspect ratio
    aspect_ratio = image_width / image_height
    vertical_fov_degrees = video_fov_degrees / aspect_ratio
    
    # Calculate horizontal movement (Yaw)
    # Note: If red dot moves right, person turned left (inverse relationship)
    horizontal_pixels_per_degree = image_width / video_fov_degrees
    horizontal_angle_degrees = -horizontal_pixel_change / horizontal_pixels_per_degree  # Negative for inverse
    horizontal_radians = np.radians(horizontal_angle_degrees)
    
    # Calculate vertical movement (Pitch)
    # Note: If red dot moves down, person tilted head up (inverse relationship)
    vertical_pixels_per_degree = image_height / vertical_fov_degrees
    vertical_angle_degrees = -vertical_pixel_change / vertical_pixels_per_degree  # Negative for inverse
    vertical_radians = np.radians(vertical_angle_degrees)
    
    return {
        "horizontal": {
            "radians": horizontal_radians,
            "degrees": horizontal_angle_degrees
        },
        "vertical": {
            "radians": vertical_radians,
            "degrees": vertical_angle_degrees
        }
    } 