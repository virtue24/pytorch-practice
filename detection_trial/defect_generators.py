import cv2
import numpy as np
import random, string
import math, os

def yolo_format(class_id, nx_center, ny_center, nwidth, nheight):
    """Returns the YOLO format of the defect."""
    return f"{class_id} {nx_center} {ny_center} {nwidth} {nheight}"

def draw_line_on_frame(frame: np.ndarray = None, defect_class=0, R_range=[0, 255], G_range=[0, 255], B_range=[0, 255], thickness_range=[1, 10], opacity_range=[0.1, 1.0]):
    """Draws a line on the frame with a specified opacity."""
    # Randomly generate RGB values, thickness, and opacity
    R = random.randint(R_range[0], R_range[1])
    G = random.randint(G_range[0], G_range[1])
    B = random.randint(B_range[0], B_range[1])
    thickness = random.randint(thickness_range[0], thickness_range[1])
    opacity = random.uniform(opacity_range[0], opacity_range[1])

    # Generate random start and end points for the line
    line_start = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
    line_end = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))

    overlay = frame.copy()
    cv2.line(overlay, line_start, line_end, (R, G, B), thickness)

    # Blend the overlay with the original frame using the specified opacity
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # Calculate defect label in YOLO format (optional)
    defect_nx_center = (line_start[0] + line_end[0]) / 2 / frame.shape[1]
    defect_ny_center = (line_start[1] + line_end[1]) / 2 / frame.shape[0]
    defect_nwidth = abs(line_start[0] - line_end[0]) / frame.shape[1]
    defect_nheight = abs(line_start[1] - line_end[1]) / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def draw_cocentric_lines_on_frame(
    frame: np.ndarray = None, 
    defect_class=0, 
    R_range=[0, 255], 
    G_range=[0, 255], 
    B_range=[0, 255], 
    thickness_range=[1, 10], 
    opacity_range=[0.1, 1.0], 
    num_lines_range=[5, 25], 
    random_offset_range=[-5, 5],
    rotation_angle_range=[-5, 5],
    bbox_range=[0.2, 0.5],  # Normalized range for the bbox (relative to the image size)
    line_length_range=[0.1, 0.3]  # Normalized range for the line length (relative to the image size)
    ):
    """
    Draws multiple slightly rotated parallel lines on the frame with a specified opacity.
    The next line's starting point is determined by adding random offsets to the previous starting point.
    The smallest bounding box including all the lines is returned as a label.

    :param frame: Input image as a NumPy array.
    :param defect_class: Class ID for the defect.
    :param R_range: Range for the Red color component.
    :param G_range: Range for the Green color component.
    :param B_range: Range for the Blue color component.
    :param thickness_range: Range for the thickness of the lines.
    :param opacity_range: Range for the opacity of the lines.
    :param num_lines_range: Range for the number of parallel lines to draw.
    :param line_spacing_range: Range for the spacing between the lines.
    :param random_offset_range: Range for random offsets applied to line starting positions.
    :param rotation_angle_range: Range for the slight rotation of each line (in degrees).
    :param bbox_range: Normalized range for bounding box size (relative to the frame size).
    :param line_length_range: Normalized range for the line lengths (relative to the frame size).
    :return: Tuple of (modified frame, YOLO label string).
    """
    height, width = frame.shape[:2]

    # Randomly generate RGB values, thickness, and opacity
    R = random.randint(R_range[0], R_range[1])
    G = random.randint(G_range[0], G_range[1])
    B = random.randint(B_range[0], B_range[1])
    thickness = random.randint(thickness_range[0], thickness_range[1])
    opacity = random.uniform(opacity_range[0], opacity_range[1])

    # Randomly define the bounding box where the lines will start
    bbox_width = int(random.uniform(bbox_range[0], bbox_range[1]) * width)
    bbox_height = int(random.uniform(bbox_range[0], bbox_range[1]) * height)

    x_min = random.randint(0, width - bbox_width)
    y_min = random.randint(0, height - bbox_height)
    x_max = x_min + bbox_width
    y_max = y_min + bbox_height

    # Function to apply slight rotation to a point
    def rotate_point(point, center, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        
        x, y = point
        cx, cy = center

        # Translate point back to origin
        x -= cx
        y -= cy

        # Apply rotation
        x_new = x * cos_theta - y * sin_theta
        y_new = x * sin_theta + y * cos_theta

        # Translate point back
        x_new += cx
        y_new += cy

        return int(x_new), int(y_new)

    overlay = frame.copy()

    # Initialize the first starting point randomly within the bounding box
    line_start = (random.randint(x_min, x_max), random.randint(y_min, y_max))

    # Variables to track the min and max coordinates for the bounding box
    global_x_min = float('inf')
    global_y_min = float('inf')
    global_x_max = float('-inf')
    global_y_max = float('-inf')

    # Draw multiple parallel lines
    num_lines = random.randint(num_lines_range[0], num_lines_range[1])
    for i in range(num_lines):
        # Generate a random length for the line (normalized)
        line_length = random.uniform(line_length_range[0], line_length_range[1]) * min(width, height)
        
        # Generate a random direction for the line end point
        angle = random.uniform(0, 2 * math.pi)
        line_end = (
            int(line_start[0] + line_length * math.cos(angle)),
            int(line_start[1] + line_length * math.sin(angle))
        )

        # Apply random offset to the start point for the next line
        offset_x = random.randint(random_offset_range[0], random_offset_range[1])
        offset_y = random.randint(random_offset_range[0], random_offset_range[1])
        
        # Calculate the next start point by adding random offsets to the current start point
        next_line_start = (line_start[0] + offset_x, line_start[1] + offset_y)

        # Apply slight rotation to the line
        rotation_angle = random.uniform(rotation_angle_range[0], rotation_angle_range[1])
        line_start_rotated = rotate_point(line_start, line_start, rotation_angle)
        line_end_rotated = rotate_point(line_end, line_start, rotation_angle)

        # Ensure the lines stay within the image bounds
        line_start_rotated = (min(max(line_start_rotated[0], 0), width), min(max(line_start_rotated[1], 0), height))
        line_end_rotated = (min(max(line_end_rotated[0], 0), width), min(max(line_end_rotated[1], 0), height))

        # Draw the rotated line
        cv2.line(overlay, line_start_rotated, line_end_rotated, (R, G, B), thickness)

        # Update the global bounding box coordinates
        global_x_min = min(global_x_min, line_start_rotated[0], line_end_rotated[0])
        global_y_min = min(global_y_min, line_start_rotated[1], line_end_rotated[1])
        global_x_max = max(global_x_max, line_start_rotated[0], line_end_rotated[0])
        global_y_max = max(global_y_max, line_start_rotated[1], line_end_rotated[1])

        # Update the current start point to the next line start point for the next iteration
        line_start = next_line_start

    # Blend the overlay with the original frame using the specified opacity
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # Calculate the smallest bounding box that includes all the lines
    bbox_center_x = (global_x_min + global_x_max) / 2 / width
    bbox_center_y = (global_y_min + global_y_max) / 2 / height
    bbox_width_normalized = (global_x_max - global_x_min) / width
    bbox_height_normalized = (global_y_max - global_y_min) / height

    defect_label = yolo_format(defect_class, bbox_center_x, bbox_center_y, bbox_width_normalized, bbox_height_normalized)

    return frame, defect_label

def sweep_copied_section_arbitrary(
    frame: np.ndarray = None,
    defect_class=0,
    sweep_range=[0.1, 0.3],  # Normalized range for how far the section is swept (relative to the image size)
    section_size_range=[0.1, 0.3],  # Normalized range for the size of the copied section (relative to frame size)
    num_steps_range=[5, 20]  # Number of steps in the sweeping effect
    ):
    """
    Copies a section of the image and sweeps it in an arbitrary direction to create a distortion effect.

    :param frame: Input image as a NumPy array.
    :param defect_class: Class ID for the defect.
    :param sweep_range: Normalized range for how far the section is swept (relative to the image size).
    :param section_size_range: Normalized range for the size of the copied section (relative to frame size).
    :param num_steps_range: Range for the number of steps in the sweeping effect.
    :return: Tuple of (modified frame, YOLO label string).
    """
    height, width = frame.shape[:2]

    # Define the size of the section to be copied
    section_width = int(random.uniform(section_size_range[0], section_size_range[1]) * width)
    section_height = int(random.uniform(section_size_range[0], section_size_range[1]) * height)

    # Select a random starting point for the section to copy
    x_start = random.randint(0, width - section_width)
    y_start = random.randint(0, height - section_height)
    x_end = x_start + section_width
    y_end = y_start + section_height

    # Copy the selected section
    section = frame[y_start:y_end, x_start:x_end].copy()

    # Generate a random angle for the sweeping direction (in radians)
    angle = random.uniform(0, 2 * math.pi)

    # Calculate the maximum sweep distance in the direction of the angle
    sweep_distance = random.uniform(sweep_range[0], sweep_range[1]) * min(width, height)

    # Number of steps in the sweep
    num_steps = random.randint(num_steps_range[0], num_steps_range[1])
    step_size = sweep_distance / num_steps

    # Calculate the step sizes in x and y directions based on the angle
    step_x = int(step_size * math.cos(angle))
    step_y = int(step_size * math.sin(angle))

    overlay = frame.copy()

    for i in range(num_steps):
        # Calculate the new position based on the direction and step size
        x_offset = min(max(x_start + i * step_x, 0), width - section_width)
        y_offset = min(max(y_start + i * step_y, 0), height - section_height)

        # Paste the copied section in its new position
        overlay[y_offset:y_offset + section_height, x_offset:x_offset + section_width] = section

    # Blend the overlay with the original frame
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Calculate the YOLO label for the sweeping effect
    defect_nx_center = (x_start + section_width / 2) / width
    defect_ny_center = (y_start + section_height / 2) / height
    defect_nwidth = section_width / width
    defect_nheight = section_height / height
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)

    return frame, defect_label

def draw_circle_on_frame(frame: np.ndarray = None, defect_class=0, R_range=[0, 255], G_range=[0, 255], B_range=[0, 255], thickness_range=[1, 10], opacity_range=[0.1, 1.0], radius_range=[5, 50], infill_probability = 0.5):
    """Draws a circle on the frame with a specified opacity."""
    R = random.randint(R_range[0], R_range[1])
    G = random.randint(G_range[0], G_range[1])
    B = random.randint(B_range[0], B_range[1])
    thickness = random.randint(thickness_range[0], thickness_range[1])
    opacity = random.uniform(opacity_range[0], opacity_range[1])
    radius = random.randint(radius_range[0], radius_range[1])

    center = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))

    overlay = frame.copy()    
    if random.random() < infill_probability:
        cv2.circle(overlay, center, radius, (R, G, B), -1)
    else:
        cv2.circle(overlay, center, radius, (R, G, B), thickness)

    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    defect_nx_center = center[0] / frame.shape[1]
    defect_ny_center = center[1] / frame.shape[0]
    defect_nwidth = 2 * radius / frame.shape[1]
    defect_nheight = 2 * radius / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def draw_concave_polygon_on_frame(
    frame: np.ndarray = None,
    defect_class=0,
    R_range=[0, 255],
    G_range=[0, 255],
    B_range=[0, 255],
    thickness_range=[1, 10],
    opacity_range=[0.1, 1.0],
    num_vertices_range=[5, 10],
    size_range=[20, 100],
    infill_probability=0.5
    ):
    """
    Draws a concave simple polygon on the frame with a specified opacity.

    :param frame: Input image as a NumPy array.
    :param defect_class: Class ID for the defect.
    :param R_range: Range for the Red color component.
    :param G_range: Range for the Green color component.
    :param B_range: Range for the Blue color component.
    :param thickness_range: Range for the thickness of the polygon edges.
    :param opacity_range: Range for the opacity of the polygon.
    :param num_vertices_range: Range for the number of vertices of the polygon.
    :param size_range: Range for the size (radius) of the polygon.
    :param infill_probability: Probability that the polygon will be filled.
    :return: Tuple of (modified frame, YOLO label string).
    """
    height, width = frame.shape[:2]

    # Random color selection
    R = random.randint(R_range[0], R_range[1])
    G = random.randint(G_range[0], G_range[1])
    B = random.randint(B_range[0], B_range[1])

    # Random thickness and opacity
    thickness = random.randint(thickness_range[0], thickness_range[1])
    opacity = random.uniform(opacity_range[0], opacity_range[1])

    # Random number of vertices
    num_vertices = random.randint(num_vertices_range[0], num_vertices_range[1])

    # Random size
    size = random.randint(size_range[0], size_range[1])

    # Random center position, ensuring the polygon stays within the frame
    center_x = random.randint(size, width - size)
    center_y = random.randint(size, height - size)
    center = (center_x, center_y)

    # Generate a simple polygon
    # Create points in polar coordinates, ensuring they are sorted by angle
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
    radii = size * (0.5 + np.random.uniform(0, 0.5, num_vertices))
    x = center_x + (radii * np.cos(angles)).astype(int)
    y = center_y + (radii * np.sin(angles)).astype(int)
    points = np.vstack((x, y)).T

    # Introduce concavity by moving one or more vertices inward
    num_concave_vertices = random.randint(1, max(1, num_vertices // 3))
    concave_indices = random.sample(range(num_vertices), num_concave_vertices)
    for idx in concave_indices:
        # Move the vertex towards the center to create a concave shape
        direction = np.array([center_x - x[idx], center_y - y[idx]])
        displacement = direction * random.uniform(0.2, 0.5)
        points[idx] += displacement.astype(int)

    # Ensure the polygon is simple by keeping the points sorted by angle
    # Recompute angles after introducing concavity
    vectors = points - np.array([center_x, center_y])
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(angles)
    points = points[sorted_indices]

    # Create the overlay
    overlay = frame.copy()

    # Draw the polygon
    if random.random() < infill_probability:
        cv2.fillPoly(overlay, [points], color=(B, G, R))
    else:
        cv2.polylines(overlay, [points], isClosed=True, color=(B, G, R), thickness=thickness)

    # Blend the overlay with the frame
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # Calculate bounding box for the polygon
    x_min = np.clip(np.min(points[:, 0]), 0, width)
    x_max = np.clip(np.max(points[:, 0]), 0, width)
    y_min = np.clip(np.min(points[:, 1]), 0, height)
    y_max = np.clip(np.max(points[:, 1]), 0, height)

    # Calculate YOLO format labels
    bbox_center_x = (x_min + x_max) / 2 / width
    bbox_center_y = (y_min + y_max) / 2 / height
    bbox_width = (x_max - x_min) / width
    bbox_height = (y_max - y_min) / height

    defect_label = yolo_format(defect_class, bbox_center_x, bbox_center_y, bbox_width, bbox_height)

    return frame, defect_label

def draw_rectangle_on_frame(frame: np.ndarray = None, defect_class=0, R_range=[0, 255], G_range=[0, 255], B_range=[0, 255], thickness_range=[1, 10], opacity_range=[0.1, 1.0], infill_probability=0.5):
    """Draws a rectangle on the frame with a specified opacity."""
    R = random.randint(R_range[0], R_range[1])
    G = random.randint(G_range[0], G_range[1])
    B = random.randint(B_range[0], B_range[1])
    thickness = random.randint(thickness_range[0], thickness_range[1])
    opacity = random.uniform(opacity_range[0], opacity_range[1])
    
    top_left = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
    width = random.randint(10, frame.shape[1] // 4)
    height = random.randint(10, frame.shape[0] // 4)
    bottom_right = (min(top_left[0] + width, frame.shape[1]), min(top_left[1] + height, frame.shape[0]))
    
    overlay = frame.copy()
    if random.random() < infill_probability:
        cv2.rectangle(overlay, top_left, bottom_right, (R, G, B), -1)
    else:
        cv2.rectangle(overlay, top_left, bottom_right, (R, G, B), thickness)
    
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # YOLO format calculations
    defect_nx_center = (top_left[0] + bottom_right[0]) / 2 / frame.shape[1]
    defect_ny_center = (top_left[1] + bottom_right[1]) / 2 / frame.shape[0]
    defect_nwidth = abs(bottom_right[0] - top_left[0]) / frame.shape[1]
    defect_nheight = abs(bottom_right[1] - top_left[1]) / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def add_gaussian_noise(frame: np.ndarray = None, defect_class=0, mean=0, var=0.01):
    """Adds Gaussian noise to the frame."""
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, frame.shape).astype('uint8')
    noisy_frame = cv2.add(frame, gauss)
    
    # Since noise is over the entire image, YOLO label covers the whole image
    defect_label = yolo_format(defect_class, 0.5, 0.5, 1.0, 1.0)
    
    return noisy_frame, defect_label

def add_salt_and_pepper_noise(frame: np.ndarray = None, defect_class=0, amount=0.005, s_vs_p=0.5, dimension_range=[0.1, 0.3]):
    """Adds salt and pepper noise to a specific region of the frame."""
    
    noisy_frame = frame.copy()
    
    # Calculate the region size based on the dimension_range
    region_width = int(random.uniform(dimension_range[0], dimension_range[1]) * frame.shape[1])
    region_height = int(random.uniform(dimension_range[0], dimension_range[1]) * frame.shape[0])
    
    # Randomly select the top-left corner of the region
    x_start = random.randint(0, frame.shape[1] - region_width)
    y_start = random.randint(0, frame.shape[0] - region_height)
    
    # Define the region of interest (ROI)
    roi = noisy_frame[y_start:y_start + region_height, x_start:x_start + region_width]
    
    # Calculate the number of salt and pepper pixels within the region
    num_salt = np.ceil(amount * roi.size * s_vs_p)
    num_pepper = np.ceil(amount * roi.size * (1.0 - s_vs_p))
    
    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in roi.shape]
    roi[coords[0], coords[1], :] = 255
    
    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in roi.shape]
    roi[coords[0], coords[1], :] = 0
    
    # Update the noisy frame with the modified region
    noisy_frame[y_start:y_start + region_height, x_start:x_start + region_width] = roi
    
    # YOLO format calculations
    defect_nx_center = (x_start + region_width / 2) / frame.shape[1]
    defect_ny_center = (y_start + region_height / 2) / frame.shape[0]
    defect_nwidth = region_width / frame.shape[1]
    defect_nheight = region_height / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return noisy_frame, defect_label

def draw_ellipse_on_frame(
    frame: np.ndarray = None,
    defect_class=4,
    R_range=[0, 255],
    G_range=[0, 255],
    B_range=[0, 255],
    angle_range=[0, 360],
    axes_length_range=[10, 100],
    thickness_range=[1, 10],
    opacity_range=[0.1, 1.0]
    ):
    """Draws an ellipse on the frame with a specified opacity and updates the label accordingly."""
    # Randomly generate RGB values, thickness, opacity, and angle
    R = random.randint(R_range[0], R_range[1])
    G = random.randint(G_range[0], G_range[1])
    B = random.randint(B_range[0], B_range[1])
    angle = random.randint(angle_range[0], angle_range[1])
    axes_length = (
        random.randint(axes_length_range[0], axes_length_range[1]),
        random.randint(axes_length_range[0], axes_length_range[1])
    )
    thickness = random.randint(thickness_range[0], thickness_range[1])
    opacity = random.uniform(opacity_range[0], opacity_range[1])

    # Generate random center point for the ellipse
    center = (
        random.randint(axes_length[0], frame.shape[1] - axes_length[0]),
        random.randint(axes_length[1], frame.shape[0] - axes_length[1])
    )

    # Create an overlay to draw the ellipse
    overlay = frame.copy()

    # Draw the ellipse on the overlay
    cv2.ellipse(
        overlay,
        center,
        axes_length,
        angle,
        0,
        360,
        (B, G, R),  # OpenCV uses BGR format
        thickness
    )

    # Blend the overlay with the original frame using the specified opacity
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # Calculate the rotated rectangle that bounds the ellipse
    # Approximate the ellipse with a polyline (set of points)
    ellipse_pts = cv2.ellipse2Poly(center, axes_length, angle, 0, 360, 1)

    # Get the axis-aligned bounding rectangle of the rotated ellipse
    x, y, w, h = cv2.boundingRect(ellipse_pts)

    # Ensure the bounding box is within the frame boundaries
    x_min = max(0, x)
    y_min = max(0, y)
    x_max = min(frame.shape[1], x + w)
    y_max = min(frame.shape[0], y + h)

    # Calculate normalized center coordinates and dimensions for YOLO format
    defect_nx_center = ((x_min + x_max) / 2) / frame.shape[1]
    defect_ny_center = ((y_min + y_max) / 2) / frame.shape[0]
    defect_nwidth = (x_max - x_min) / frame.shape[1]
    defect_nheight = (y_max - y_min) / frame.shape[0]
    defect_label = yolo_format(
        defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight
    )

    return frame, defect_label

def add_blurred_region(frame: np.ndarray = None, defect_class=0, kernel_size_range=[5, 31]):
    """Adds a blurred region to the frame."""
    kernel_size = random.choice(range(kernel_size_range[0], kernel_size_range[1], 2))  # Kernel size must be odd
    x_start = random.randint(0, frame.shape[1] - kernel_size)
    y_start = random.randint(0, frame.shape[0] - kernel_size)
    x_end = x_start + kernel_size
    y_end = y_start + kernel_size
    
    blurred_region = cv2.GaussianBlur(frame[y_start:y_end, x_start:x_end], (kernel_size, kernel_size), 0)
    frame[y_start:y_end, x_start:x_end] = blurred_region
    
    # YOLO format calculations
    defect_nx_center = (x_start + x_end) / 2 / frame.shape[1]
    defect_ny_center = (y_start + y_end) / 2 / frame.shape[0]
    defect_nwidth = (x_end - x_start) / frame.shape[1]
    defect_nheight = (y_end - y_start) / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def add_scratch_on_frame(frame: np.ndarray = None, defect_class=0, num_lines=5, length_range=[10, 100], thickness_range=[1, 3], color=(0, 0, 0)):
    """
    Adds scratches (random lines) on the frame and creates a separate YOLO label for each line.

    :param frame: Input image frame as a NumPy array.
    :param defect_class: Class ID for the defect.
    :param num_lines: Number of scratches to add.
    :param length_range: Range of lengths for the scratches.
    :param thickness_range: Range of thicknesses for the scratches.
    :param color: Color of the scratches (B, G, R).
    :return: Tuple of (modified frame, labels as a single string with newlines).
    """
    overlay = frame.copy()
    labels = []
    height, width = frame.shape[:2]
    
    for _ in range(num_lines):
        # Random starting point
        x_start = random.randint(0, width - 1)
        y_start = random.randint(0, height - 1)
        
        # Random length and angle
        length = random.randint(length_range[0], length_range[1])
        angle = random.uniform(0, 2 * math.pi)
        
        # Calculate ending point
        x_end = int(x_start + length * math.cos(angle))
        y_end = int(y_start + length * math.sin(angle))
        
        # Ensure end points are within the frame boundaries
        x_end = max(0, min(x_end, width - 1))
        y_end = max(0, min(y_end, height - 1))
        
        # Random thickness
        thickness = random.randint(thickness_range[0], thickness_range[1])
        
        # Draw the scratch (line) on the overlay
        cv2.line(overlay, (x_start, y_start), (x_end, y_end), color, thickness)
        
        # Calculate bounding box coordinates
        x_min = max(0, min(x_start, x_end) - thickness // 2)
        y_min = max(0, min(y_start, y_end) - thickness // 2)
        x_max = min(width - 1, max(x_start, x_end) + thickness // 2)
        y_max = min(height - 1, max(y_start, y_end) + thickness // 2)
        
        # Calculate normalized center coordinates and dimensions
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        x_center = x_min + bbox_width / 2
        y_center = y_min + bbox_height / 2
        
        defect_nx_center = x_center / width
        defect_ny_center = y_center / height
        defect_nwidth = bbox_width / width
        defect_nheight = bbox_height / height
        
        # Create the YOLO label for this scratch
        defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
        labels.append(defect_label)
    
    # Blend the overlay with the original frame
    opacity = 1.0
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # Combine all labels into a single string with newlines
    defect_labels = '\n'.join(labels)
    
    return frame, defect_labels

def add_watermark_text(frame: np.ndarray = None, defect_class=0, text_length = 10, font_scale=2, thickness=3, opacity=0.3, R_range=[0, 255], G_range=[0, 255], B_range=[0, 255]):
    """Adds watermark text on the frame."""
    characters = string.ascii_letters + string.digits + string.punctuation
    text = ''.join(random.choice(characters) for i in range(text_length))
    
    color = (random.randint(R_range[0], R_range[1]), random.randint(G_range[0], G_range[1]), random.randint(B_range[0], B_range[1]))
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = random.randint(0, frame.shape[1] - text_size[0])
    y = random.randint(text_size[1], frame.shape[0])
    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # YOLO format covering the text area
    defect_nx_center = (x + text_size[0] / 2) / frame.shape[1]
    defect_ny_center = (y - text_size[1] / 2) / frame.shape[0]
    defect_nwidth = text_size[0] / frame.shape[1]
    defect_nheight = text_size[1] / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def add_lens_flare(frame: np.ndarray = None, defect_class=0, flare_center=None, radius_range=[50, 150], opacity_range=[0.1, 0.5]):
    """Adds a lens flare effect to the frame."""
    if flare_center is None:
        flare_center = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
    radius = random.randint(radius_range[0], radius_range[1])
    opacity = random.uniform(opacity_range[0], opacity_range[1])
    
    overlay = frame.copy()
    mask = np.zeros_like(frame)
    cv2.circle(mask, flare_center, radius, (255, 255, 255), -1)
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius/2, sigmaY=radius/2)
    overlay = cv2.addWeighted(overlay, 1, blurred_mask, opacity, 0)
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # YOLO format calculations
    defect_nx_center = flare_center[0] / frame.shape[1]
    defect_ny_center = flare_center[1] / frame.shape[0]
    defect_nwidth = 2 * radius / frame.shape[1]
    defect_nheight = 2 * radius / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def add_shadow(frame: np.ndarray = None, defect_class=0, top_left_ratio=(0.2, 0.2), bottom_right_ratio=(0.8, 0.8), opacity=0.5):
    """Adds a shadow to a region of the frame."""
    h, w = frame.shape[:2]
    top_left = (int(w * top_left_ratio[0]), int(h * top_left_ratio[1]))
    bottom_right = (int(w * bottom_right_ratio[0]), int(h * bottom_right_ratio[1]))
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # YOLO format calculations
    defect_nx_center = (top_left[0] + bottom_right[0]) / 2 / w
    defect_ny_center = (top_left[1] + bottom_right[1]) / 2 / h
    defect_nwidth = abs(bottom_right[0] - top_left[0]) / w
    defect_nheight = abs(bottom_right[1] - top_left[1]) / h
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def add_color_tint(frame: np.ndarray = None, defect_class=0, tint_color=(0, 255, 255), opacity=0.3):
    """Adds a color tint to the frame."""
    overlay = np.full(frame.shape, tint_color, dtype=np.uint8)
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # Since the tint is over the whole image
    defect_label = yolo_format(defect_class, 0.5, 0.5, 1.0, 1.0)
    
    return frame, defect_label

def add_crack_texture(frame: np.ndarray = None, defect_class=0, crack_image_path='crack.png', opacity_range=[0.3, 0.7], size_percentage_range=[0.1, 0.3]):
    """Overlays a resized crack texture on a portion of the frame."""
    opacity = random.uniform(opacity_range[0], opacity_range[1])
    crack_img = cv2.imread(crack_image_path, cv2.IMREAD_UNCHANGED)
    
    if crack_img is None:
        print("Crack image not found.")
        return frame, None
    
    # Resize crack image to a smaller size (based on a percentage of the frame size)
    crack_width = int(random.uniform(size_percentage_range[0], size_percentage_range[1]) * frame.shape[1])
    crack_height = int(random.uniform(size_percentage_range[0], size_percentage_range[1]) * frame.shape[0])
    crack_img_resized = cv2.resize(crack_img, (crack_width, crack_height))

    # Generate random position where the crack will be placed
    x_offset = random.randint(0, frame.shape[1] - crack_width)
    y_offset = random.randint(0, frame.shape[0] - crack_height)

    # Overlay the crack image on the frame
    if crack_img_resized.shape[2] == 4:
        # If the crack image has an alpha channel, blend using the alpha mask
        alpha_mask = crack_img_resized[:, :, 3] / 255.0 * opacity
        for c in range(0, 3):
            frame[y_offset:y_offset + crack_height, x_offset:x_offset + crack_width, c] = (
                frame[y_offset:y_offset + crack_height, x_offset:x_offset + crack_width, c] * (1 - alpha_mask) +
                crack_img_resized[:, :, c] * alpha_mask
            )
    else:
        # If no alpha channel, use standard blending
        overlay = frame.copy()
        overlay[y_offset:y_offset + crack_height, x_offset:x_offset + crack_width] = crack_img_resized
        frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # YOLO label to cover the area where the crack is applied
    defect_nx_center = (x_offset + crack_width / 2) / frame.shape[1]
    defect_ny_center = (y_offset + crack_height / 2) / frame.shape[0]
    defect_nwidth = crack_width / frame.shape[1]
    defect_nheight = crack_height / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def add_dust_particles(frame: np.ndarray = None, defect_class=0, num_particles_range=[10,50], size_range=[1, 5], bbox_range=[0.2, 0.5]):
    """
    Adds dust particles to a randomly selected bounding box within the frame.

    :param frame: Input image as a NumPy array.
    :param defect_class: Class ID for the defect.
    :param num_particles: Number of dust particles to add.
    :param size_range: Size range for the dust particles.
    :param bbox_range: Range for the normalized width and height of the bounding box (relative to frame dimensions).
    :return: Tuple of (modified frame, YOLO label string).
    """
    height, width = frame.shape[:2]

    # Randomly determine bounding box size (normalized relative to the image size)
    bbox_width = int(random.uniform(bbox_range[0], bbox_range[1]) * width)
    bbox_height = int(random.uniform(bbox_range[0], bbox_range[1]) * height)

    # Randomly select the top-left corner of the bounding box, ensuring it fits within the frame
    x_min = random.randint(0, width - bbox_width)
    y_min = random.randint(0, height - bbox_height)
    x_max = x_min + bbox_width
    y_max = y_min + bbox_height

    # Create an overlay for the dust particles
    overlay = frame.copy()

    # Add dust particles within the bounding box
    num_particles = random.randint(num_particles_range[0], num_particles_range[1])
    for _ in range(num_particles):
        x = random.randint(x_min, x_max - 1)
        y = random.randint(y_min, y_max - 1)
        size = random.randint(size_range[0], size_range[1])
        cv2.circle(overlay, (x, y), size, (255, 255, 255), -1)

    # Blend the overlay with the original frame using opacity
    opacity = 0.5
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # YOLO label for the bounding box where particles were added
    bbox_center_x = (x_min + x_max) / 2 / width
    bbox_center_y = (y_min + y_max) / 2 / height
    bbox_width_normalized = (x_max - x_min) / width
    bbox_height_normalized = (y_max - y_min) / height

    defect_label = yolo_format(defect_class, bbox_center_x, bbox_center_y, bbox_width_normalized, bbox_height_normalized)

    return frame, defect_label
def add_fog_effect(frame: np.ndarray = None, defect_class=0, fog_density=0.5):
    """Adds a fog effect to the frame."""
    fog = np.full(frame.shape, (255, 255, 255), dtype=np.uint8)
    frame = cv2.addWeighted(frame, 1 - fog_density, fog, fog_density, 0)
    
    # YOLO label covers the entire image
    defect_label = yolo_format(defect_class, 0.5, 0.5, 1.0, 1.0)
    
    return frame, defect_label

def draw_copy_paste_on_frame(
        frame: np.ndarray = None,
        defect_class=0,
        R_range=[0, 255],
        G_range=[0, 255],
        B_range=[0, 255],
        ncut_length_range=[0.05, 0.2],
        ncopy_region=[0.0, 0.0, 1.0, 1.0]
    ):
    """Draws a cut on the frame by copying and pasting a region."""
    # Random color components (currently unused)
    R = random.randint(R_range[0], R_range[1])
    G = random.randint(G_range[0], G_range[1])
    B = random.randint(B_range[0], B_range[1])
    
    height, width = frame.shape[:2]
    
    # Calculate cut length based on the frame size
    cut_length = random.randint(
        int(ncut_length_range[0] * width),
        int(ncut_length_range[1] * width)
    )

    # Define the region for copying
    copy_region = (
        int(ncopy_region[0] * width),  # x_min
        int(ncopy_region[1] * height),  # y_min
        int(ncopy_region[2] * width),  # x_max
        int(ncopy_region[3] * height)   # y_max
    )
    
    # Adjust copy_region to ensure it can accommodate cut_length
    max_copy_x = copy_region[2] - cut_length
    max_copy_y = copy_region[3] - cut_length
    min_copy_x = copy_region[0]
    min_copy_y = copy_region[1]
    
    if max_copy_x <= min_copy_x or max_copy_y <= min_copy_y:
        raise ValueError("Copy region is too small for the specified cut_length.")
    
    # Generate random start point for the copied region
    copy_start_x = random.randint(min_copy_x, max_copy_x)
    copy_start_y = random.randint(min_copy_y, max_copy_y)
    copy_end_x = copy_start_x + cut_length
    copy_end_y = copy_start_y + cut_length
    
    # Copy the selected section from the frame
    copied_section = frame[copy_start_y:copy_end_y, copy_start_x:copy_end_x].copy()
    
    # Generate random shift values
    max_shift_x = width - cut_length
    max_shift_y = height - cut_length
    
    if max_shift_x <= 0 or max_shift_y <= 0:
        raise ValueError("Frame is too small for the copied section.")
    
    # Random shift within the frame boundaries
    x_shift = random.randint(-cut_length, cut_length)
    y_shift = random.randint(-cut_length, cut_length)
    
    paste_start_x = copy_start_x + x_shift
    paste_start_y = copy_start_y + y_shift
    
    # Ensure paste coordinates are within the frame
    paste_start_x = max(0, min(paste_start_x, width - cut_length))
    paste_start_y = max(0, min(paste_start_y, height - cut_length))
    
    paste_end_x = paste_start_x + cut_length
    paste_end_y = paste_start_y + cut_length
    
    # Paste the copied section onto the new location
    frame[paste_start_y:paste_end_y, paste_start_x:paste_end_x] = copied_section
    
    # Calculate normalized coordinates for YOLO format
    defect_nx_center = (paste_start_x + cut_length / 2) / width
    defect_ny_center = (paste_start_y + cut_length / 2) / height
    defect_nwidth = cut_length / width
    defect_nheight = cut_length / height
    defect_label = yolo_format(
        defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight
    )
    
    return frame, defect_label

def apply_wave_distortion(
    frame: np.ndarray = None,
    defect_class=14,
    amplitude_range=[5, 20],
    wavelength_range=[10, 50],
    phase_range=[0, np.pi * 2],
    dimension_range=[0.1, 0.3]
    ):
    """
    Applies a wave distortion effect to a specific region of the frame.
    
    :param frame: Input image as a NumPy array.
    :param defect_class: Class ID for the defect.
    :param amplitude_range: Range of amplitudes for the wave effect.
    :param wavelength_range: Range of wavelengths for the wave effect.
    :param phase_range: Range of phases for the wave effect.
    :param dimension_range: Range of relative sizes for the affected region (relative to frame dimensions).
    :return: Tuple of (modified frame, YOLO label string).
    """
    height, width = frame.shape[:2]
    
    # Define the region size based on the dimension_range
    region_width = int(random.uniform(dimension_range[0], dimension_range[1]) * width)
    region_height = int(random.uniform(dimension_range[0], dimension_range[1]) * height)
    
    # Randomly select the top-left corner of the region
    x_start = random.randint(0, width - region_width)
    y_start = random.randint(0, height - region_height)
    x_end = x_start + region_width
    y_end = y_start + region_height
    
    # Extract the region of interest (ROI)
    roi = frame[y_start:y_end, x_start:x_end].copy()
    
    # Generate the wave distortion parameters
    amplitude = random.uniform(amplitude_range[0], amplitude_range[1])
    wavelength = random.uniform(wavelength_range[0], wavelength_range[1])
    phase = random.uniform(phase_range[0], phase_range[1])
    
    # Create meshgrid for pixel indices
    map_y, map_x = np.indices((region_height, region_width), dtype=np.float32)
    
    # Apply wave distortion along x-axis
    map_x_distorted = map_x + amplitude * np.sin(2 * np.pi * map_y / wavelength + phase)
    map_y_distorted = map_y
    
    # Ensure the distorted indices are within the image boundaries
    map_x_distorted = np.clip(map_x_distorted, 0, region_width - 1)
    map_y_distorted = np.clip(map_y_distorted, 0, region_height - 1)
    
    # Remap the ROI using the distorted indices
    distorted_roi = cv2.remap(roi, map_x_distorted, map_y_distorted, interpolation=cv2.INTER_LINEAR)
    
    # Place the distorted ROI back into the frame
    frame[y_start:y_end, x_start:x_end] = distorted_roi
    
    # Calculate YOLO label for the distorted region
    defect_nx_center = (x_start + region_width / 2) / width
    defect_ny_center = (y_start + region_height / 2) / height
    defect_nwidth = region_width / width
    defect_nheight = region_height / height
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def apply_twirl_distortion(
    frame: np.ndarray = None,
    defect_class=15,
    strength_range=[0.5, 2.0],
    radius_range=[50, 150],
    dimension_range=[0.1, 0.3]
    ):
    """
    Applies a twirl distortion effect to a specific region of the frame.
    """
    height, width = frame.shape[:2]
    
    # Define the region size based on the dimension_range
    region_width = int(random.uniform(dimension_range[0], dimension_range[1]) * width)
    region_height = int(random.uniform(dimension_range[0], dimension_range[1]) * height)
    
    # Ensure region size is at least 2x2 pixels
    if region_width < 2 or region_height < 2:
        print("Selected region is too small. Skipping twirl effect.")
        return frame, ''
    
    # Randomly select the top-left corner of the region
    x_start = random.randint(0, width - region_width)
    y_start = random.randint(0, height - region_height)
    x_end = x_start + region_width
    y_end = y_start + region_height
    
    # Extract the region of interest (ROI)
    roi = frame[y_start:y_end, x_start:x_end].copy()
    
    # Generate the twirl distortion parameters
    strength = random.uniform(strength_range[0], strength_range[1])
    max_radius = min(region_width, region_height) // 2

    # Adjust radius limits
    upper_radius_limit = min(radius_range[1], max_radius)
    lower_radius_limit = min(radius_range[0], upper_radius_limit)

    if lower_radius_limit > upper_radius_limit or upper_radius_limit <= 0:
        print("Region too small for the desired radius. Skipping twirl effect.")
        return frame, ''
    
    # Generate radius within the valid range
    radius = random.randint(int(lower_radius_limit), int(upper_radius_limit))
    
    # Center of the twirl in the ROI
    center_x = region_width // 2
    center_y = region_height // 2
    
    # Create meshgrid for pixel indices
    map_x, map_y = np.meshgrid(np.arange(region_width), np.arange(region_height))
    map_x = map_x.astype(np.float32) - center_x
    map_y = map_y.astype(np.float32) - center_y
    
    # Calculate the distance and angle for each pixel
    distance = np.sqrt(map_x**2 + map_y**2)
    angle = np.arctan2(map_y, map_x)
    
    # Apply the twirl effect
    mask = distance < radius
    distance_masked = distance[mask]
    angle_masked = angle[mask]
    
    # Calculate the amount of twirl based on distance
    beta = strength * (radius - distance_masked) / radius
    angle_masked += beta
    
    # Convert polar coordinates back to cartesian
    map_x_new = distance_masked * np.cos(angle_masked) + center_x
    map_y_new = distance_masked * np.sin(angle_masked) + center_y
    
    # Update the mapping for the twirled pixels
    map_x_twirled = np.copy(map_x + center_x)
    map_y_twirled = np.copy(map_y + center_y)
    map_x_twirled[mask] = map_x_new
    map_y_twirled[mask] = map_y_new
    
    # Ensure the mapping indices are within valid range
    map_x_twirled = np.clip(map_x_twirled, 0, region_width - 1)
    map_y_twirled = np.clip(map_y_twirled, 0, region_height - 1)
    
    # Remap the ROI using the distorted indices
    distorted_roi = cv2.remap(
        roi,
        map_x_twirled.astype(np.float32),
        map_y_twirled.astype(np.float32),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Place the distorted ROI back into the frame
    frame[y_start:y_end, x_start:x_end] = distorted_roi
    
    # Calculate YOLO label for the distorted region
    defect_nx_center = (x_start + region_width / 2) / width
    defect_ny_center = (y_start + region_height / 2) / height
    defect_nwidth = region_width / width
    defect_nheight = region_height / height
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def apply_concentric_warp_defect(
    frame: np.ndarray = None,
    defect_class=17,
    strength_range=[0.5, 2.0],
    nradius_range=[0.2, 0.4],
    dimension_range=[0.3, 0.5],
    warp_type='inward'  # Options: 'inward', 'outward'
    ):
    """
    Applies a concentric warp distortion effect to a specific region of the frame.

    :param frame: Input image as a NumPy array.
    :param defect_class: Class ID for the defect.
    :param strength_range: Range of strengths for the warp effect.
    :param nradius_range: Range of normalized radii for the warp effect (relative to min(height, width)).
    :param dimension_range: Range of relative sizes for the affected region (relative to frame dimensions).
    :param warp_type: Type of warp ('inward' or 'outward').
    :return: Tuple of (modified frame, YOLO label string).
    """
    height, width = frame.shape[:2]
    radius_range = [int(nradius_range[0] * min(height, width)), int(nradius_range[1] * min(height, width))]

    # Ensure radius_range values are valid
    if radius_range[0] >= radius_range[1]:
        radius_range[1] = radius_range[0] + 1  # Increase upper limit if necessary

    # Define the region size based on the dimension_range
    min_region_size = 2 * radius_range[1]
    region_width = max(int(random.uniform(dimension_range[0], dimension_range[1]) * width), min_region_size)
    region_height = max(int(random.uniform(dimension_range[0], dimension_range[1]) * height), min_region_size)

    # Ensure region size does not exceed frame dimensions
    region_width = min(region_width, width)
    region_height = min(region_height, height)

    # Randomly select the top-left corner of the region
    x_start = random.randint(0, width - region_width)
    y_start = random.randint(0, height - region_height)

    # Extract the region of interest (ROI)
    roi = frame[y_start:y_start + region_height, x_start:x_start + region_width].copy()

    # Generate the warp distortion parameters
    strength = random.uniform(strength_range[0], strength_range[1])
    max_radius = min(region_width, region_height) // 2

    # Adjust radius limits
    upper_radius_limit = min(radius_range[1], max_radius)
    lower_radius_limit = min(radius_range[0], upper_radius_limit)

    if lower_radius_limit > upper_radius_limit or upper_radius_limit <= 0:
        print("Region too small for the desired radius. Skipping concentric warp effect.")
        return frame, ''

    # Generate radius within the valid range
    radius = random.randint(int(lower_radius_limit), int(upper_radius_limit))

    # Center of the warp in the ROI
    center_x = region_width // 2
    center_y = region_height // 2

    # Create meshgrid for pixel indices
    map_x, map_y = np.meshgrid(np.arange(region_width), np.arange(region_height))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Shift coordinates to center
    map_x_shifted = map_x - center_x
    map_y_shifted = map_y - center_y

    # Calculate the distance for each pixel from the center
    distance = np.sqrt(map_x_shifted**2 + map_y_shifted**2)
    angle = np.arctan2(map_y_shifted, map_x_shifted)

    # Normalize the distance
    distance_normalized = distance / radius
    distance_normalized[distance_normalized > 1] = 1  # Cap the values at 1

    # Calculate the displacement based on the warp effect
    if warp_type == 'inward':
        # Pixels move towards the center
        displacement = strength * (distance_normalized ** 2)
        new_distance = distance * (1 - displacement)
    elif warp_type == 'outward':
        # Pixels move away from the center
        displacement = strength * (1 - distance_normalized ** 2)
        new_distance = distance * (1 + displacement)
    else:
        print("Invalid warp_type specified. Choose 'inward' or 'outward'.")
        return frame, ''

    # Convert polar coordinates back to cartesian
    map_x_new = new_distance * np.cos(angle) + center_x
    map_y_new = new_distance * np.sin(angle) + center_y

    # Ensure the mapping indices are within valid range
    map_x_new = np.clip(map_x_new, 0, region_width - 1)
    map_y_new = np.clip(map_y_new, 0, region_height - 1)

    # Remap the ROI using the distorted indices
    distorted_roi = cv2.remap(
        roi,
        map_x_new,
        map_y_new,
        interpolation=cv2.INTER_LINEAR
    )

    # Place the distorted ROI back into the frame
    frame[y_start:y_start + region_height, x_start:x_start + region_width] = distorted_roi

    # Calculate YOLO label for the distorted region
    defect_nx_center = (x_start + region_width / 2) / width
    defect_ny_center = (y_start + region_height / 2) / height
    defect_nwidth = region_width / width
    defect_nheight = region_height / height
    defect_label = yolo_format(
        defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight
    )

    return frame, defect_label

def draw_random_png_images(frame: np.ndarray = None, defect_class=0, random_images_folder_path:str = None, number_of_images_range:int =[1,3] ,opacity_range=[0.3, 0.7], size_percentage_range=[0.1, 0.3]):    
    number_of_images = random.randint(number_of_images_range[0], number_of_images_range[1])

    images_paths_picked = [img for img in random.sample(os.listdir(random_images_folder_path), number_of_images) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_picked = [cv2.imread(os.path.join(random_images_folder_path, img), cv2.IMREAD_UNCHANGED) for img in images_paths_picked]
    
    final_defect_label = ""
    for img in images_picked:
        opacity = random.uniform(opacity_range[0], opacity_range[1])    
        img_width = int(random.uniform(size_percentage_range[0], size_percentage_range[1]) * frame.shape[1])
        img_height = int(random.uniform(size_percentage_range[0], size_percentage_range[1]) * frame.shape[0])
        img_resized = cv2.resize(img, (img_width, img_height))

        # Generate random position where the crack will be placed
        x_offset = random.randint(0, frame.shape[1] - img_width)
        y_offset = random.randint(0, frame.shape[0] - img_height)

        # Overlay the crack image on the frame
        if img_resized.shape[2] == 4:
            # If the crack image has an alpha channel, blend using the alpha mask
            alpha_mask = img_resized[:, :, 3] / 255.0 * opacity
            for c in range(0, 3):
                frame[y_offset:y_offset + img_height, x_offset:x_offset + img_width, c] = (
                    frame[y_offset:y_offset + img_height, x_offset:x_offset + img_width, c] * (1 - alpha_mask) +
                    img_resized[:, :, c] * alpha_mask
                )
        else:
            # If no alpha channel, use standard blending
            overlay = frame.copy()
            overlay[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img_resized
            frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

        # YOLO label to cover the area where the crack is applied
        defect_nx_center = (x_offset + img_width / 2) / frame.shape[1]
        defect_ny_center = (y_offset + img_height / 2) / frame.shape[0]
        defect_nwidth = img_width / frame.shape[1]
        defect_nheight = img_height / frame.shape[0]
        defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
        final_defect_label += defect_label + "\n"
        
    final_defect_label = final_defect_label[:-1]
    return frame, final_defect_label

def draw_random_png_images_with_circular_opacity(
    frame: np.ndarray = None,
    defect_class=0,
    random_images_folder_path: str = None,
    number_of_images_range: int = [1, 3],
    opacity_range=[0.3, 0.7],
    size_percentage_range=[0.1, 0.3]
    ):
    number_of_images = random.randint(number_of_images_range[0], number_of_images_range[1])

    # Get a list of valid image files
    image_files = [f for f in os.listdir(random_images_folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Randomly select images to overlay
    images_paths_picked = random.sample(image_files, min(number_of_images, len(image_files)))
    images_picked = [cv2.imread(os.path.join(random_images_folder_path, img), cv2.IMREAD_UNCHANGED)
                     for img in images_paths_picked]

    final_defect_label = ""
    for img in images_picked:
        opacity = random.uniform(opacity_range[0], opacity_range[1])
        img_width = int(random.uniform(size_percentage_range[0], size_percentage_range[1]) * frame.shape[1])
        img_height = int(random.uniform(size_percentage_range[0], size_percentage_range[1]) * frame.shape[0])
        img_resized = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)

        # Generate random position where the image will be placed
        x_offset = random.randint(0, frame.shape[1] - img_width)
        y_offset = random.randint(0, frame.shape[0] - img_height)

        # Create the radial gradient mask
        center_x = img_width // 2
        center_y = img_height // 2
        Y, X = np.ogrid[:img_height, :img_width]
        distance_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        # Normalize and invert the distance to get opacity decreasing from center
        mask = (1 - (distance_from_center / max_distance)) * opacity
        mask = np.clip(mask, 0, 1)

        # If the image has an alpha channel, multiply the existing alpha by the mask
        if img_resized.shape[2] == 4:
            alpha_channel = img_resized[:, :, 3] / 255.0
            alpha_mask = alpha_channel * mask
            img_rgb = img_resized[:, :, :3]
        else:
            alpha_mask = mask
            img_rgb = img_resized

        # Expand mask dimensions to match image channels
        alpha_mask_expanded = np.dstack([alpha_mask] * 3)

        # Prepare the region of interest on the frame
        roi = frame[y_offset:y_offset + img_height, x_offset:x_offset + img_width].astype(float)
        img_rgb = img_rgb.astype(float)

        # Blend the image with the frame using the mask
        blended = roi * (1 - alpha_mask_expanded) + img_rgb * alpha_mask_expanded

        # Replace the region on the frame with the blended result
        frame[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = blended.astype(np.uint8)

        # YOLO label to cover the area where the image is applied
        defect_nx_center = (x_offset + img_width / 2) / frame.shape[1]
        defect_ny_center = (y_offset + img_height / 2) / frame.shape[0]
        defect_nwidth = img_width / frame.shape[1]
        defect_nheight = img_height / frame.shape[0]
        defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
        final_defect_label += defect_label + "\n"
    
    final_defect_label = final_defect_label[:-1]
    return frame, final_defect_label