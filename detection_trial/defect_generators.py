import cv2
import numpy as np
import random, string
import math

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

def draw_rectangle_on_frame(frame: np.ndarray = None, defect_class=1, R_range=[0, 255], G_range=[0, 255], B_range=[0, 255], thickness_range=[1, 10], opacity_range=[0.1, 1.0], infill_probability=0.5):
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

def add_gaussian_noise(frame: np.ndarray = None, defect_class=2, mean=0, var=0.01):
    """Adds Gaussian noise to the frame."""
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, frame.shape).astype('uint8')
    noisy_frame = cv2.add(frame, gauss)
    
    # Since noise is over the entire image, YOLO label covers the whole image
    defect_label = yolo_format(defect_class, 0.5, 0.5, 1.0, 1.0)
    
    return noisy_frame, defect_label

def add_salt_and_pepper_noise(frame: np.ndarray = None, defect_class=3, amount=0.005, s_vs_p=0.5, dimension_range=[0.1, 0.3]):
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

def add_blurred_region(frame: np.ndarray = None, defect_class=5, kernel_size_range=[5, 31]):
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

def add_scratch_on_frame(frame: np.ndarray = None, defect_class=6, num_lines=5, length_range=[10, 100], thickness_range=[1, 3], color=(0, 0, 0)):
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

def add_watermark_text(frame: np.ndarray = None, defect_class=7, text_length = 10, font_scale=2, thickness=3, opacity=0.3):
    """Adds watermark text on the frame."""
    characters = string.ascii_letters + string.digits + string.punctuation
    text = ''.join(random.choice(characters) for i in range(text_length))
    
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = random.randint(0, frame.shape[1] - text_size[0])
    y = random.randint(text_size[1], frame.shape[0])
    cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # YOLO format covering the text area
    defect_nx_center = (x + text_size[0] / 2) / frame.shape[1]
    defect_ny_center = (y - text_size[1] / 2) / frame.shape[0]
    defect_nwidth = text_size[0] / frame.shape[1]
    defect_nheight = text_size[1] / frame.shape[0]
    defect_label = yolo_format(defect_class, defect_nx_center, defect_ny_center, defect_nwidth, defect_nheight)
    
    return frame, defect_label

def add_lens_flare(frame: np.ndarray = None, defect_class=8, flare_center=None, radius_range=[50, 150], opacity_range=[0.1, 0.5]):
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

def add_shadow(frame: np.ndarray = None, defect_class=9, top_left_ratio=(0.2, 0.2), bottom_right_ratio=(0.8, 0.8), opacity=0.5):
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

def add_color_tint(frame: np.ndarray = None, defect_class=10, tint_color=(0, 255, 255), opacity=0.3):
    """Adds a color tint to the frame."""
    overlay = np.full(frame.shape, tint_color, dtype=np.uint8)
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # Since the tint is over the whole image
    defect_label = yolo_format(defect_class, 0.5, 0.5, 1.0, 1.0)
    
    return frame, defect_label

def add_crack_texture(frame: np.ndarray = None, defect_class=11, crack_image_path='crack.png', opacity_range=[0.3, 0.7], size_percentage_range=[0.1, 0.3]):
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

def add_dust_particles(frame: np.ndarray = None, defect_class=12, num_particles=50, size_range=[1, 5]):
    """Adds dust particles to the frame."""
    overlay = frame.copy()
    for _ in range(num_particles):
        x = random.randint(0, frame.shape[1] - 1)
        y = random.randint(0, frame.shape[0] - 1)
        size = random.randint(size_range[0], size_range[1])
        cv2.circle(overlay, (x, y), size, (255, 255, 255), -1)
    opacity = 0.5
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # YOLO label covers the entire image
    defect_label = yolo_format(defect_class, 0.5, 0.5, 1.0, 1.0)
    
    return frame, defect_label

def add_fog_effect(frame: np.ndarray = None, defect_class=13, fog_density=0.5):
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

def draw_random_png_images():
    """Draws random small png images."""
    #TODO: Implement this function. Create a folder consisting of random small png images.


  