import numpy as np
import cv2
import copy, random

def create_bold_white_line(image: np.ndarray, IMAGE_SIZE: tuple = (572, 572), MASK_SIZE: tuple = (388, 388)) -> np.ndarray:
    """
    Create a bold line on the image and a corresponding defect mask with one color channel.
    """    
    line_stroke_range = (5, 25)

    # create images
    white_lined_image = copy.deepcopy(image)
    
    # Create a defect mask with a single channel (grayscale)
    defect_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # define line start and end points
    start_point = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))
    end_point = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))

    # draw lines
    cv2.line(defect_mask, start_point, end_point, 255, random.randint(line_stroke_range[0], line_stroke_range[1]))  # Single-channel line (white)
    cv2.line(white_lined_image, start_point, end_point, (255, 255, 255), random.randint(line_stroke_range[0], line_stroke_range[1]))  # RGB line

    # resize images
    white_lined_image = cv2.resize(white_lined_image, IMAGE_SIZE)
    defect_mask = cv2.resize(defect_mask, MASK_SIZE)  # Single-channel mask

    # Add a channel dimension to the mask to make it (388, 388, 1)
    defect_mask = defect_mask[:, :, np.newaxis]

    return white_lined_image, defect_mask


