import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple
from pathlib import Path
import pprint

from cosine_similarity_module import calculate_cosine_similarity_between_frames

class Dataset:
    def __init__(self, dataset_folder_path:Path=None):
        self.DATASET_FOLDER_PATH = dataset_folder_path

        extensions = ['jpg', 'png', 'gif']
        self.IMAGE_PATHS = [img for ext in extensions for img in self.DATASET_FOLDER_PATH.glob(f'*.{ext}')]
        self.FRAMES = []

    def load_frames(self, image_size: Tuple[int, int] = (640, 640), n_zoom_bbox: Tuple[int, int, int, int] = (0, 0, 1, 1)):
        self.FRAMES = []
        for image_path in self.IMAGE_PATHS:
            image = cv2.imread(str(image_path))
            
            x0, y0, x1, y1 = n_zoom_bbox
            zoomed_section = image[int(y0 * image.shape[0]):int(y1 * image.shape[0]), int(x0 * image.shape[1]):int(x1 * image.shape[1])]
            zoomed_section = cv2.resize(zoomed_section, image_size)
            
            self.FRAMES.append(zoomed_section)
        print(f"[INFO] Loaded {len(self.FRAMES)} frames from the dataset folder.")
        
    def get_number_of_frames(self):
        return len(self.FRAMES)

    def get_frame(self, frame_index:int):
        return self.FRAMES[frame_index]
    
    def get_frames(self):
        return self.FRAMES
    
    def show_frame(self, frame_index:int):
        cv2.imshow('Frame', self.FRAMES[frame_index])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_frame_section(self, frame_index:int = None, x:int=None, y:int=None, kernel_size:Tuple[int, int, int] = None):
        return self.FRAMES[frame_index][y:y+kernel_size[1], x:x+kernel_size[0]]
    
class ImageSection:
    def __init__(self, section_size: Tuple[int, int]= ( 5, 5, 3 )  ):
        self.SIZE = section_size
        self.image_section = np.random.randint(0, 256, size=section_size, dtype=np.uint8)

    def set_image_section(self, new_image_section: np.ndarray = None):
        if not new_image_section.shape == self.SIZE:
            raise ValueError(f"New image section shape {new_image_section.shape} does not match the expected shape {self.SIZE}")
        self.image_section = new_image_section

    def get_image_section(self):
        return self.image_section
    
train_dataset_folder_path = Path(__file__).resolve().parent / 'base_train_images'
evaluation_dataset_folder_path = Path(__file__).resolve().parent / 'base_defect_images'
print(f"[INFO] Train dataset folder path: {train_dataset_folder_path}")
print(f"[INFO] Evaluation dataset folder path: {evaluation_dataset_folder_path}")

train_dataset_manager = Dataset(dataset_folder_path=train_dataset_folder_path)
evaluation_dataset_manager = Dataset(dataset_folder_path=evaluation_dataset_folder_path)

IMAGE_SIZE = (512, 512)
N_ZOOM_BBOX = (0.3, 0.1, 0.7, 0.9) # (x0, y0, x1, y1)
KERNEL_SIZE = (8, 8, 3) #must divide the image size without remainder
SIMILARITY_THRESHOLD = 0.98
#NUMBER_OF_MAX_IMAGE_SECTIONS_PER_PARTITION = 8

train_dataset_manager.load_frames(image_size = IMAGE_SIZE, n_zoom_bbox=N_ZOOM_BBOX) # Load all frames from the dataset folder with the specified transformations
evaluation_dataset_manager.load_frames(image_size = IMAGE_SIZE, n_zoom_bbox=N_ZOOM_BBOX) # Load all frames from the dataset folder with the specified transformations


# Learn image sections
image_partitions = {}
for y in range(0, IMAGE_SIZE[0], KERNEL_SIZE[0] ):
    for x in range(0, IMAGE_SIZE[1], KERNEL_SIZE[1] ):
        image_partitions[(x, y)] = []

for x_y, value in image_partitions.items():
    for frame_index in range(train_dataset_manager.get_number_of_frames()):
        image_section = train_dataset_manager.get_frame_section(frame_index, x_y[0], x_y[1], KERNEL_SIZE)
        image_partitions[x_y].append(image_section)

image_partitions_reduced = {}
for x_y, value in image_partitions.items():
    reduced_sections = []
    for section in value:
        for reduced_section in reduced_sections:
            cosine_similarity = calculate_cosine_similarity_between_frames(reduced_section, section)
            if cosine_similarity > SIMILARITY_THRESHOLD:
                break
        else:
            reduced_sections.append(section)
    image_partitions_reduced[x_y] = reduced_sections

for x_y, value in image_partitions_reduced.items():
    if len(value) > 3:
        print(f"X: {x_y[0]}, Y: {x_y[1]}, Number of sections: {len(value)}")

# Evaluate the result
def continuous_color_mapping(resized_result: np.ndarray) -> np.ndarray:
    """
    Maps grayscale pixel values to colors based on the mean value with continuous gradients.
    
    - min_value: Red
    - intermediate values: Gradient from Red to White
    - max_value: Green

    Args:
    - resized_result (np.ndarray): The resized result array (H x W x 3).
    
    Returns:
    - np.ndarray: Color-mapped image (H x W x 3).
    """
    # Ensure the image has three channels
    if len(resized_result.shape) != 3 or resized_result.shape[2] != 3:
        raise ValueError("resized_result must be a 3-channel (H x W x 3) image.")
    
    # Convert to grayscale by taking one channel (assuming all channels are equal)
    gray = resized_result[:, :, 0].astype(np.float64)

    threshold_low = np.min(gray)
    mean = np.mean(gray)
    threshold_high = np.max(gray)
    print(f"Threshold Low: {threshold_low:.2f}")
    print(f"Threshold High: {threshold_high:.2f}")
    
    color_mapped = np.where(gray > mean * 0.99, 255, 0).astype(np.uint8)
    
    return color_mapped
    
test_image = evaluation_dataset_manager.get_frame(1)
for test_image in evaluation_dataset_manager.get_frames():
    cv2.imshow('Test Image', test_image)
    cv2.waitKey(0)

    # Initialize the result array
    num_rows = IMAGE_SIZE[0] // KERNEL_SIZE[0]
    num_cols = IMAGE_SIZE[1] // KERNEL_SIZE[1]
    result_array = np.zeros((num_rows, num_cols, 3), dtype=np.uint8)

    for (x, y), sections in image_partitions_reduced.items():
        # Extract the image section using y and x
        test_image_section = test_image[
            y:y + KERNEL_SIZE[1],
            x:x + KERNEL_SIZE[0]
        ].copy()
        
        max_cosine_similarity = 0.0
        
        for section in sections:
            cosine_similarity = calculate_cosine_similarity_between_frames(test_image_section, section)
            if cosine_similarity > max_cosine_similarity:
                max_cosine_similarity = cosine_similarity
        
        # Clamp the cosine similarity to [0, 1] just in case
        max_cosine_similarity = max(0.0, min(1.0, max_cosine_similarity))
        
        # Calculate grayscale value
        gray_value = int(max_cosine_similarity * 255)
        
        # Determine the position in result_array
        row = y // KERNEL_SIZE[0]
        col = x // KERNEL_SIZE[1]
        
        # Assign the grayscale value to all RGB channels
        result_array[row, col] = [gray_value, gray_value, gray_value]

    DISPLAY_SIZE = (480, 480)  # Adjust as needed
    result_array = continuous_color_mapping(resized_result=result_array)
    resized_result = cv2.resize(result_array, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)

    # Display the result
    cv2.imshow('Result', resized_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



        





