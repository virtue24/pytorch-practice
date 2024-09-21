import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple

def calculate_cosine_similarity_between_frames(rgb_frame_1: np.ndarray, rgb_frame_2: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    cos(theta) = (v1.v2) / (||v1|| ||v2||)

    :param v1: np.ndarray
    :param v2: np.ndarray

    :return: float cosine similarity [-1, 1]
    """
    v1 = rgb_frame_1.flatten().astype(np.float64)
    v2 = rgb_frame_2.flatten().astype(np.float64) 
    
    v1_dot_v2 = np.dot(v1, v2)   
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Check for zero vectors to avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        print("One of the vectors is zero, cosine similarity is not defined.")
        return 0.0  # You can return 0, NaN, or handle this case as needed

    cosine_similarity = v1_dot_v2 / (norm_v1 * norm_v2)
    return cosine_similarity

#====================== EXPERIMENTAL FUNCTIONS ======================#
def generate_rgb_frame(n: int) -> np.ndarray:
    """
    Generate an RGB frame of size nxn with random values.

    :param n: int, size of the frame (nxn)
    :return: np.ndarray, RGB frame of size nxn
    """
    return np.random.randint(0, 256, size=(n, n, 3), dtype=np.uint8)

def random_pixel_change(image, P):
    """
    Randomly changes pixels of an RGB image to a new random RGB value with probability P.

    Args:
    - image (np.array): Input image as an (n x n x 3) NumPy array.
    - P (float): Probability of changing each pixel (between 0 and 1).

    Returns:
    - np.array: The modified image.
    """
    # Copy the image to avoid modifying the original one
    modified_image = image.copy()

    # Create a random mask of the same shape as the image but only for the n x n pixels
    mask = np.random.rand(image.shape[0], image.shape[1]) < P

    # Generate random RGB values for all pixels
    random_colors = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)

    # Apply the mask to change only the selected pixels
    modified_image[mask] = random_colors[mask]

    return modified_image

def random_pixel_change_with_same_color(image: np.ndarray, P: float) -> np.ndarray:
    """
    Randomly changes pixels of an RGB image to a single new random RGB value with probability P.

    Args:
    - image (np.array): Input image as an (n x n x 3) NumPy array.
    - P (float): Probability of changing each pixel (between 0 and 1).

    Returns:
    - np.array: The modified image.
    """
    # Validate probability
    if not 0 <= P <= 1:
        raise ValueError("Probability P must be between 0 and 1.")

    # Copy the image to avoid modifying the original one
    modified_image = image.copy()

    # Create a random mask where each pixel has a P chance to be altered
    mask = np.random.rand(image.shape[0], image.shape[1]) < P

    # Generate a single random RGB color (shape: (1, 1, 3))
    random_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)

    # Apply the mask: set selected pixels to the random color
    modified_image[mask] = random_color

    return modified_image

def run_similarity_test(
    image_size: int,
    p_values: np.ndarray,
    n_trials: int,
    show_frames: bool = False,
    image_display_size: int = 360,
    wait_time: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a test by incrementing P and averaging cosine similarity scores over N trials.

    :param image_size: Size of the RGB frame (n x n).
    :param p_values: Array of P values to test.
    :param n_trials: Number of trials per P value.
    :param show_frames: Whether to display frames using OpenCV.
    :param image_display_size: Size to display images (width and height).
    :param wait_time: Time to wait between frame displays (in milliseconds).
    :return: Tuple containing the array of P values and their corresponding average cosine similarities.
    """
    average_cosine_similarities = []
    min_cosine_similarities = []

    # Generate a fixed RGB frame for consistency across trials (optional)
    # Alternatively, generate a new frame for each trial
    original_frame = generate_rgb_frame(image_size)

    for P in p_values:
        cosine_similarities = []
        for trial in range(n_trials):
            # Apply random pixel changes with probability P
            #distorted_frame = random_pixel_change(original_frame, P)
            distorted_frame = random_pixel_change_with_same_color(original_frame, P)

            # Calculate cosine similarity
            cosine_similarity = calculate_cosine_similarity_between_frames(original_frame, distorted_frame)
            cosine_similarities.append(cosine_similarity)

        # Compute average cosine similarity for current P
        avg_cos_sim = np.mean(cosine_similarities)
        min_cos_sim = np.min(cosine_similarities)
        average_cosine_similarities.append(avg_cos_sim)
        min_cosine_similarities.append(min_cos_sim)

        print(f"P={P:.2f}: Average Cosine Similarity over {n_trials} trials = {avg_cos_sim:.4f} and Min Cosine Similarity = {min_cos_sim:.4f}")

        # Optionally display frames for specific P values (e.g., P=0.0, 0.5, 1.0)
        if show_frames:
            # Resize images for display
            resized_original = cv2.resize(original_frame, (image_display_size, image_display_size), interpolation=cv2.INTER_NEAREST)
            resized_distorted = cv2.resize(distorted_frame, (image_display_size, image_display_size), interpolation=cv2.INTER_NEAREST)

            # Display images
            cv2.imshow(f"Original RGB Frame", resized_original)
            cv2.imshow(f"Distorted RGB Frame", resized_distorted)
            cv2.waitKey(wait_time)  # 0 means wait indefinitely, adjust as needed

    # Close all OpenCV windows if any were opened
    if show_frames:
        cv2.destroyAllWindows()

    return p_values, np.array(average_cosine_similarities), np.array(min_cosine_similarities)

def plot_results(p_values: np.ndarray, average_cosine_similarities: np.ndarray):
    """
    Plots the average cosine similarity against P values.

    :param p_values: Array of P values.
    :param average_cosine_similarities: Corresponding average cosine similarities.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, average_cosine_similarities, marker='o', linestyle='-', color='b')
    plt.title('Average Cosine Similarity vs. Probability P of Pixel Change')
    plt.xlabel('Probability P of Pixel Change')
    plt.ylabel('Average Cosine Similarity')
    plt.grid(True)
    plt.ylim(-1, 1)  # Cosine similarity ranges from -1 to 1
    plt.show()

if __name__ == "__main__":
    # Parameters
    image_size = 50  # Size of the RGB frame (n x n)
    N_trials = 1000    # Number of trials per P value
    P_start = 0.0
    P_end = 1.0
    P_step = 0.01
    show_frames = True  # Set to True to display frames
    image_display_size = 360
    wait_time = 1  # in milliseconds; adjust as needed

    # Generate array of P values
    p_values = np.arange(P_start, P_end + P_step, P_step)

    # Run the similarity test
    p_vals, avg_cos_sims, min_cos_sims = run_similarity_test(
        image_size=image_size,
        p_values=p_values,
        n_trials=N_trials,
        show_frames=show_frames,
        image_display_size=image_display_size,
        wait_time=wait_time
    )

    # Plot the results
    plot_results(p_vals, avg_cos_sims)
    plot_results(p_vals, min_cos_sims)



