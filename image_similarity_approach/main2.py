# https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3
# !pip install git+https://github.com/openai/CLIP.git
# !pip install open_clip_torch
# !pip install sentence_transformers

import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import uuid, os, random

# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)
def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1
def generateScore(test_img, data_img):
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

THRESHOLD_SCORE = 52.5
NUMBER_OF_GROUND_TRUTH_IMAGES = 16
IMAGE_SIZE = (512, 512)
PASS_PROBABILITY = 0.0
THRESHOLD_LOWERING_FACTOR = 0.85

ground_truth_sections = {}

ground_truth_paths = [os.path.join("test_good_images", file_name) for file_name in os.listdir("test_good_images") if file_name.endswith(".png")]
ground_truth_paths = ground_truth_paths[:NUMBER_OF_GROUND_TRUTH_IMAGES]
ground_truth_images = [cv2.imread(file_path, cv2.IMREAD_UNCHANGED) for file_path in ground_truth_paths]

# Fix index calculation in grid
for GRID_FORM in [(3, 3), (4, 4), (5, 5), (6,6), (7,7), (8,8)]:
    ground_truth_sections = {}  # Reset the ground truth sections for the next test image
    GRID_SIZE = (IMAGE_SIZE[0] // GRID_FORM[0], IMAGE_SIZE[1] // GRID_FORM[1])

    for ground_truth_image in ground_truth_images:
        for section_idx in range(GRID_FORM[0] * GRID_FORM[1]):
            if section_idx not in ground_truth_sections:
                ground_truth_sections[section_idx] = []

            x = section_idx % GRID_FORM[1]  # Use columns for mod
            y = section_idx // GRID_FORM[1]  # Use rows for division

            # Append the sliced sections
            ground_truth_sections[section_idx].append(
                ground_truth_image[y * GRID_SIZE[0]:(y + 1) * GRID_SIZE[0], x * GRID_SIZE[1]:(x + 1) * GRID_SIZE[1]]
            )

    # calculate mean threshold by comparing all ground truth images
    mean_threshold = 0
    minimum_threshold = 100
    counter = 0

    for section_index, ground_truth_section_list in ground_truth_sections.items():
        ground_truth_section = ground_truth_section_list[0]  # Use the first section in the list

        max_pooled_similarity_score = 0
        for other_ground_truth_section in ground_truth_section_list[1:]:
            if random.random() < PASS_PROBABILITY:
                continue

            score = generateScore(ground_truth_section, other_ground_truth_section)
            if score > max_pooled_similarity_score:
                max_pooled_similarity_score = score

        if max_pooled_similarity_score < minimum_threshold and max_pooled_similarity_score != 0:
            minimum_threshold = max_pooled_similarity_score
            mean_threshold += max_pooled_similarity_score
            counter += 1

        print(f"Section max score: {max_pooled_similarity_score}, All sections min score: {minimum_threshold}")
    if counter > 0:
        mean_threshold /= counter
    else:
        mean_threshold = 0  # Avoid division by zero if no sections were processed

    print(f"Mean threshold: {mean_threshold}")
    THRESHOLD_SCORE = minimum_threshold*THRESHOLD_LOWERING_FACTOR

    for test_image_path in [os.path.join("test_defect_images", file_name) for file_name in os.listdir("test_defect_images") if file_name.endswith(".png")]:

        test_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)
        test_image = cv2.resize(test_image, IMAGE_SIZE)  # Ensure the test image is resized

        # Split the test image into sections
        test_image_sections = []
        for i in range(GRID_FORM[0]):
            for j in range(GRID_FORM[1]):
                test_image_sections.append(test_image[i*GRID_SIZE[0]:(i+1)*GRID_SIZE[0], j*GRID_SIZE[1]:(j+1)*GRID_SIZE[1]])

        # Compare the test image sections with the ground truth sections
        below_threshold_sections = []
        for section_index, ground_truth_section_list in ground_truth_sections.items():
            max_pooled_similarity_score = 0
            test_image_section = test_image_sections[section_index]
            cv2.imshow("Test Image Section", cv2.resize(test_image_section, IMAGE_SIZE))
            cv2.waitKey(1)

            # Compare the current test section with all ground truth sections
            for ground_truth_section in ground_truth_section_list:
                similarity_score = generateScore(test_image_section, ground_truth_section)
                print(f"    {section_index} | similarity Score: {round(similarity_score,2)} | Threshold: {THRESHOLD_SCORE:.2f}")
                if similarity_score >= THRESHOLD_SCORE:
                    max_pooled_similarity_score = similarity_score
                    print(f" Already above threshold, skipping the rest")
                    break

                # Keep track of the highest similarity score found
                if similarity_score > max_pooled_similarity_score:
                    max_pooled_similarity_score = similarity_score
            
            # After comparing all ground truth sections, check if the max score is below the threshold
            if max_pooled_similarity_score < THRESHOLD_SCORE:
                below_threshold_sections.append(section_index)
            
            print(f"{section_index} | max pooled similarity Score: {round(max_pooled_similarity_score,2)}")

        print(f"Below threshold sections: {below_threshold_sections}")
        for section_index in below_threshold_sections:
            x = section_index % GRID_FORM[1]
            y = section_index // GRID_FORM[1]
            cv2.rectangle(test_image, (x*GRID_SIZE[1], y*GRID_SIZE[0]), ((x+1)*GRID_SIZE[1], (y+1)*GRID_SIZE[0]), (0, 0, 255), 2)

        cv2.imshow("Test Image", test_image)
        cv2.waitKey(10000)
        cv2.imwrite(f"results/g_{GRID_FORM[0]}_{uuid.uuid4()}.jpg", test_image)
        cv2.destroyAllWindows()
