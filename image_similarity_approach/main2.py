import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import uuid

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

THRESHOLD_SCORE = 85
IMAGE_SIZE = (240, 240)
GRID_FORM = (4, 4)
GRID_SIZE = (IMAGE_SIZE[0]//GRID_FORM[0], IMAGE_SIZE[1]//GRID_FORM[1])

ground_truth_sections = {}

ground_truth_0 = cv2.resize(cv2.imread("test_good_images/6dd496b1-2036-4fec-8ccb-3f4c53df00dd.jpg", cv2.IMREAD_UNCHANGED), IMAGE_SIZE)
ground_truth_1 = cv2.resize(cv2.imread("test_good_images/7ecd279b-3121-41b5-b244-43b640a3e7db.jpg", cv2.IMREAD_UNCHANGED), IMAGE_SIZE)
ground_truth_2 = cv2.resize(cv2.imread("test_good_images/d340a2e0-4659-4a0c-a355-2d91ffe25fb8.jpg", cv2.IMREAD_UNCHANGED), IMAGE_SIZE)
ground_truth_3 = cv2.resize(cv2.imread("test_good_images/fcba0125-3d64-41c1-8011-ec33b54614d5.jpg", cv2.IMREAD_UNCHANGED), IMAGE_SIZE)

# Fix index calculation in grid
for ground_truth_image in [ground_truth_0, ground_truth_1, ground_truth_2, ground_truth_3]:
    for section_idx in range(GRID_FORM[0]*GRID_FORM[1]):
        if section_idx not in ground_truth_sections:
            ground_truth_sections[section_idx] = []
        
        x = section_idx % GRID_FORM[1]  # Use columns for mod
        y = section_idx // GRID_FORM[1]  # Use rows for division
        
        # Append the sliced sections
        ground_truth_sections[section_idx].append(
            ground_truth_image[y*GRID_SIZE[0]:(y+1)*GRID_SIZE[0], x*GRID_SIZE[1]:(x+1)*GRID_SIZE[1]]
        )

# # Display ground truth sections
# for section_index, section_images in ground_truth_sections.items():
#     for ground_truth_section in section_images:
#         cv2.imshow("Ground Truth", ground_truth_section)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

while True:
    test_image = cv2.imread(input("Enter the path of test image: "), cv2.IMREAD_UNCHANGED)
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
        
        # Compare the current test section with all ground truth sections
        for ground_truth_section in ground_truth_section_list:
            similarity_score = generateScore(test_image_section, ground_truth_section)
            print(f"    {section_index} | similarity Score: {round(similarity_score,2)}")
            
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
    cv2.waitKey(0)
    cv2.imwrite(f"test_images/{uuid.uuid4()}.jpg", test_image)
    cv2.destroyAllWindows()