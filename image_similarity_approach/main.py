# https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3
# !pip install git+https://github.com/openai/CLIP.git
# !pip install open_clip_torch
# !pip install sentence_transformers

import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
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

import os
while True:
    similarity_threshold = 85
    good_images = [cv2.imread(f"base_test_good_images/{file_name}") for file_name in os.listdir("base_test_good_images") if file_name.endswith(".png")]

    test_image = cv2.imread(input("Enter the path of image1: "), cv2.IMREAD_UNCHANGED)
    for data_image in good_images:
        print(f"similarity Score: ", round(generateScore(test_image, data_image), 2))


#95.75 (0, 1)
#96.72 (0, 2)
#95.44 (0, 3)
#84.41 (0, 10) red vs blue
#96.79 (0, 0g)




