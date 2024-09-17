import unet_model
import torch
from torch import optim, nn
from tqdm import tqdm

import numpy as np
import cv2
import random, time
import unet_model
import defect_generator

MODEL_SAVE_PATH = "unet.pth"
IMAGE_SIZE = (572, 572)
MASK_SIZE = (388, 388)

NUMBER_OF_DEFECTED_IMAGES = 10
BASE_IMAGE_NP = np.array(cv2.imread('base_image.png'))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = [] # [(defected_image, defect_mask), ...] where defected_image and defect_mask are numpy arrays
for _ in range(NUMBER_OF_DEFECTED_IMAGES):
    defected_image, defect_mask = defect_generator.create_bold_white_line(BASE_IMAGE_NP, IMAGE_SIZE, MASK_SIZE)
    dataset.append((defected_image, defect_mask))
dataset_tensor = [(torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float(), torch.tensor(mask / 255.0).permute(2, 0, 1).unsqueeze(0).float()) for img, mask in dataset]


model = unet_model.UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.to(DEVICE)

for sample_no, sample in enumerate(dataset_tensor):
    img = sample[0].to(DEVICE)
    mask = sample[1].to(DEVICE)

    y_pred = model(img)
    cv2.imshow("Input Image", img[0].permute(1, 2, 0).cpu().numpy())
    cv2.imshow("Predicted Mask", y_pred[0][0].detach().cpu().numpy())
    cv2.imshow("Ground Truth Mask", mask[0][0].cpu().numpy())
    cv2.waitKey(2500)




