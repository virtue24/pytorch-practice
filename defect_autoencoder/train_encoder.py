import torch
from torch import optim, nn
from tqdm import tqdm

import numpy as np
import cv2
import random, time
import unet_model
import defect_generator

IMAGE_SIZE = (572, 572)
MASK_SIZE = (388, 388)

NUMBER_OF_DEFECTED_IMAGES = 50
TRAIN_PERCENTAGE = 0.8
BASE_IMAGE_NP = np.array(cv2.imread('base_image.png'))

dataset = [] # [(defected_image, defect_mask), ...] where defected_image and defect_mask are numpy arrays
for _ in range(NUMBER_OF_DEFECTED_IMAGES):
    defected_image, defect_mask = defect_generator.create_bold_white_line(BASE_IMAGE_NP, IMAGE_SIZE, MASK_SIZE)
    dataset.append((defected_image, defect_mask))

train_dataset = dataset[:int(NUMBER_OF_DEFECTED_IMAGES * TRAIN_PERCENTAGE)]
val_dataset = dataset[int(NUMBER_OF_DEFECTED_IMAGES * TRAIN_PERCENTAGE):]

train_dataset_tensor = [(torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float(), torch.tensor(mask / 255.0).permute(2, 0, 1).unsqueeze(0).float()) for img, mask in train_dataset]
val_dataset_tensor = [(torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float(), torch.tensor(mask / 255.0).permute(2, 0, 1).unsqueeze(0).float()) for img, mask in val_dataset]

LEARNING_RATE = 3e-4
EPOCHS = 100
MODEL_SAVE_PATH = "unet.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = unet_model.UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_running_loss = 0
    for train_sample_no, train_sample in enumerate(tqdm(train_dataset_tensor)):
        
        img = train_sample[0].to(device)
        mask = train_sample[1].to(device)        

        y_pred = model(img)
        optimizer.zero_grad()

        loss = criterion(y_pred, mask)
        train_running_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (train_sample_no + 1)
    
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for validation_sample_no, validation_sample in enumerate(tqdm(val_dataset_tensor)):

            img = validation_sample[0].to(device)
            mask = validation_sample[1].to(device)      
            
            y_pred = model(img)
            loss = criterion(y_pred, mask)

            val_running_loss += loss.item()

        val_loss = val_running_loss / (validation_sample_no + 1)

    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print("-"*30)

torch.save(model.state_dict(), MODEL_SAVE_PATH)