import cv2
import torch
import numpy as np
import copy

FILTER_RATIO = 0.01
KERNEL_ADD_THRESHOLD = 50 # 0.1 * 255
IMAGE_SIZE = 256, 256

train_image_np = np.array(cv2.resize(cv2.imread('train.png'), IMAGE_SIZE))
test_image_np = np.array(cv2.resize(cv2.imread('test.png'), IMAGE_SIZE))

FILTER_SIZE = int(train_image_np.shape[0] * FILTER_RATIO), int(train_image_np.shape[1] * FILTER_RATIO)
print(f"FILTER_SIZE: {FILTER_SIZE}")

kernels = []
for i in range(0, train_image_np.shape[0], 1):
    for j in range(0, train_image_np.shape[1], 1):
        section = copy.deepcopy(train_image_np[i:i+FILTER_SIZE[0], j:j+FILTER_SIZE[1]])
        if section.shape[:2] != FILTER_SIZE:
            continue

        for kernel in kernels:           
            filter_applied_section = section.astype(np.int16) - kernel.astype(np.int16)   
            absolute_mean = np.mean(np.abs(filter_applied_section))
            if absolute_mean < KERNEL_ADD_THRESHOLD:
                break
        else:
            print(f"kernel added at i: {i}, j: {j}, len(kernels): {len(kernels)}")
            kernels.append(section)

        # kernel = train_image_np[i:i+FILTER_SIZE[0], j:j+FILTER_SIZE[1]]
        # cv2.imshow('train_image', section)
        # cv2.waitKey(1)
        # print(f"i: {i}, j: {j}")

while True:
    image_path_to_apply_filter = input('Enter the image path to apply the filter: ')
    image_to_apply_filter = np.array(cv2.resize(cv2.imread(image_path_to_apply_filter), IMAGE_SIZE))
    empty_image = np.zeros(image_to_apply_filter.shape, dtype=np.uint8)

    for i in range(0, image_to_apply_filter.shape[0], FILTER_SIZE[0]):
        for j in range(0, image_to_apply_filter.shape[1], FILTER_SIZE[0]):
            section = copy.deepcopy(image_to_apply_filter[i:i+FILTER_SIZE[0], j:j+FILTER_SIZE[1]])
            print(f"i: {i}, j: {j}")
            if section.shape[:2] != FILTER_SIZE:
                continue

            min_absolute_mean = float('inf')
            best_filter_applied_section = None
            for kernel in kernels:
                filter_applied_section = section.astype(np.int16) - kernel.astype(np.int16)
                absolute_mean = np.mean(np.abs(filter_applied_section))
                if absolute_mean < min_absolute_mean:
                    min_absolute_mean = absolute_mean
                    best_filter_applied_section = np.where(filter_applied_section < 0, 0, filter_applied_section)     
                                
            empty_image[i:i+FILTER_SIZE[0], j:j+FILTER_SIZE[1]] = best_filter_applied_section

    cv2.imshow('empty_image', empty_image)
    cv2.waitKey(0)