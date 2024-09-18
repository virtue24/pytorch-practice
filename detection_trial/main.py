import os, shutil, datetime, yaml, uuid, random
from pathlib import Path
import cv2
import numpy as np

from ultralytics import YOLO
import torch

import defect_generators

DATASET_FOLDER_NAME = "dataset" # Never change this
NUMBER_OF_DEFECTS = 240 
IMAGE_ZOOM_REGION = [0.3, 0.3, 0.7, 0.7]
DESIRED_IMAGE_SIZE = (640, 640)
DEFECTS_PER_IMAGE_RANGE = [1, 4+1] # last number is exclusive

def paste_image_and_label(image:np.ndarray = None, name_prefix:str = None, label = "", dataset_folder_name:str =None):
    sample_uuid = str(uuid.uuid4())
    image_name = name_prefix+"_"+sample_uuid + ".jpg" if name_prefix else sample_uuid + ".jpg"
    label_name = name_prefix+"_"+sample_uuid + ".txt" if name_prefix else sample_uuid + ".txt"

    # Save the image
    image_path = os.path.join(dataset_folder_name, "images", image_name)
    cv2.imwrite(image_path, image)
    # Save the label
    label_path = os.path.join(dataset_folder_name, "labels", label_name)
    with open(label_path, "w") as f:
        f.write(label)

def recreate_dataset_folder(dataset_folder_name = None):
    # clear the dataset folder or create it if it does not exist
    if os.path.exists(dataset_folder_name):
        file_list = os.listdir(dataset_folder_name)
        for file_name in file_list:
            file_path = os.path.join(dataset_folder_name, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.mkdir(dataset_folder_name)
  
    # create the dataset.yaml file
    data = {
        'nc': 1,  # number of classes
        'names': ['defect'],  # class names
        'train': str(Path(__file__).resolve().parent / dataset_folder_name / 'images'),  # path to training images
        'val': str(Path(__file__).resolve().parent / dataset_folder_name / 'images'),  # path to training images
    }

    YAML_PATH = Path(__file__).resolve().parent / dataset_folder_name / 'dataset.yaml'
    with open(YAML_PATH, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    # create the classes.txt and predefined_classes.txt file
    with open(os.path.join(dataset_folder_name, "predefined_classes.txt"), "w") as f:
        f.write("defect")
        # Add more classes if needed
    os.mkdir(os.path.join(dataset_folder_name, "images"))
    os.mkdir(os.path.join(dataset_folder_name, "labels"))
    with open(os.path.join(dataset_folder_name, "labels", "classes.txt"), "w") as f:
        f.write("defect")
        # Add more classes if needed

def recreate_dataset_images_and_labels(dataset_folder_name=None, number_of_defects:int = 100, defects_per_image_range = [1,5], desired_image_size = (640, 640), image_zoom_region = [0.0, 0.0, 1.0, 1.0]):
    BASE_IMAGES_FOLDER_PATH = Path(__file__).resolve().parent / 'base_images'
    base_image_paths = [BASE_IMAGES_FOLDER_PATH / f for f in os.listdir(BASE_IMAGES_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    base_image_paths = [str(image_path.resolve()) for image_path in base_image_paths]

    # crop and resize the base images
    base_images = [cv2.imread(image_path) for image_path in base_image_paths]
    processed_images = []
    for image in base_images:
        height, width = image.shape[:2]
        
        x_min = int(width * IMAGE_ZOOM_REGION[0])
        y_min = int(height * IMAGE_ZOOM_REGION[1])
        x_max = int(width * IMAGE_ZOOM_REGION[2])
        y_max = int(height * IMAGE_ZOOM_REGION[3])
        
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))

        cropped_image = image[y_min:y_max, x_min:x_max]
        resized_image = cv2.resize(cropped_image, DESIRED_IMAGE_SIZE)
        processed_images.append(resized_image)
    base_images = processed_images

    # add background images to the dataset
    for image in base_images:
        paste_image_and_label(image = image, name_prefix = "background", label = "", dataset_folder_name = dataset_folder_name)
    
    # add defect images to the dataset
    DEFECT_GENERATORS = [
        [True, defect_generators.draw_line_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "thickness_range": [2, 13], "opacity_range": [0.5, 1.0]}],
        [True, defect_generators.draw_circle_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "thickness_range": [2, 13], "opacity_range": [0.5, 1.0], "radius_range": [10, 50]}],
        [True, defect_generators.draw_copy_paste_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "ncut_length_range": [0.05, 0.2], "ncopy_region": [0.33,0.1, 0.66, 0.9]}],
        [True, defect_generators.draw_rectangle_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "thickness_range": [2, 13], "opacity_range": [0.5, 1.0], "infill_probability": 0.75}],
        [False, defect_generators.add_gaussian_noise, {"defect_class": 0, "mean": 0, "var": 0.01}],
        [True, defect_generators.add_salt_and_pepper_noise, {"defect_class": 0, "amount": 0.015, "s_vs_p": 0.5, "dimension_range": [0.1, 0.3]}],        
        [True, defect_generators.draw_ellipse_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "angle_range": [0, 360], "axes_length_range": [10, 100], "thickness_range": [3, 10], "opacity_range": [0.4, 1.0]}],
        [False, defect_generators.add_blurred_region, {"defect_class": 0, "kernel_size_range": [5, 31]}],
        [True, defect_generators.add_scratch_on_frame, {"defect_class": 0, "num_lines": 5, "length_range": [30, 150], "thickness_range": [3, 10], "color": (random.randint(75, 255), random.randint(75, 255), random.randint(75, 255))}],
        [True, defect_generators.add_watermark_text, {"defect_class": 0, "text_length": 10, "font_scale": 2, "thickness": 3, "opacity": 0.3}],
        [False, defect_generators.add_lens_flare, {"defect_class": 0, "flare_center": None, "radius_range": [50, 150], "opacity_range": [0.1, 0.5]}],
        [False, defect_generators.add_shadow, {"defect_class": 0, "top_left_ratio": (0.2, 0.2), "bottom_right_ratio": (0.8, 0.8), "opacity": 0.5}],
        [False, defect_generators.add_color_tint, {"defect_class": 0, "tint_color": (0, 255, 255), "opacity": 0.3}],
        [True, defect_generators.add_crack_texture, {"defect_class": 0, "crack_image_path": "src_image/crack.png", "opacity_range": [0.7, 0.95], "size_percentage_range": [0.25, 0.4]}],
        [False, defect_generators.add_dust_particles, {"defect_class": 0, "num_particles": 50, "size_range": [1, 5]}],
        [False, defect_generators.add_fog_effect, {"defect_class": 0, "fog_density": 0.5}],
    ]       
    ACTIVE_DEFECT_GENERATORS = [defect_generator for defect_generator in DEFECT_GENERATORS if defect_generator[0]]

    for _ in range(number_of_defects):
        frame = base_images[np.random.randint(0, len(base_images))].copy()
        number_of_defects_applied = np.random.randint(defects_per_image_range[0], defects_per_image_range[1])
        
        final_defect_label = ""
        for __ in range(number_of_defects_applied):
            defect_to_generate = ACTIVE_DEFECT_GENERATORS[np.random.randint(0, len(ACTIVE_DEFECT_GENERATORS))]
            defect_function = defect_to_generate[1]
            defect_function_kwargs = defect_to_generate[2]

            frame, defect_label = defect_function(frame = frame, **defect_function_kwargs)
            final_defect_label += defect_label+"\n"
        final_defect_label = final_defect_label[:-1]

        paste_image_and_label(image = frame, label = final_defect_label, dataset_folder_name = dataset_folder_name)

def train_yolo_model(dataset_folder_name = None):
    print(f" torch.cuda.device_count(): {torch.cuda.device_count()}")
    torch.cuda.set_device(0) # Set to your desired GPU number

    is_existing_model = True if input("Do you have an existing model to train on? (y/n): ")=="y" else False
    if is_existing_model:
        #==== Option 1: Build from YAML and transfer pretrained weights
        model_path_to_train_on = input("Enter the path to the model to train on ( original one is not effected ) : ")
        model = YOLO('yolov8n.yaml').load(model_path_to_train_on)
    else:
        #==== Option 2: Train directly from the model definition
        model = YOLO('yolov8n.yaml')

    RUN_ON_CUDA = True
    if RUN_ON_CUDA and torch.cuda.is_available():
        model.to('cuda')
        print("GPU (CUDA) is detected. Training will be done on GPU.")
    else:
        r = input("GPU (CUDA) is not detected or prefered. Should continue with CPU? (y/n):")
        if r != 'y':
            print("Exiting...")
            exit()

    #Train the model
    experiment = f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    yaml_file = Path(__file__).resolve().parent / dataset_folder_name / "dataset.yaml"
    save_dir = Path(__file__).resolve().parent / "training_results"
    epochs = int(input("Enter the number of epochs to train: "))
    model.train(
        data=yaml_file,
        classes = [0], # 
        epochs= epochs, 
        save_dir=save_dir, 
        project=save_dir,
        name=experiment,
        imgsz=640,
        save_period = epochs//5,
        batch = 0.7, 
        plots = True,
        amp = False, # Nan Reading if set to TRUE -> BUG: https://stackoverflow.com/questions/75178762/i-got-nan-for-all-losses-while-training-yolov8-model

        #Augmentation (https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)
        hsv_h=0.0, #(0.0 - 1.0) Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
        hsv_s=0.0, #(0.0 - 1.0) Adjusts the saturation of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
        hsv_v=0.0, #(0.0 - 1.0) Adjusts the value of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
        degrees=0, #(0.0 - 180.0) Rotates the image by a random angle within the specified range. Helps the model generalize across different orientations.
        translate = 0.0, #(0.0 - 1.0) Translates the image by a random fraction of the image size. Helps the model generalize across different positions.
        scale = 0.0, #(0.0 - 1.0) Scales the image by a random factor. Helps the model generalize across different scales.
        shear = 0.0, #(0.0 - 1.0) Shears the image by a random angle within the specified range. Helps the model generalize across different perspectives.
        perspective = 0.0, #(0.0 - 1.0) Distorts the image by a random fraction of the image size. Helps the model generalize across different perspectives.
        flipud = 0.0, #(0.0 - 1.0) Flips the image vertically. Helps the model generalize across different orientations.
        fliplr = 0.0, #(0.0 - 1.0) Flips the image horizontally. Helps the model generalize across different orientations.
        bgr = False, #Converts the image from RGB to BGR. May improve performance on some hardware.
        mosaic = 0, #(0.0 - 1.0) Adds mosaic augmentation to the image. Helps the model generalize across different positions, scales, and perspectives.
        mixup = 0, #(0.0 - 1.0) Adds mixup augmentation to the image. Helps the model generalize across different objects.
        copy_paste = 0.0, #(0.0 - 1.0) Adds copy-paste augmentation to the image. Helps the model generalize across different objects.
        erasing = 1, #(0.0 - 1.0) Adds random erasing augmentation to the image. Helps the model generalize across different objects.
        crop_fraction = 0.0, #(0.0 - 1.0) Crops the image by a random fraction of the image size. Helps the model generalize across different positions.
        
    )

def detect_with_yolo(is_verbose = False, bbox_threshold_confidence = 0.5):
    MODEL_PATH = input("Enter the path to the YOLO model file: ")
    YOLO_OBJECT = YOLO( MODEL_PATH, verbose=is_verbose)

    while True:
        frame = cv2.imread(input("Enter the path to the image file: "))
        detections = YOLO_OBJECT(frame, task = "detection", verbose= is_verbose)[0]
        
        for detection in detections: # Each detection is a single person

            boxes = detection.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])       
            box_cls_name = YOLO_OBJECT.names[box_cls_no]  
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxyn = boxes.xyxyn.cpu().numpy()[0]
            if box_conf < bbox_threshold_confidence: continue

            cv2.rectangle(frame, (int(box_xyxyn[0]*frame.shape[1]), int(box_xyxyn[1]*frame.shape[0])), (int(box_xyxyn[2]*frame.shape[1]), int(box_xyxyn[3]*frame.shape[0])), (0,0,255), 2)
            detection_dict = {'bbox_class_name': box_cls_name, "bbox_confidence": box_conf, "normalized_bbox": [box_xyxyn[0], box_xyxyn[1], box_xyxyn[2], box_xyxyn[3]], 'keypoints': None}
            print(detection_dict)
        
        cv2.imshow("Detection", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        cv2.destroyAllWindows()


        
if __name__ == "__main__":
    task = input("Do you want to train or inference? (train/inference): ")
    if task == "train":        
        recreate_dataset_folder(dataset_folder_name = DATASET_FOLDER_NAME)
        recreate_dataset_images_and_labels(dataset_folder_name = DATASET_FOLDER_NAME, number_of_defects = NUMBER_OF_DEFECTS, defects_per_image_range = DEFECTS_PER_IMAGE_RANGE, desired_image_size = DESIRED_IMAGE_SIZE, image_zoom_region = IMAGE_ZOOM_REGION)
        train_yolo_model(dataset_folder_name = DATASET_FOLDER_NAME)
    elif task == "inference":
        detect_with_yolo(is_verbose = False, bbox_threshold_confidence = 0.5)


