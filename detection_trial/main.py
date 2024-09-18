import os, shutil, datetime, yaml, uuid, random
from pathlib import Path
import cv2
import numpy as np

from ultralytics import YOLO
import torch

import defect_generators

DATASET_FOLDER_NAME = "dataset" # Never change this
NUMBER_OF_DEFECTS = 1960 
IMAGE_ZOOM_REGION = [0.3, 0.3, 0.7, 0.7]
DESIRED_IMAGE_SIZE = (640, 640)
DEFECTS_PER_IMAGE_RANGE = [1, 3+1] # last number is exclusive

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
    os.mkdir(os.path.join(dataset_folder_name, "test_defect_images"))
    os.mkdir(os.path.join(dataset_folder_name, "test_good_images"))
    
    with open(os.path.join(dataset_folder_name, "labels", "classes.txt"), "w") as f:
        f.write("defect")
        # Add more classes if needed

def recreate_dataset_images_and_labels(dataset_folder_name=None, number_of_defects:int = 100, defects_per_image_range = [1,5], desired_image_size = (640, 640), image_zoom_region = [0.0, 0.0, 1.0, 1.0]):
    BASE_IMAGES_FOLDER_PATH = Path(__file__).resolve().parent / 'base_images'
    BASE_TEST_DEFECT_IMAGES_FOLDER_PATH = Path(__file__).resolve().parent / 'base_test_defect_images'  
    DATASET_TEST_DEFECT_IMAGES_DATASET_FOLDER_PATH = Path(__file__).resolve().parent / dataset_folder_name / 'test_defect_images'  
    BASE_TEST_GOOD_IMAGES_FOLDER_PATH = Path(__file__).resolve().parent / 'base_test_good_images'
    DATASET_TEST_GOOD_IMAGES_DATASET_FOLDER_PATH = Path(__file__).resolve().parent / dataset_folder_name / 'test_good_images'

    base_image_paths = [BASE_IMAGES_FOLDER_PATH / f for f in os.listdir(BASE_IMAGES_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    base_image_paths = [str(image_path.resolve()) for image_path in base_image_paths]

    base_test_defect_image_paths = [BASE_TEST_DEFECT_IMAGES_FOLDER_PATH / f for f in os.listdir(BASE_TEST_DEFECT_IMAGES_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    base_test_defect_image_paths = [str(image_path.resolve()) for image_path in base_test_defect_image_paths]

    base_test_good_image_paths = [BASE_TEST_GOOD_IMAGES_FOLDER_PATH / f for f in os.listdir(BASE_TEST_GOOD_IMAGES_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    base_test_good_image_paths = [str(image_path.resolve()) for image_path in base_test_good_image_paths]
    
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

    # crop and resize the test defect images
    test_defect_images = [cv2.imread(image_path) for image_path in base_test_defect_image_paths]
    for image in test_defect_images:
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
        cv2.imwrite(DATASET_TEST_DEFECT_IMAGES_DATASET_FOLDER_PATH / (str(uuid.uuid4())+".jpg"), resized_image)
    
    # crop and resize the test good images
    test_good_images = [cv2.imread(image_path) for image_path in base_test_good_image_paths]
    for image in test_good_images:
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

        cv2.imwrite(DATASET_TEST_GOOD_IMAGES_DATASET_FOLDER_PATH / (str(uuid.uuid4())+".jpg"), resized_image)

    # add background images to the dataset
    for image in base_images:
        paste_image_and_label(image = image, name_prefix = "background", label = "", dataset_folder_name = dataset_folder_name)
    
    # add defect images to the dataset
    DEFECT_GENERATORS = [
        #[True, defect_generators.draw_line_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "thickness_range": [2, 13], "opacity_range": [0.5, 1.0]}],
        [True, defect_generators.draw_cocentric_lines_on_frame, {"defect_class": 0, "R_range": [0, 75], "G_range": [0, 255], "B_range": [0, 75], "thickness_range": [1, 7], "opacity_range": [0.1, 1.0], "num_lines_range": [5, 25], "random_offset_range":[-8, 8], "rotation_angle_range":[-3, 3], "bbox_range":[0.15, 0.4], "line_length_range":[0.05, 0.3]}],
        [True, defect_generators.draw_circle_on_frame, {"defect_class": 0, "R_range": [0, 75], "G_range": [0, 75], "B_range": [0, 75], "thickness_range": [2, 13], "opacity_range": [0.5, 0.9], "radius_range": [5, 30], "infill_probability": 0.9}],
        [True, defect_generators.draw_copy_paste_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "ncut_length_range": [0.05, 0.2], "ncopy_region": [0.33,0.1, 0.66, 0.9]}],
        #[False, defect_generators.draw_rectangle_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "thickness_range": [2, 13], "opacity_range": [0.5, 1.0], "infill_probability": 0.75}],
        #[False, defect_generators.add_gaussian_noise, {"defect_class": 0, "mean": 0, "var": 0.01}],
        [True, defect_generators.add_salt_and_pepper_noise, {"defect_class": 0, "amount": 0.09, "s_vs_p": 0.3, "dimension_range": [0.1, 0.3]}],        
        #[False, defect_generators.draw_ellipse_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "angle_range": [0, 360], "axes_length_range": [10, 100], "thickness_range": [3, 10], "opacity_range": [0.4, 1.0]}],
        #[False, defect_generators.add_blurred_region, {"defect_class": 0, "kernel_size_range": [5, 31]}],
        [True, defect_generators.add_scratch_on_frame, {"defect_class": 0, "num_lines": 2, "length_range": [20, 50], "thickness_range": [3, 7], "color": (random.randint(75, 255), random.randint(75, 255), random.randint(75, 255))}],
        [True, defect_generators.add_watermark_text, {"defect_class": 0, "text_length": 10, "font_scale": 2, "thickness": 3, "opacity": 0.7, "R_range":[0, 255], "G_range":[0, 255], "B_range":[0, 255]}],
        #[False, defect_generators.add_lens_flare, {"defect_class": 0, "flare_center": None, "radius_range": [50, 150], "opacity_range": [0.1, 0.5]}],
        #[False, defect_generators.add_shadow, {"defect_class": 0, "top_left_ratio": (0.2, 0.2), "bottom_right_ratio": (0.8, 0.8), "opacity": 0.5}],
        #[False, defect_generators.add_color_tint, {"defect_class": 0, "tint_color": (0, 255, 255), "opacity": 0.3}],
        #[False, defect_generators.add_crack_texture, {"defect_class": 0, "crack_image_path": "src_image/crack.png", "opacity_range": [0.7, 0.95], "size_percentage_range": [0.25, 0.4]}],
        [True, defect_generators.add_dust_particles, {"defect_class": 0, "num_particles_range": [10,50], "size_range": [1, 5]}],
        #[False, defect_generators.add_fog_effect, {"defect_class": 0, "fog_density": 0.5}],
        [True, defect_generators.draw_random_png_images, {"defect_class": 0, "random_images_folder_path": str(Path(__file__).resolve().parent / 'src_image' / 'lorem_ipsums') , "number_of_images_range": [1, 2], "opacity_range": [0.6, 0.95], "size_percentage_range": [0.05, 0.15]}],
        [True, defect_generators.draw_random_png_images_with_circular_opacity, {"defect_class": 0, "random_images_folder_path": str(Path(__file__).resolve().parent / 'src_image' / 'lorem_ipsums') , "number_of_images_range": [1, 2], "opacity_range": [0.3, 0.7], "size_percentage_range": [0.1, 0.3]}],
        [True, defect_generators.apply_wave_distortion, {"defect_class": 0, "amplitude_range": [5, 20], "wavelength_range": [10, 50], "phase_range": [0, np.pi * 2], "dimension_range": [0.1, 0.3]}],
        [True, defect_generators.apply_twirl_distortion, {"defect_class": 0, "strength_range": [0.5, 2.0], "radius_range": [50, 150], "dimension_range": [0.1, 0.3]}],
        [False, defect_generators.apply_concentric_warp_defect, {"defect_class": 0, "strength_range": [0.5, 2.0], "nradius_range": [0.05, 0.15], "dimension_range": [0.05, 0.15], "warp_type": "inward"}],
        [True, defect_generators.apply_concentric_warp_defect, {"defect_class": 0, "strength_range": [0.5, 2.0], "nradius_range": [0.05, 0.15], "dimension_range": [0.05, 0.15], "warp_type": "outward"}],
        [True, defect_generators.draw_concave_polygon_on_frame, {"defect_class": 0, "R_range": [75, 255], "G_range": [75, 255], "B_range": [75, 255], "thickness_range": [2, 13], "opacity_range": [0.5, 1.0], "num_vertices_range": [3, 25], "size_range": [10, 100], "infill_probability": 0.9}],
        [True, defect_generators.sweep_copied_section_arbitrary, {"defect_class": 0, "sweep_range":[0.1, 0.3], "section_size_range":[0.05, 0.3], "num_steps_range":[5, 20]}],   
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

def detect_with_yolo(image_folder_path = None, model_path = None, is_verbose=False, bbox_threshold_confidence=0.5):
    """
    Processes images in the given folder using the YOLO model and displays detections.

    :param image_folder_path: Path to the folder containing images.
    :param model_path: Path to the YOLO model file.
    :param is_verbose: If True, prints additional information.
    :param bbox_threshold_confidence: Confidence threshold for displaying bounding boxes.
    """
    # Initialize the YOLO model
    model_path = model_path if model_path else input("Enter the path to the model to use: ")
    YOLO_OBJECT = YOLO(model_path, verbose=is_verbose)

    while True:
        # Get list of image files in the folder
        image_folder_path = image_folder_path if image_folder_path else input("Enter the path to the folder containing images: ")
        image_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path)
                    if f.lower().endswith(image_extensions)]

        if not image_files:
            print(f"No images found in folder: {image_folder_path}")
            return

        for image_path in image_files:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to read image {image_path}. Skipping.")
                continue

            # Perform detection
            results = YOLO_OBJECT(frame, task="detection", verbose=is_verbose)

            if not results:
                print(f"No detections in image {image_path}")
                continue

            detections = results[0]

            for detection in detections:
                boxes = detection.boxes
                box_cls_no = int(boxes.cls.cpu().numpy()[0])
                box_cls_name = YOLO_OBJECT.names[box_cls_no]
                box_conf = boxes.conf.cpu().numpy()[0]
                box_xyxyn = boxes.xyxyn.cpu().numpy()[0]

                if box_conf < bbox_threshold_confidence:
                    continue

                # Draw bounding box on the frame
                x1 = int(box_xyxyn[0] * frame.shape[1])
                y1 = int(box_xyxyn[1] * frame.shape[0])
                x2 = int(box_xyxyn[2] * frame.shape[1])
                y2 = int(box_xyxyn[3] * frame.shape[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{box_cls_name}: {box_conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                detection_dict = {
                    'bbox_class_name': box_cls_name,
                    'bbox_confidence': box_conf,
                    'normalized_bbox': [box_xyxyn[0], box_xyxyn[1], box_xyxyn[2], box_xyxyn[3]],
                    'keypoints': None
                }
                print(f"Image: {image_path}, Detection: {detection_dict}")

            # Display the image with detections
            cv2.imshow("Detection", frame)
            print(f"Press any key to proceed to the next image or 'q' to quit.")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"Detection completed for all images in folder: {image_folder_path}.")
        should_continue = input("Do you want to continue with another folder? (y/n): ")
        if should_continue != 'y':
            break
        image_folder_path = None # so that the the folder path is asked again

if __name__ == "__main__":
    task = input("What do you want? (dataset/train/inference): ")
    if task == "dataset":        
        recreate_dataset_folder(dataset_folder_name = DATASET_FOLDER_NAME)
        recreate_dataset_images_and_labels(dataset_folder_name = DATASET_FOLDER_NAME, number_of_defects = NUMBER_OF_DEFECTS, defects_per_image_range = DEFECTS_PER_IMAGE_RANGE, desired_image_size = DESIRED_IMAGE_SIZE, image_zoom_region = IMAGE_ZOOM_REGION)
    elif task == "train": 
        train_yolo_model(dataset_folder_name = DATASET_FOLDER_NAME)
    elif task == "inference":
        detect_with_yolo(is_verbose = False, bbox_threshold_confidence = 0.05)


