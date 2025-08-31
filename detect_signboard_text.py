import torch
import cv2
import os
from pathlib import Path
import numpy as np
import random
from utils_custom import draw_bounding_box
import sys
sys.path.append('Text_Detection/src/TextSnake.pytorch')
from network.textnet import TextNet
from util.detection import TextDetector
from util.config import config as cfg, update_config, print_config
from util.augmentation import BaseTransform
random.seed(42)

# Detect Signboard
# Load model and weights 
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path='Street_View_Signboard_Detection/Signboard_Detection/baseline_yolov5s_30epoch/weights/best.pt')
model.eval()

# Load textsnake model architecture
textsnake_model = TextNet(backbone='vgg', is_training=False)
textsnake_model_path = 'Text_Detection/textsnake_vgg_180.pth'
# Load weights
textsnake_model.load_model(model_path=textsnake_model_path, device='cpu')

def map_contour_to_orginal_image(image, contours, input_size):
    # Calculate resize ratio
    original_height, original_width = image.shape[:2]
    ratio = [original_width / input_size, original_height / input_size]
    # Multiply the contour coordinates by the resize ratio
    original_contours = [(contour * ratio).astype(np.int32) for contour in contours]

    return original_contours

# Detect Signboard
def detect_signboard(test_image, test_image_path: str=None):
    """
    Args: 
        test_image: (H, W, 3)
    Return: 
        list information of each object
    """
    results = model(test_image, size=640)
    info_object = results.xyxy[0].cpu().numpy()

    # Create mask to get pixel area inside rectangle
    mask = np.zeros((test_image.shape[:2]), dtype=np.uint8)
    # Get object information
    if len(info_object) != 0:
        list_info_image = {}
        for i, object in enumerate(info_object):
            class_id = object[5]
            boxes = object[:4]
            confident = object[4]

            xmin, ymin, xmax, ymax = [int(box) for box in boxes]
            cv2.rectangle(mask, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(255, 255, 255), thickness=-1)
            masked_image = cv2.bitwise_and(test_image, test_image, mask=mask)
            cropped_image = test_image[ymin:ymax, xmin:xmax].copy()

            list_info_image[f'object_{i + 1}'] = {
                'bbox_xyxy': [xmin, ymin, xmax, ymax],
                'class_id': class_id,
                'masked_image': masked_image,
                'cropped_image': cropped_image
            }

    # Save masked image
    # output_masked_image_path = os.path.join(Path(test_image_path).parent, f'output_masked_image_{Path(test_image_path).name}')
    # cv2.imwrite(output_masked_image_path, masked_image)

    return list_info_image

def detect_text(original_image, image_detection, test_image_path: str=None, xmin: int=None, ymin: int=None, methods: str='masked'):
    """
    Args: 
        original_image: (H, W, 3)
        image_detection: (new_H, new_W, 3), (masked image or cropped image), new image after detect signboard
        xmin, ymin: xmin, ymin coordinate of each object to map image_detection contours to original image, if use methods "cropped" 
        methods: ('masked', 'cropped'), default: 'masked'
    Return: 
        Annotated orginal_image and image_detection
    """
    image_detection_copy = image_detection.copy()
    # Transform image
    input_size = 512
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    base_transform = BaseTransform(size=input_size, mean=means, std=stds)
    transformed_image = base_transform(image_detection_copy)
    print(transformed_image[0].shape)

    # convert numpy image to tensor image
    tensor_transformed_image = torch.from_numpy(transformed_image[0])
    # Permute dimension
    tensor_transformed_image = torch.permute(tensor_transformed_image, dims=(2, 0, 1))
    # Add batch dimension
    batch_tensor_transformed_image = tensor_transformed_image.unsqueeze(dim=0)

    # tr_thresh: 0.6, tcl_threshold: 0.4
    detector = TextDetector(textsnake_model, tr_thresh=cfg.tr_thresh, tcl_thresh=cfg.tcl_thresh)
    # Get detection result
    contours, output = detector.detect(batch_tensor_transformed_image)

    masked_contours = map_contour_to_orginal_image(image_detection_copy, contours, input_size)
    cv2.polylines(image_detection_copy, pts=masked_contours, isClosed=True, color=(0, 0, 255), thickness=2)
    if methods == 'masked': 
        original_contours = map_contour_to_orginal_image(original_image, contours, input_size)
        cv2.polylines(original_image, pts=original_contours, isClosed=True, color=(0, 0, 255), thickness=2)
    if methods == 'cropped': 
        if xmin is None or ymin is None: print(f'xmin or ymin is not None !!!')
        else: 
            original_contours = [masked_cont + [xmin, ymin] for masked_cont in masked_contours]
            cv2.polylines(original_image, pts=original_contours, isClosed=True, color=(0, 0, 255), thickness=2)

    # Save masked image
    # output_text_image_path = os.path.join(Path(test_image_path).parent, f'output_text_image_{Path(test_image_path).name}')
    # cv2.imwrite(output_text_image_path, original_image)

    return original_image, image_detection_copy

if __name__ == '__main__':
    test_image_path = 'Text_Detection/test_images/img546.jpg'
    test_image = cv2.imread(test_image_path)

    cv2.imshow('Original image', test_image)
    # Detect Signboard
    list_info_image = detect_signboard(test_image, test_image_path)
    test_image_copy = test_image.copy()
    # Detect text each object 
    for object in list_info_image:
        boxes = list_info_image[object]['bbox_xyxy']
        xmin, ymin = boxes[:2]
        class_id = list_info_image[object]['class_id']
        draw_bounding_box(test_image_copy, boxes, class_id, format_box='xyxy')
        masked_image = list_info_image[object]['masked_image']
        cropped_image = list_info_image[object]['cropped_image']

        test_image, image_detection_copy = detect_text(test_image, cropped_image, xmin=xmin, ymin=ymin, methods='cropped')

        # cv2.imshow(f'Masked image {object}', masked_image)
        # cv2.imshow(f'Detected text maksed image {object}', image_detection_copy)
        cv2.imshow(f'Cropped image {object}', cropped_image)
        cv2.imshow(f'Detected text cropped image {object}', image_detection_copy)

    cv2.imshow('Detected bboxes original image', test_image_copy)
    cv2.imshow('Detected text original_image', test_image)

    cv2.waitKey()
    cv2.destroyAllWindows()
