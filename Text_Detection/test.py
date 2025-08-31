import sys
sys.path.append('src/TextSnake.pytorch')
from network.textnet import TextNet
from util.detection import TextDetector
from util.config import config as cfg, update_config, print_config
from util.augmentation import BaseTransform
import cv2
import torch
import os
from pathlib import Path
import numpy as np

def map_contour_to_orginal_image(image, contours, input_size):
    # Calculate resize ratio
    original_height, original_width = image.shape[:2]
    ratio = [original_width / input_size, original_height / input_size]
    # Multiply the contour coordinates by the resize ratio
    original_contours = [(contour * ratio).astype(np.int32) for contour in contours]

    return original_contours

# Load textsnake model architecture
textsnake_model = TextNet(backbone='vgg', is_training=False)
textsnake_model_path = 'textsnake_vgg_180.pth'
# Load weights
textsnake_model.load_model(model_path=textsnake_model_path, device='cpu')

image_path = Path('test_images/img546.png')
output_path = os.path.join(image_path.parent, f'output_{image_path.name}')
image = cv2.imread(image_path)
print(image.shape)
# Transform image
input_size = 512
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)
base_transform = BaseTransform(size=input_size, mean=means, std=stds)
transformed_image = base_transform(image)

# convert numpy image to tensor image
tensor_transformed_image = torch.from_numpy(transformed_image[0])
# Permute dimension
tensor_transformed_image = torch.permute(tensor_transformed_image, dims=(2, 0, 1))
# Add batch dimension
batch_tensor_transformed_image = tensor_transformed_image.unsqueeze(dim=0)
print(batch_tensor_transformed_image.shape)

# tr_thresh: 0.6, tcl_threshold: 0.4
detector = TextDetector(textsnake_model, tr_thresh=cfg.tr_thresh, tcl_thresh=cfg.tcl_thresh)
# Get detection result
contours, output = detector.detect(batch_tensor_transformed_image)

print(output.keys())

original_contours = map_contour_to_orginal_image(image, contours, input_size)
cv2.polylines(image, pts=original_contours, isClosed=True, color=(0, 0, 255), thickness=3)

cv2.imshow(image_path.name, image)
cv2.imwrite(output_path, image)
cv2.waitKey()
cv2.destroyAllWindows()