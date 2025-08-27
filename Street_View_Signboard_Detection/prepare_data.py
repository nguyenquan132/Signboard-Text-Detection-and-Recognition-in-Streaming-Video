import os
from utils import extract_xml, convert_box_yolo_format
from tqdm import tqdm
import shutil

FOLDER_TRAIN_IMAGE = 'PIL_RGB_ Train-Validation'
FOLDER_TRAIN_LABEL = 'Labelled Train-Val XML'
FOLDER_VAL_IMAGE = 'Test Data'
FOLDER_VAL_LABEL = 'Labeled Test data'

list_train_image = sorted(os.listdir(FOLDER_TRAIN_IMAGE), key=lambda x: int(x.split('.')[0]))
list_train_label = sorted(os.listdir(FOLDER_TRAIN_LABEL), key=lambda x: int(x.split('.')[0]))
list_val_image = sorted(os.listdir(FOLDER_VAL_IMAGE), key=lambda x: int(x.split('.')[0]))
list_val_label = sorted(os.listdir(FOLDER_VAL_LABEL), key=lambda x: int(x.split('.')[0]))

# Move image train and val to images 
print('----------------Move Train Image--------------------')
if len(list_train_image) != 0:
    for train_image in tqdm(list_train_image):
        src_train_path = os.path.join(FOLDER_TRAIN_IMAGE, train_image)
        dst_train_path = os.path.join('images/train', train_image)
        shutil.move(src=src_train_path, dst=dst_train_path)
else: 
    print('Train Image is empty!')

print('-----------------Move Val Image----------------------')
if len(list_val_image) != 0:
    for val_image in tqdm(list_val_image):
        src_val_path = os.path.join(FOLDER_VAL_IMAGE, val_image)
        dst_val_path = os.path.join('images/val', val_image)
        shutil.move(src=src_val_path, dst=dst_val_path)
else: 
    print('Val Image is empty!')

# Prepare label data train and val 
print('-------------Annotation Train Label------------------')
for train_label in tqdm(list_train_label):
    train_label_path = os.path.join(FOLDER_TRAIN_LABEL, train_label)
    train_info_dict = extract_xml(train_label_path)
    convert_box_yolo_format(train_info_dict, folder='labels/train')

print('-------------Annotation Val Label-------------------')
for val_label in tqdm(list_val_label):
    val_label_path = os.path.join(FOLDER_VAL_LABEL, val_label)
    val_info_dict = extract_xml(val_label_path)
    convert_box_yolo_format(val_info_dict, folder='labels/val')