import cv2
import xml.etree.ElementTree as ET
import os

class_name_to_id_mapping = {'signboard': 0}

def draw_bounding_box(image, boxes):
    [x_min, y_min, x_max, y_max] = boxes

    cv2.rectangle(img=image, pt1=(x_min, y_min), pt2=(x_max, y_max), 
                  color=(0, 255, 0), thickness=2)
    return image

def write_txt_yolo(file_path, class_id, x_center, y_center, width, height):
    with open(file_path, 'a') as f: 
        line = f"{class_id} {x_center} {y_center} {width} {height}"
        f.write(line + '\n')

def extract_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Get filename
    file_name = root.find('filename').text
    info_dict['file_name'] = file_name
    # Get image size, ex: [224, 224, 3]
    image_size = [int(size.text) for size in root.find('size')]
    # Convert list to tuple, ex: [224, 224, 3] -> (224, 224, 3)
    image_size = tuple(image_size)
    info_dict['image_size'] = image_size
    # Loop object to extract object's boxes 
    for object in root.iter('object'):
        bbox = {}
        class_name = object.find('name').text
        bbox['class'] = class_name
        for box in object.find('bndbox'):
            bbox[box.tag] = int(box.text)
        info_dict['bboxes'].append(bbox)

    return info_dict

def convert_box_yolo_format(info_dict, folder=None):
    # image_width, image_height = info_dict['image_size'][:2]
    image_width, image_height = 1000, 600
    for bbox in info_dict['bboxes']:
        try:
            class_id = class_name_to_id_mapping[bbox['class']]
        except KeyError:
            print(f"Invalid Class. Must be one from {class_name_to_id_mapping.keys()}")

        # Get boxes from info_dict
        x_min, y_min, x_max, y_max = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        # Convert xyxy format to x_center, y_center, width, height
        x_center = (x_max + x_min) // 2
        y_center = (y_max + y_min) // 2
        width = (x_max - x_min)
        height = (y_max - y_min)
        # Normalize boxes range [0, 1]
        n_x_center = x_center / image_width
        n_y_center = y_center / image_height
        n_width = width / image_width
        n_height = height / image_height

        file_path = os.path.join(folder, f"{info_dict['file_name'].split('.')[0]}.txt")
        # Write file txt
        write_txt_yolo(file_path, class_id, n_x_center, n_y_center, n_width, n_height)