import cv2
import xml.etree.ElementTree as ET
import os

class_name_to_id_mapping = {'signboard': 0}
class_id_to_name_mapping = {0: 'signboard'}

def denormalize(image_width, image_height, boxes):
    """
    Format boxes can both xyxy and yolo
    """
    [x1, y1, x2, y2] = boxes
    x1 = float(x1) * image_width
    y1 = float(y1) * image_height
    x2 = float(x2) * image_width
    y2 = float(y2) * image_height

    return [x1, y1, x2, y2]

def normalize(image_width, image_height, boxes):
    """
    Format boxes can both xyxy and yolo
    """
    [x1, y1, x2, y2] = boxes
    x1 = float(x1) / image_width
    y1 = float(y1) / image_height
    x2 = float(x2) / image_width
    y2 = float(y2) / image_height

    return [x1, y1, x2, y2]

def convert_box_format(boxes, current_format_box: str='yolo', output_format_box: str='none'):
    """
    Args:
        format_box: str ('yolo', 'xyxy', 'xywh')
            yolo: x_center, y_center, width, height
            xyxy: xmin, ymin, xmax, ymax
            xywh: xmin, ymin, width, height
        EX: 
            If your format box is yolo, you should set: current_format_box: 'yolo'. 
            Then, you want to convert to 'xyxy' format box, you should set: output_format_box: 'xyxy'
    Return: 
        output format box, you need
        EX: output format: [int(x_min), int(y_min), int(x_max), int(y_max)]
    """
    # Yolo format
    if current_format_box == 'yolo':
        [x_center, y_center, width, height] = boxes 
        if output_format_box == 'xyxy': 
            x_min, y_min, x_max, y_max = x_center - (width / 2), y_center - (height / 2), x_center + (width / 2), y_center + (height / 2)

            return [x_min, y_min, x_max, y_max]
        if output_format_box == 'xywh': 
            x_min, y_min, width, height = x_center - (width / 2), y_center - (height / 2), width, height

            return [x_min, y_min, width, height]
    # xyxy Format 
    if current_format_box == 'xyxy':
        [x_min, y_min, x_max, y_max] = boxes 
        if output_format_box == 'yolo': 
            x_center, y_center, width, height = (x_max + x_min) / 2, (y_max + y_min) / 2, (x_max - x_min), (y_max - y_min)

            return [x_center, y_center, width, height]
        if output_format_box == 'xywh': 
            x_min, y_min, width, height = x_min, y_min, (x_max - x_min), (y_max - y_min)

            return [x_min, y_min, width, height]
    
    if current_format_box == 'xywh':
        [x_min, y_min, width, height] = boxes 
        if output_format_box == 'xyxy': 
            x_min, y_min, x_max, y_max = x_min, y_min, x_min + width, y_min + height

            return [x_min, y_min, x_max, y_max]
        if output_format_box == 'yolo': 
            x_center, y_center, width, height = x_min + (width / 2), y_min + (height / 2), width, height

            return [x_center, y_center, width, height]

def draw_bounding_box(image, boxes, class_id, format_box: str='yolo', mode: str='none'):
    """
    Args: 
        format_box: 'yolo' or 'xywh'. If format_box is yolo, we need to convert to xyxy to plot annotated image.
        mode: 'normalize', 'denormalize', 'none'. 
            if mode == 'normalize': yolo boxes format need to normalize with range [0, 1]
            if mode == 'denormalize': yolo boxes format need to denormalize to plot annotated image
            if mode == 'none': only convert yolo to xyxy
        
        Note: format_box is xyxy, you don't need to call convert_box_format function.
    Return: 
        Annotated image
    """
    image_height, image_width = image.shape[:2]
    # Scale boxes 
    if mode == 'normalize': boxes = normalize(image_width, image_height, boxes)
    if mode == 'denormalize': boxes = denormalize(image_width, image_height, boxes)
    # Convert format boxes 
    if format_box == 'yolo': 
        boxes = convert_box_format(boxes, current_format_box=format_box,
                                                          output_format_box='xyxy')
    if format_box == 'xywh':
        boxes = convert_box_format(boxes, current_format_box=format_box, 
                                                          output_format_box='xyxy')
        
    [x_min, y_min, x_max, y_max] = [int(box) for box in boxes]
    x_text_center = x_min + (x_max - x_min) // 4
    class_name = class_id_to_name_mapping[int(class_id)]
    
    cv2.rectangle(img=image, pt1=(x_min, y_min), pt2=(x_max, y_max), 
                  color=(0, 255, 0), thickness=2)
    cv2.putText(img=image, text=class_name, org=(x_text_center, y_min), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(255, 265, 0), thickness=2, lineType=cv2.LINE_AA)
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

def prepare_yolo_format(info_dict, folder=None):
    # image_width, image_height = info_dict['image_size'][:2]
    image_width, image_height = 1000, 600
    for bbox in info_dict['bboxes']:
        try:
            class_id = class_name_to_id_mapping[bbox['class']]
        except KeyError:
            print(f"Invalid Class. Must be one from {class_name_to_id_mapping.keys()}")

        # Get boxes from info_dict
        boxes = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
        # Convert xyxy format to x_center, y_center, width, height
        boxes = convert_box_format(boxes, current_format_box='xyxy',
                                                               output_format_box='yolo')
        # Normalize boxes range [0, 1]
        n_x_center, n_y_center, n_width, n_height = normalize(image_width, image_height, boxes)

        file_path = os.path.join(folder, f"{info_dict['file_name'].split('.')[0]}.txt")
        # Write file txt
        write_txt_yolo(file_path, class_id, n_x_center, n_y_center, n_width, n_height)