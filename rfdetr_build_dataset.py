import argparse
import enum
from fileinput import filename
import os
import shutil
import json

def generate_motion_mask(img1_path, img2_path, output_path):
    """Generate motion mask between two consecutive frames"""
    import cv2
    
    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Warning: Could not read images {img1_path} or {img2_path}")
        return False
    
    # Resize images if they don't match
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Calculate absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Apply Gaussian blur to reduce noise
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Threshold to create binary mask
    _, motion_mask = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    # Convert back to 3-channel for consistency
    motion_mask_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    
    # Save the motion mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, motion_mask_color)
    
    return True

def create_yolomg_annotations(annotation_file_path, image_name_dict, seq_name, split_folder, base_output_dir):
    """Create YOLO format annotations and motion masks for YOLOMG"""
    if not os.path.exists(annotation_file_path):
        print(f"Warning: Annotation file not found: {annotation_file_path}")
        return
    
    # Parse the annotation file
    frame_annotations = {}
    with open(annotation_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 8:
                continue
            
            frame_id = int(parts[0])
            object_id = parts[1]
            bbox_left = float(parts[2])
            bbox_top = float(parts[3])
            bbox_width = float(parts[4])
            bbox_height = float(parts[5])
            confidence = float(parts[6])
            object_category = parts[7]
            
            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []
            
            annotation = {
                "category_id": int(object_category) - 1,  # Convert to 0-based indexing
                "bbox": [float(bbox_left), float(bbox_top), float(bbox_width), float(bbox_height)]
            }
            frame_annotations[frame_id].append(annotation)
    
    # Create YOLO annotations and motion masks for each frame
    sorted_frame_ids = sorted(frame_annotations.keys())
    
    for i, frame_id in enumerate(sorted_frame_ids):
        if frame_id in image_name_dict:
            img_entry = image_name_dict[frame_id]
            annotations = frame_annotations[frame_id]
            
            # Create YOLO format annotation
            yolo_content = create_yolo_annotation(img_entry, annotations, img_entry.width, img_entry.height)
            
            # Determine annotation file path
            label_filename = img_entry.new_file_name.replace('.jpg', '.txt').replace('.JPEG', '.txt')
            labels_dir = os.path.join(base_output_dir, "labels")
            os.makedirs(labels_dir, exist_ok=True)
            label_path = os.path.join(labels_dir, label_filename)
            
            # Write YOLO annotation file
            with open(label_path, 'w') as f:
                f.write(yolo_content)
            
            # Generate motion mask (if not the first frame)
            if i > 0:
                prev_frame_id = sorted_frame_ids[i-1]
                if prev_frame_id in image_name_dict:
                    prev_img_entry = image_name_dict[prev_frame_id]
                    
                    # Paths to current and previous images
                    curr_img_path = os.path.join(base_output_dir, "images", img_entry.new_file_name)
                    prev_img_path = os.path.join(base_output_dir, "images", prev_img_entry.new_file_name)
                    
                    # Motion mask output path
                    mask_filename = img_entry.new_file_name
                    mask_dir = os.path.join(base_output_dir, "masks")
                    mask_path = os.path.join(mask_dir, mask_filename)
                    
                    # Generate motion mask
                    if os.path.exists(curr_img_path) and os.path.exists(prev_img_path):
                        generate_motion_mask(prev_img_path, curr_img_path, mask_path)
            else:
                # For the first frame, create an empty mask
                img_path = os.path.join(base_output_dir, "images", img_entry.new_file_name)
                if os.path.exists(img_path):
                    import cv2
                    img = cv2.imread(img_path)
                    if img is not None:
                        empty_mask = np.zeros_like(img)
                        mask_filename = img_entry.new_file_name
                        mask_dir = os.path.join(base_output_dir, "masks")
                        os.makedirs(mask_dir, exist_ok=True)
                        mask_path = os.path.join(mask_dir, mask_filename)
                        cv2.imwrite(mask_path, empty_mask)

import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from dataclasses import dataclass

datasetTypes = ['visdrone', 'uavdt']
visdrone_split_names = ['VisDrone2019-VID-train','VisDrone2019-VID-val','VisDrone2019-VID-test-dev']
uavdt_split_names = ['UAVDT-train', 'UAVDT-val', 'UAVDT-test']

batch_size = 100

visdrone_categories = [
    {"id": 0, "name": "pedestrian","supercategory": "person"},
    {"id": 1, "name": "people","supercategory": "person"},
    {"id": 2, "name": "bicycle","supercategory": "vehicle"},
    {"id": 3, "name": "car","supercategory": "vehicle"},
    {"id": 4, "name": "van","supercategory": "vehicle"},
    {"id": 5, "name": "truck","supercategory": "vehicle"},
    {"id": 6, "name": "tricycle","supercategory": "vehicle"},
    {"id": 7, "name": "awning-tricycle","supercategory": "vehicle"},
    {"id": 8, "name": "bus","supercategory": "vehicle"},
    {"id": 9, "name": "motor","supercategory": "vehicle"},
    {"id": 10, "name": "others","supercategory": "unknown"},
]

uavdt_categories = [
    {"id": 0, "name": "car","supercategory": "vehicle"},
    {"id": 1, "name": "truck","supercategory": "vehicle"},
    {"id": 2, "name": "bus","supercategory": "vehicle"},
]

# VID classes for YOLOV (30 classes)
# VID to ImageNet code mapping (for YOLOV compatibility)
vid_to_imagenet_codes = {
    "airplane": "n02691156", "antelope": "n02419796", "bear": "n02131653", "bicycle": "n02834778",
    "bird": "n01503061", "bus": "n02924116", "car": "n02958343", "cattle": "n02402425",
    "dog": "n02084071", "domestic_cat": "n02121808", "elephant": "n02503517", "fox": "n02118333",
    "giant_panda": "n02510455", "hamster": "n02342885", "horse": "n02374451", "lion": "n02129165",
    "lizard": "n01674464", "monkey": "n02484322", "motorcycle": "n03790512", "rabbit": "n02324045",
    "red_panda": "n02509815", "sheep": "n02411705", "snake": "n01726692", "squirrel": "n02355227",
    "tiger": "n02129604", "train": "n04468005", "turtle": "n01662784", "watercraft": "n04530566",
    "whale": "n02062744", "zebra": "n02391049"
}

vid_class_names = ["airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", "cattle", "dog", "domestic_cat", "elephant", "fox", "giant_panda", "hamster", "horse", "lion", "lizard", "monkey", "motorcycle", "rabbit", "red_panda", "sheep", "snake", "squirrel", "tiger", "train", "turtle", "watercraft", "whale", "zebra"]

def create_yolo_annotation(img_entry, annotations, img_width, img_height):
    """Create YOLO format annotation (.txt file) for YOLOMG"""
    lines = []
    
    # UAVDT class mapping for YOLOMG (only 3 classes)
    # UAVDT classes: 1=car, 2=truck, 3=bus, 4=van/other -> map to 0=car, 1=truck, 2=bus
    uavdt_class_mapping = {
        0: 0,  # car -> car (class 0)
        1: 1,  # truck -> truck (class 1) 
        2: 2,  # bus -> bus (class 2)
        3: 1   # van/other -> truck (class 1) - merge with trucks
    }
    
    for ann in annotations:
        # For UAVDT+YOLOMG, use direct class mapping without VID conversion
        if DATASETTYPE == DatasetType.UAVDT:
            category_id = ann["category_id"]  # This is already 0-based from the parsing
            if category_id in uavdt_class_mapping:
                class_id = uavdt_class_mapping[category_id]
            else:
                print(f"Warning: Unknown UAVDT class ID {category_id}, skipping annotation")
                continue
        else:
            # For other datasets, use the VID mapping approach
            category_name = get_category_name(ann["category_id"])
            vid_class_name = map_to_vid_class(category_name)
            
            uavdt_to_yolo_mapping = {
                "car": 0,
                "truck": 1, 
                "bus": 2
            }
            
            if vid_class_name in uavdt_to_yolo_mapping:
                class_id = uavdt_to_yolo_mapping[vid_class_name]
            else:
                print(f"Warning: Unknown class {vid_class_name}, skipping annotation")
                continue  # Skip unknown classes
        
        # Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized)
        bbox = ann["bbox"]
        x_left, y_top, bbox_width, bbox_height = bbox
        
        # Calculate center coordinates
        x_center = x_left + bbox_width / 2.0
        y_center = y_top + bbox_height / 2.0
        
        # Normalize to 0-1 range
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = bbox_width / img_width
        height_norm = bbox_height / img_height
        
        # Ensure values are within [0, 1] range
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        # YOLO format: class_id x_center y_center width height
        line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        lines.append(line)
    
    return "\n".join(lines)

# Mapping from dataset categories to VID classes
visdrone_to_vid_mapping = {
    "pedestrian": "rabbit",  # map pedestrian to closest VID class
    "people": "rabbit",      # map people to closest VID class  
    "bicycle": "bicycle",
    "car": "car",
    "van": "car",
    "truck": "train",
    "tricycle": "bicycle",
    "awning-tricycle": "bicycle", 
    "bus": "bus",
    "motor": "motorcycle",
    "others": "rabbit"
}

uavdt_to_vid_mapping = {
    "car": "car",
    "truck": "train", 
    "bus": "bus"
}

#{"file_name": "VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00025014/000004.JPEG",
#  "height": 720, "width": 1280, "id": 56595, "frame_id": 4, "video_id": 200, "is_vid_train_frame": true}

#{"id": 44614, "video_id": -1, "image_id": 35357, "category_id": 22, "instance_id": -1,
#  "bbox": [61, 50, 237, 243], "area": 57591, "iscrowd": false, "occluded": -1, "generated": -1}

@dataclass
class ImageEntry:
    original_file_name: str
    original_file_path: str
    new_file_name: str
    height: int
    width: int
    id: int
    video_name: str

class DatasetType(enum.Enum):
    VISDRONE = "visdrone"
    UAVDT = "uavdt"

class ModelType(enum.Enum):
    RFDETR = "rfdetr"
    TRANSVOD = "transvod"
    YOLOV = "yolov"
    YOLOMG = "yolomg"

DATASETTYPE = DatasetType.VISDRONE
MODELTYPE = ModelType.RFDETR

# Global variables for YOLOV sequence tracking
yolov_train_sequences = {}
yolov_val_sequences = {}


def create_xml_annotation(img_entry, annotations, width, height):
    """Create XML annotation file for YOLOV in VID format"""
    root = ET.Element("annotation")
    
    # Add folder
    folder = ET.SubElement(root, "folder")
    folder.text = "VID"
    
    # Add filename
    filename_elem = ET.SubElement(root, "filename")
    filename_elem.text = img_entry.new_file_name
    
    # Add size
    size = ET.SubElement(root, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = "3"
    
    # Add segmented
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    
    # Add objects
    for ann in annotations:
        obj = ET.SubElement(root, "object")
        
        name = ET.SubElement(obj, "name")
        # Map category to VID class name
        category_name = get_category_name(ann["category_id"])
        vid_class_name = map_to_vid_class(category_name)
        
        # For YOLOV, use ImageNet codes instead of human-readable names
        if MODELTYPE == ModelType.YOLOV:
            if vid_class_name in vid_to_imagenet_codes:
                name.text = vid_to_imagenet_codes[vid_class_name]
            else:
                name.text = vid_class_name  # Fallback to original name
        else:
            name.text = vid_class_name
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        # Bounding box
        bbox = ann["bbox"]
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(bbox[0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(bbox[1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(bbox[0] + bbox[2]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(bbox[1] + bbox[3]))
    
    # Format XML with minidom for pretty printing
    rough_string = ET.tostring(root, 'unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def get_category_name(category_id):
    """Get category name from category ID"""
    if DATASETTYPE == DatasetType.VISDRONE:
        for cat in visdrone_categories:
            if cat["id"] == category_id:
                return cat["name"]
    elif DATASETTYPE == DatasetType.UAVDT:
        for cat in uavdt_categories:
            if cat["id"] == category_id:
                return cat["name"]
    return "unknown"


def map_to_vid_class(category_name):
    """Map dataset category to VID class name"""
    if DATASETTYPE == DatasetType.VISDRONE:
        return visdrone_to_vid_mapping.get(category_name, "rabbit")  # Default to rabbit
    elif DATASETTYPE == DatasetType.UAVDT:
        return uavdt_to_vid_mapping.get(category_name, "car")  # Default to car
    return "rabbit"


def map_to_vid_class(category_name):
    """Map dataset category to VID class name"""
    if DATASETTYPE == DatasetType.VISDRONE:
        return visdrone_to_vid_mapping.get(category_name, "rabbit")
    elif DATASETTYPE == DatasetType.UAVDT:
        return uavdt_to_vid_mapping.get(category_name, "car")
    return "rabbit"


def generate_sequence_files(output_dir, train_sequences, val_sequences):
    """Generate train_seq.npy and val_seq.npy files for YOLOV"""
    # Create sequence lists (lists of frame paths per video)
    train_seq_list = []
    val_seq_list = []
    
    # Process training sequences
    for seq_name, frames in train_sequences.items():
        frame_paths = []
        for frame in sorted(frames):
            # Path relative to dataset root
            rel_path = f"Data/VID/train/{seq_name}/{frame}"
            frame_paths.append(rel_path)
        if frame_paths:  # Only add non-empty sequences
            train_seq_list.append(frame_paths)
    
    # Process validation sequences
    for seq_name, frames in val_sequences.items():
        frame_paths = []
        for frame in sorted(frames):
            # Path relative to dataset root
            rel_path = f"Data/VID/val/{seq_name}/{frame}"
            frame_paths.append(rel_path)
        if frame_paths:  # Only add non-empty sequences
            val_seq_list.append(frame_paths)
    
    # Save as numpy arrays
    train_seq_array = np.array(train_seq_list, dtype=object)
    val_seq_array = np.array(val_seq_list, dtype=object)
    
    np.save(os.path.join(output_dir, "train_seq.npy"), train_seq_array)
    np.save(os.path.join(output_dir, "val_seq.npy"), val_seq_array)
    
    print(f"Generated train_seq.npy with {len(train_seq_list)} sequences")
    print(f"Generated val_seq.npy with {len(val_seq_list)} sequences")


def create_yolov_xml_annotations(annotation_file_path, image_name_dict, seq_name, split_folder, base_output_dir):
    """Create XML annotation files for YOLOV format"""
    try:
        # annotation_file_path is already the correct path (including _gt_whole if needed)
        with open(annotation_file_path, 'r') as f:
            lines = f.readlines()
            
        # Parse annotations and group by frame
        frame_annotations = {}
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
                
            if DATASETTYPE == DatasetType.UAVDT:
                frame_id, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category = parts[:8]
            else:  # VisDrone
                frame_id, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion = parts[:10]
                
            frame_id = int(frame_id)
            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []
                
            # Convert to proper format
            annotation = {
                "category_id": int(object_category) - 1,  # Convert to 0-based indexing
                "bbox": [float(bbox_left), float(bbox_top), float(bbox_width), float(bbox_height)]
            }
            frame_annotations[frame_id].append(annotation)
        
        # Create XML files for each frame
        for frame_id, annotations in frame_annotations.items():
            if frame_id in image_name_dict:
                img_entry = image_name_dict[frame_id]
                
                # Create XML content
                xml_content = create_xml_annotation(img_entry, annotations, img_entry.width, img_entry.height)
                
                # Determine XML file path
                xml_filename = img_entry.new_file_name.replace('.jpg', '.xml').replace('.JPEG', '.xml')
                xml_dir = os.path.join(base_output_dir, "Annotations", "VID", split_folder, seq_name)
                os.makedirs(xml_dir, exist_ok=True)
                xml_path = os.path.join(xml_dir, xml_filename)
                
                # Write XML file
                with open(xml_path, 'w') as f:
                    f.write(xml_content)
                    
    except Exception as e:
        print(f"Error creating XML annotations for sequence {seq_name}: {e}")


def insertNewAnnotationEntry(annotation_data, src_ann, image_name_dict, annotation_id):
    if src_ann.endswith('.txt'):
        try:
            isUavdt = False
            ann_path = src_ann
            if(DATASETTYPE == DatasetType.UAVDT):
                base, ext = os.path.splitext(src_ann)
                ann_path = base + '_gt_whole' + ext
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if DATASETTYPE == DatasetType.VISDRONE:
                        if len(parts) != 10:
                            print("Error: Annotation line does not contain 10 parts.")
                            continue
                        frame_id = int(parts[0])
                        track_id = int(parts[1])
                        x_top_left = float(parts[2])
                        y_top_left = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        score = float(parts[6])
                        category_id = int(parts[7])
                        truncation = int(parts[8])
                        occlusion = int(parts[9])
                        if(category_id < 1 or category_id > 11):
                            continue
                        category_id -= 1  # Convert from 1-based to 0-based indexing
                    elif DATASETTYPE == DatasetType.UAVDT:
                        if len(parts) != 9:
                            print("Error: UAVDT Annotation line does not contain 10 parts.")
                            continue
                        frame_id = int(parts[0])  # frame_index
                        track_id = int(parts[1])  # target_id
                        x_top_left = float(parts[2])  # bbox_left
                        y_top_left = float(parts[3])  # bbox_top
                        width = float(parts[4])  # bbox_width
                        height = float(parts[5])  # bbox_height
                        category_id = int(parts[6])  # object_category
                        occlusion = int(parts[7])  # occlusion
                        truncation = int(parts[8])  # truncation
                        if category_id < 1 or category_id > 3:
                            continue
                        category_id -= 1  # Convert from 1-based to 0-based indexing
                    else:
                        print(f"Error: Unknown DATASETTYPE '{DATASETTYPE}'.")
                        return annotation_id
                    
                    image_entry = image_name_dict.get(frame_id)
                    if image_entry is None:
                        print(f"Warning: No image entry found for frame ID {frame_id} in annotation file '{src_ann}'.")
                        continue
                    
                    # Create annotation entry with model-specific fields
                    if MODELTYPE == ModelType.RFDETR:
                        annotation_entry = {
                            "id": annotation_id,
                            "image_id": image_entry.id,
                            "category_id": category_id,
                            "bbox": [x_top_left, y_top_left, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        }
                    elif MODELTYPE == ModelType.TRANSVOD:
                        # Extract video ID from image entry
                        video_name_base = image_entry.video_name.replace('.txt', '')
                        if 'uav' in video_name_base.lower():
                            video_id = int(video_name_base.replace('uav', '').replace('_', ''))
                        else:
                            import re
                            numbers = re.findall(r'\d+', video_name_base)
                            video_id = int(numbers[-1]) if numbers else 0
                        
                        annotation_entry = {
                            "id": annotation_id,
                            "image_id": image_entry.id,
                            "category_id": category_id,
                            "bbox": [x_top_left, y_top_left, width, height],
                            "area": width * height,
                            "iscrowd": False,         # Boolean for TransVOD
                            "video_id": video_id,
                            "instance_id": track_id,  # Use track_id as instance_id for object tracking
                            "occluded": -1,           # Default values as in ImageNet VID
                            "generated": -1
                        }
                    
                    annotation_id += 1
                    annotation_data.append(annotation_entry)
                return annotation_id

        except Exception as e:
            print(f"Error reading annotation file '{src_ann}': {e}")

def insertPictureInAnnotationFile(batch_images, imgEntry, split_folder=None):
    # YOLOV and YOLOMG don't use batch_images - they create their own annotation formats
    if MODELTYPE == ModelType.YOLOV or MODELTYPE == ModelType.YOLOMG:
        return
        
    if MODELTYPE == ModelType.RFDETR:
        img_dict = {
            "file_name": imgEntry.new_file_name,
            "height": imgEntry.height,
            "width": imgEntry.width,
            "id": imgEntry.id,
            "video_name": imgEntry.video_name
        }
    elif MODELTYPE == ModelType.TRANSVOD:
        # Extract video number from video_name (e.g., "uav0000013.txt" -> 13)
        video_name_base = imgEntry.video_name.replace('.txt', '')
        if 'uav' in video_name_base.lower():
            video_id = int(video_name_base.replace('uav', '').replace('_', ''))
        else:
            # For other formats, try to extract number
            import re
            numbers = re.findall(r'\d+', video_name_base)
            video_id = int(numbers[-1]) if numbers else 0
        
        frame_id = int(imgEntry.original_file_name.split('.')[0])
        is_train_frame = split_folder == 'train' if split_folder else True
        
        img_dict = {
            "file_name": imgEntry.new_file_name,  # Just the filename, since images are in Data/VID/train etc.
            "height": imgEntry.height,
            "width": imgEntry.width,
            "id": imgEntry.id,
            "frame_id": frame_id,
            "video_id": video_id,
            "is_vid_train_frame": is_train_frame
        }
    
    batch_images.append(img_dict)
    return



def createDatasetSplitDirectory(folder_root, folder_name, split_name, dataset_type):
    print(f"\nProcessing split: {split_name}")
    if 'train' in split_name.lower():
        splitFolder = 'train'
    elif 'val' in split_name.lower():
        if MODELTYPE == ModelType.RFDETR:
            splitFolder = 'valid'
        elif MODELTYPE == ModelType.TRANSVOD:
            splitFolder = 'val'
        elif MODELTYPE == ModelType.YOLOV:
            splitFolder = 'val'
        elif MODELTYPE == ModelType.YOLOMG:
            splitFolder = 'val'
    elif 'test' in split_name.lower():
        splitFolder = 'test'
    else:
        print(f"Error: Unknown split type in '{split_name}'.")
        return

    # Create YOLOV-specific directory structure
    if MODELTYPE == ModelType.YOLOV:
        # Create Data/VID structure for YOLOV
        output_base = os.path.join(folder_root, folder_name)
        data_vid_dir = os.path.join(output_base, "Data", "VID", splitFolder)
        annotations_vid_dir = os.path.join(output_base, "Annotations", "VID", splitFolder)
        
        try:
            os.makedirs(data_vid_dir, exist_ok=True)
            os.makedirs(annotations_vid_dir, exist_ok=True)
            print(f"Created YOLOV Data directory: {data_vid_dir}")
            print(f"Created YOLOV Annotations directory: {annotations_vid_dir}")
        except Exception as e:
            print(f"Error creating YOLOV directories: {e}")
            return
        
        split_dir = data_vid_dir
        annotation_dst = None  # YOLOV uses XML files, not JSON
        dst = split_dir  # Set dst for image copying
        
    # Create YOLOMG-specific directory structure
    elif MODELTYPE == ModelType.YOLOMG:
        # Create YOLOv5-style flat structure for YOLOMG
        output_base = os.path.join(folder_root, folder_name)
        images_dir = os.path.join(output_base, "images")
        labels_dir = os.path.join(output_base, "labels")
        masks_dir = os.path.join(output_base, "masks")
        imagesets_dir = os.path.join(output_base, "ImageSets", "Main")
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(imagesets_dir, exist_ok=True)
            print(f"Created YOLOMG images directory: {images_dir}")
            print(f"Created YOLOMG labels directory: {labels_dir}")
            print(f"Created YOLOMG masks directory: {masks_dir}")
            print(f"Created YOLOMG ImageSets directory: {imagesets_dir}")
        except Exception as e:
            print(f"Error creating YOLOMG directories: {e}")
            return
        
        split_dir = images_dir  # Images go in the flat images directory
        annotation_dst = None  # YOLOMG uses YOLO format .txt files, not JSON
        dst = split_dir  # Set dst for image copying
        
    # Create TransVOD-specific directory structure
    elif MODELTYPE == ModelType.TRANSVOD:
        # Create Data/VID structure for TransVOD
        output_base = os.path.join(folder_root, folder_name)
        data_vid_dir = os.path.join(output_base, "Data", "VID", splitFolder)
        annotations_dir = os.path.join(output_base, "annotations")
        
        try:
            os.makedirs(data_vid_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)
            print(f"Created TransVOD directory structure: {data_vid_dir}")
            print(f"Created annotations directory: {annotations_dir}")
        except Exception as e:
            print(f"Error creating TransVOD directories: {e}")
            return
        
        split_dir = data_vid_dir
        dst = split_dir  # Set dst for image copying
        
        # Set proper annotation file names for TransVOD
        if splitFolder == 'train':
            annotation_dst = os.path.join(annotations_dir, 'imagenet_vid_train.json')
        elif splitFolder == 'val':
            annotation_dst = os.path.join(annotations_dir, 'imagenet_vid_val.json')
        elif splitFolder == 'test':
            annotation_dst = os.path.join(annotations_dir, 'imagenet_vid_test.json')
    else:
        # RFDETR structure
        split_dir = os.path.join(folder_root, folder_name, splitFolder)
        try:
            os.makedirs(split_dir, exist_ok=True)
            print(f"Created split directory: {split_dir}")
        except Exception as e:
            print(f"Error creating split directory '{split_dir}': {e}")
            return
        
        annotation_dst = os.path.join(split_dir, '_annotations.coco.json')
        dst = split_dir  # Set dst for image copying

    # Always create a new empty annotation file (except for YOLOV and YOLOMG which use different formats)
    if MODELTYPE != ModelType.YOLOV and MODELTYPE != ModelType.YOLOMG:
        empty_annotation = {"images": [], "annotations": [], "categories": []}
        
        if 'visdrone' in dataset_type.lower():
            categories = visdrone_categories
        elif 'uavdt' in dataset_type.lower():
            categories = uavdt_categories 
        else:
            print(f"Error: Unknown dataset type in folder name '{dataset_type}'.")
            return
            
        if categories:
            empty_annotation['categories'] = categories
            
        # Add videos section for TransVOD
        if MODELTYPE == ModelType.TRANSVOD:
            empty_annotation['videos'] = []
            
        try:
            with open(annotation_dst, 'w') as f:
                json.dump(empty_annotation, f, indent=2)
        except Exception as e:
            print(f"Error creating empty annotation file '{annotation_dst}': {e}")
            return
    
    #something like datasets/<dataset_name>/<split_name>/sequences
    src_seq = os.path.join(folder_root, split_name, 'sequences')
    if not os.path.exists(src_seq):
        print(f"Error: Source sequence directory does not exist: {src_seq}")
        return
    #something like datasets/<dataset_name>/<split_name>/annotations
    src_annotation_file_path = os.path.join(folder_root, split_name, 'annotations')
    if not os.path.exists(src_annotation_file_path):
        print(f"Error: Source annotation file does not exist: {src_annotation_file_path}")
        return

    dst = split_dir
    fileEnumator = 0
    batch_images = []
    batch_annotations = []
    annotation_count = 0
    video_dict = {}  # Track videos for TransVOD

    image_name_dict = {}

    # Process each sequence directory
    for seq in os.listdir(src_seq):
        seq_path = os.path.join(src_seq, seq)
        # Ensure it's a directory
        if os.path.isdir(seq_path):
            print(f"Processing sequence: {seq}")
            
            # Add video entry for TransVOD
            if MODELTYPE == ModelType.TRANSVOD:
                video_name_base = seq.replace('.txt', '') if seq.endswith('.txt') else seq
                if 'uav' in video_name_base.lower():
                    video_id = int(video_name_base.replace('uav', '').replace('_', ''))
                else:
                    import re
                    numbers = re.findall(r'\d+', video_name_base)
                    video_id = int(numbers[-1]) if numbers else len(video_dict)
                
                if video_id not in video_dict:
                    video_dict[video_id] = {
                        "id": video_id,
                        "name": seq
                    }
            
            # Process each image in the sequence directory
            for img in os.listdir(seq_path):
                if img.endswith('.jpg') or img.endswith('.png'):
                    # Remove 'img' prefix from filename if present
                    img_no_prefix = img
                    if img_no_prefix.startswith('img'):
                        img_no_prefix = img_no_prefix[3:]
                    
                    # Choose file extension based on model type
                    if MODELTYPE == ModelType.YOLOV:
                        new_img_name = f"{fileEnumator:07d}.JPEG"  # YOLOV expects .JPEG
                    else:
                        new_img_name = f"{fileEnumator:07d}.jpg"   # Others use .jpg
                    
                    src_img_path = os.path.join(seq_path, img)
                    with Image.open(src_img_path) as im:
                        img_width, img_height = im.size
                    imgEntry = ImageEntry(
                        original_file_name=img_no_prefix,
                        original_file_path=src_img_path,
                        new_file_name=new_img_name,
                        height=img_height,
                        width=img_width,
                        id=fileEnumator,
                        video_name=seq+".txt"
                    )

                    image_number = int(img_no_prefix.split('.')[0])
                    image_name_dict[image_number] = imgEntry

                    fileEnumator += 1

                    # Insert picture entry to annotation file
                    if MODELTYPE != ModelType.YOLOV:
                        insertPictureInAnnotationFile(batch_images, imgEntry, splitFolder)
                    else:
                        # For YOLOV, track sequences instead of batch images
                        if splitFolder == 'train':
                            if seq not in yolov_train_sequences:
                                yolov_train_sequences[seq] = []
                            yolov_train_sequences[seq].append(new_img_name)
                        else:  # val or test
                            if seq not in yolov_val_sequences:
                                yolov_val_sequences[seq] = []
                            yolov_val_sequences[seq].append(new_img_name)

                    # Copy image to destination
                    if MODELTYPE == ModelType.YOLOV:
                        # Create sequence-specific directory for YOLOV
                        seq_dir = os.path.join(dst, seq)
                        os.makedirs(seq_dir, exist_ok=True)
                        dst_img_path = os.path.join(seq_dir, new_img_name)
                    elif MODELTYPE == ModelType.YOLOMG:
                        # For YOLOMG, use flat directory structure
                        dst_img_path = os.path.join(dst, new_img_name)
                    else:
                        dst_img_path = os.path.join(dst, new_img_name)
                    
                    src_img_path = os.path.join(seq_path, img)
                    try:
                        os.link(src_img_path, dst_img_path)
                    except Exception as e:
                        print(f"Hard link failed, falling back to copy: {e}")
                        try:
                            shutil.copy2(src_img_path, dst_img_path)
                        except Exception as copy_e:
                            print(f"Error copying image '{src_img_path}' to '{dst_img_path}': {copy_e}")

            # Insert annotation entries for picture to annotation file
            if MODELTYPE == ModelType.YOLOV:
                # For YOLOV, create XML annotations for each frame
                annotation_file_path = os.path.join(src_annotation_file_path, seq + ".txt")
                # Handle UAVDT special case where annotation files have _gt_whole suffix
                if DATASETTYPE == DatasetType.UAVDT:
                    annotation_file_path = os.path.join(src_annotation_file_path, seq + "_gt_whole.txt")
                    
                if os.path.exists(annotation_file_path):
                    print(f"Creating XML annotations for sequence {seq}...")
                    # Parse annotations and create XML files
                    output_base = os.path.join(folder_root, folder_name)
                    create_yolov_xml_annotations(annotation_file_path, image_name_dict, seq, splitFolder, output_base)
                else:
                    print(f"Warning: Annotation file not found: {annotation_file_path}")
            elif MODELTYPE == ModelType.YOLOMG:
                # For YOLOMG, create YOLO format annotations and motion masks
                annotation_file_path = os.path.join(src_annotation_file_path, seq + ".txt")
                # Handle UAVDT special case where annotation files have _gt_whole suffix
                if DATASETTYPE == DatasetType.UAVDT:
                    annotation_file_path = os.path.join(src_annotation_file_path, seq + "_gt_whole.txt")
                    
                if os.path.exists(annotation_file_path):
                    print(f"Creating YOLO annotations and motion masks for sequence {seq}...")
                    # Parse annotations and create YOLO files + motion masks
                    output_base = os.path.join(folder_root, folder_name)
                    create_yolomg_annotations(annotation_file_path, image_name_dict, seq, splitFolder, output_base)
                else:
                    print(f"Warning: Annotation file not found: {annotation_file_path}")
            else:
                annotation_count = insertNewAnnotationEntry(batch_annotations, os.path.join(src_annotation_file_path, seq + ".txt"), image_name_dict, annotation_count)
                print(f"Inserted annotations up to ID: {annotation_count}")
        else:
            print(f"Warning: Sequence path is not a directory: {seq_path}")

    # Save annotation file once after all changes (except for YOLOV and YOLOMG)
    if MODELTYPE != ModelType.YOLOV and MODELTYPE != ModelType.YOLOMG:
        try:
            with open(annotation_dst, 'r') as f:
                data = json.load(f)
            data['images'].extend(batch_images)
            data['annotations'].extend(batch_annotations)
            
            # Add videos for TransVOD
            if MODELTYPE == ModelType.TRANSVOD:
                data['videos'] = list(video_dict.values())
                
            with open(annotation_dst, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error updating annotation file '{annotation_dst}': {e}")
    
    # Generate file lists for YOLOMG
    if MODELTYPE == ModelType.YOLOMG:
        # Create file lists in ImageSets/Main directory
        output_base = os.path.join(folder_root, folder_name)
        imagesets_dir = os.path.join(output_base, "ImageSets", "Main")
        
        # Collect all image files
        images_dir = os.path.join(output_base, "images")
        image_files = []
        
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Remove extension for file list
                    img_name = os.path.splitext(img_file)[0]
                    image_files.append(img_name)
        
        image_files.sort()  # Sort for consistency
        
        # Write file lists based on split
        if splitFolder == 'train':
            train_file = os.path.join(imagesets_dir, "train.txt")
            train2_file = os.path.join(imagesets_dir, "train2.txt")
            with open(train_file, 'w') as f:
                f.write('\n'.join(image_files))
            with open(train2_file, 'w') as f:
                f.write('\n'.join(image_files))
            print(f"Created YOLOMG train file lists with {len(image_files)} images")
        elif splitFolder == 'val':
            val_file = os.path.join(imagesets_dir, "val.txt")
            val2_file = os.path.join(imagesets_dir, "val2.txt")
            with open(val_file, 'w') as f:
                f.write('\n'.join(image_files))
            with open(val2_file, 'w') as f:
                f.write('\n'.join(image_files))
            print(f"Created YOLOMG val file lists with {len(image_files)} images")
            
    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--model', type=str, required=True, choices=['rfdetr', 'transvod', 'yolov', 'yolomg'], help='Model type: rfdetr, transvod, yolov, or yolomg')
    ap.add_argument('--dataset_type', type=str, required=True, choices=datasetTypes, help='Type of dataset. Currently only supports visdrone and uavdt')
    ap.add_argument('--data_root', type=str, required=True, help='Root folder where the datasets are stored')

    args = ap.parse_args()
    chosenDataSet = args.dataset_type
    data_root = args.data_root
    model_type = args.model

    if chosenDataSet == 'visdrone':
        split_names = visdrone_split_names
        DATASETTYPE = DatasetType.VISDRONE
    elif chosenDataSet == 'uavdt':
        split_names = uavdt_split_names
        DATASETTYPE = DatasetType.UAVDT
    else:
        print(f"Error: Unknown dataset type '{chosenDataSet}'.")
        sys.exit(1)

    if model_type == 'rfdetr':
        MODELTYPE = ModelType.RFDETR
        model_dir_name = "RF_DETR_COCO"
    elif model_type == 'transvod':
        MODELTYPE = ModelType.TRANSVOD
        model_dir_name = "TRANSVOD_VID"  # Will create datasets/uavdt/TRANSVOD_VID
    elif model_type == 'yolov':
        MODELTYPE = ModelType.YOLOV
        model_dir_name = "YOLOV_VID"
    elif model_type == 'yolomg':
        MODELTYPE = ModelType.YOLOMG
        model_dir_name = "YOLOMG_VID"
    else:
        print(f"Error: Unknown model type '{model_type}'.")
        sys.exit(1)

    print(f"Dataset type: {chosenDataSet}")
    print(f"Data root: {data_root}")
    print(f"Model type: {model_type}")
    print(f"Model directory name: {model_dir_name}")
    print(f"Using split names: {split_names}")

    rf_detr_dir = os.path.join(data_root, model_dir_name)
    if not os.path.exists(rf_detr_dir):
        try:
            os.makedirs(rf_detr_dir)
            print(f"Created {model_dir_name} directory: {rf_detr_dir}")
        except Exception as e:
            print(f"Error creating {model_dir_name} directory '{rf_detr_dir}': {e}")
            sys.exit(1)
        for split_name in split_names:
            createDatasetSplitDirectory(data_root, model_dir_name, split_name, chosenDataSet)
        
        # Generate sequence files for YOLOV after all processing is complete
        if MODELTYPE == ModelType.YOLOV:
            print("\nGenerating YOLOV sequence files...")
            generate_sequence_files(rf_detr_dir, yolov_train_sequences, yolov_val_sequences)
    else:
        print(f"{model_dir_name} directory already exists: {rf_detr_dir}")
        sys.exit(1)