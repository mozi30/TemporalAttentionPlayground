import argparse
import enum
from fileinput import filename
import os
import shutil
import json
import sys
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

DATASETTYPE = DatasetType.VISDRONE
MODELTYPE = ModelType.RFDETR


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
    elif 'test' in split_name.lower():
        splitFolder = 'test'
    else:
        print(f"Error: Unknown split type in '{split_name}'.")
        return

    # Create TransVOD-specific directory structure
    if MODELTYPE == ModelType.TRANSVOD:
        # Create Data/VID structure for TransVOD
        data_vid_dir = os.path.join(folder_root + folder_name, "Data", "VID", splitFolder)
        annotations_dir = os.path.join(folder_root + folder_name, "annotations")
        
        try:
            os.makedirs(data_vid_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)
            print(f"Created TransVOD directory structure: {data_vid_dir}")
            print(f"Created annotations directory: {annotations_dir}")
        except Exception as e:
            print(f"Error creating TransVOD directories: {e}")
            return
        
        split_dir = data_vid_dir
        
        # Set proper annotation file names for TransVOD
        if splitFolder == 'train':
            annotation_dst = os.path.join(annotations_dir, 'imagenet_vid_train.json')
        elif splitFolder == 'val':
            annotation_dst = os.path.join(annotations_dir, 'imagenet_vid_val.json')
        elif splitFolder == 'test':
            annotation_dst = os.path.join(annotations_dir, 'imagenet_vid_test.json')
    else:
        # RFDETR structure
        split_dir = os.path.join(folder_root + folder_name, splitFolder)
        try:
            os.makedirs(split_dir, exist_ok=True)
            print(f"Created split directory: {split_dir}")
        except Exception as e:
            print(f"Error creating split directory '{split_dir}': {e}")
            return
        
        annotation_dst = os.path.join(split_dir, '_annotations.coco.json')

    # Always create a new empty annotation file
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
                    new_img_name = f"{fileEnumator:07d}.jpg"
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
                    insertPictureInAnnotationFile(batch_images, imgEntry, splitFolder)

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
            annotation_count = insertNewAnnotationEntry(batch_annotations, os.path.join(src_annotation_file_path, seq + ".txt"), image_name_dict, annotation_count)
            print(f"Inserted annotations up to ID: {annotation_count}")
        else:
            print(f"Warning: Sequence path is not a directory: {seq_path}")

    # Save annotation file once after all changes
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
    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--model', type=str, required=True, choices=['rfdetr', 'transvod'], help='Model type: rfdetr or transvod')
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
    else:
        print(f"{model_dir_name} directory already exists: {rf_detr_dir}")
        sys.exit(1)