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
    {"id": 1, "name": "pedestrian","supercategory": "person"},
    {"id": 2, "name": "people","supercategory": "person"},
    {"id": 3, "name": "bicycle","supercategory": "vehicle"},
    {"id": 4, "name": "car","supercategory": "vehicle"},
    {"id": 5, "name": "van","supercategory": "vehicle"},
    {"id": 6, "name": "truck","supercategory": "vehicle"},
    {"id": 7, "name": "tricycle","supercategory": "vehicle"},
    {"id": 8, "name": "awning-tricycle","supercategory": "vehicle"},
    {"id": 9, "name": "bus","supercategory": "vehicle"},
    {"id": 10, "name": "motor","supercategory": "vehicle"},
    {"id": 11, "name": "others","supercategory": "unknown"},
]

uavdt_categories = [
    {"id": 1, "name": "car","supercategory": "vehicle"},
    {"id": 2, "name": "truck","supercategory": "vehicle"},
    {"id": 3, "name": "bus","supercategory": "vehicle"},
]

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

DATASETTYPE = DatasetType.VISDRONE

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
                    else:
                        print(f"Error: Unknown DATASETTYPE '{DATASETTYPE}'.")
                        return annotation_id
                    
                    image_entry = image_name_dict.get(frame_id)
                    if image_entry is None:
                        print(f"Warning: No image entry found for frame ID {frame_id} in annotation file '{src_ann}'.")
                        continue
                    
                    annotation_entry = {
                        "id": annotation_id,
                        "image_id": image_entry.id,
                        "category_id": category_id,
                        "bbox": [x_top_left, y_top_left, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    }
                    annotation_id += 1
                    annotation_data.append(annotation_entry)
                return annotation_id

        except Exception as e:
            print(f"Error reading annotation file '{src_ann}': {e}")

def insertPictureInAnnotationFile(batch_images, imgEntry):
    img_dict = {
        "file_name": imgEntry.new_file_name,
        "height": imgEntry.height,
        "width": imgEntry.width,
        "id": imgEntry.id,
        "video_name": imgEntry.video_name
    }
    
    batch_images.append(img_dict)
    return



def createDatasetSplitDirectory(folder_root, folder_name, split_name, dataset_type):
    print(f"\nProcessing split: {split_name}")
    if 'train' in split_name.lower():
        splitFolder = 'train'
    elif 'val' in split_name.lower():
        splitFolder = 'valid'
    elif 'test' in split_name.lower():
        splitFolder = 'test'
    else:
        print(f"Error: Unknown split type in '{split_name}'.")
        return

    split_dir = os.path.join(folder_root + folder_name, splitFolder)
    try:
        os.makedirs(split_dir, exist_ok=True)
        print(f"Created split directory: {split_dir}")
    except Exception as e:
        print(f"Error creating split directory '{split_dir}': {e}")
        return
    
    # Set annotation file paths
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

    image_name_dict = {}

    # Process each sequence directory
    for seq in os.listdir(src_seq):
        seq_path = os.path.join(src_seq, seq)
        # Ensure it's a directory
        if os.path.isdir(seq_path):
            print(f"Processing sequence: {seq}")
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
                    insertPictureInAnnotationFile(batch_images, imgEntry)

                    dst_img_path = os.path.join(dst, new_img_name)
                    src_img_path = os.path.join(seq_path, img)
                    try:
                        shutil.copy2(src_img_path, dst_img_path)
                    except Exception as e:
                        print(f"Error copying image '{src_img_path}' to '{dst_img_path}': {e}")

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
        with open(annotation_dst, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating annotation file '{annotation_dst}': {e}")
    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_type', type=str, required=True, choices=datasetTypes, help='Type of dataset. Currently only supports visdrone and uavdt')
    ap.add_argument('--data_root', type=str, required=True, help='Root folder where the datasets are stored')
    args = ap.parse_args()
    chosenDataSet = args.dataset_type
    data_root = args.data_root

    print(f"Dataset type: {chosenDataSet}")
    print(f"Data root: {data_root}")

    if chosenDataSet == 'visdrone':
        split_names = visdrone_split_names
        DATASETTYPE = DatasetType.VISDRONE
    elif chosenDataSet == 'uavdt':
        split_names = uavdt_split_names
        DATASETTYPE = DatasetType.UAVDT
    else:
        print(f"Error: Unknown dataset type '{chosenDataSet}'.")
        sys.exit(1)

    print(f"Using split names: {split_names}")

    rf_detr_dir = os.path.join(data_root, "RF_DETR_COCO")
    if not os.path.exists(rf_detr_dir):
        try:
            os.makedirs(rf_detr_dir)
            print(f"Created RF_DETR directory: {rf_detr_dir}")
        except Exception as e:
            print(f"Error creating RF_DETR directory '{rf_detr_dir}': {e}")
            sys.exit(1)
        for split_name in split_names:
            createDatasetSplitDirectory(data_root, "RF_DETR_COCO", split_name, chosenDataSet)
    else:
        print(f"RF_DETR directory already exists: {rf_detr_dir}")
        sys.exit(1)