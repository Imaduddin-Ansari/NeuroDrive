# combine_datasets_yolov8.py
import os
import shutil
import argparse
import csv
from collections import defaultdict
import random

def parse_unified_mapping(mapping_file_path):
    """Parse the unified class mapping file"""
    unified_mapping = {}
    class_name_to_id = {}
    
    print("Parsing unified class mapping...")
    
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header lines
    for line in lines:
        if '→' in line:
            parts = line.strip().split('→')
            if len(parts) == 2:
                original_class = parts[0].strip()
                unified_class = parts[1].strip()
                unified_mapping[original_class] = unified_class
    
    # Create class ID mapping
    unique_classes = sorted(set(unified_mapping.values()))
    for class_id, class_name in enumerate(unique_classes):
        class_name_to_id[class_name] = class_id
    
    print(f"Found {len(unified_mapping)} original classes mapped to {len(unique_classes)} unified classes")
    
    return unified_mapping, class_name_to_id

def process_gtsrb_dataset(gtsrb_csv_path, output_dir, unified_mapping, class_name_to_id, split_ratios=(0.85, 0.10, 0.05)):
    """Process GTSRB dataset and convert to YOLOv8 format"""
    print("\nProcessing GTSRB dataset...")
    
    # Create directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # GTSRB structure: CSV file with image paths and annotations
    if not os.path.exists(gtsrb_csv_path):
        print(f"GTSRB CSV file not found: {gtsrb_csv_path}")
        return 0
    
    # Read CSV annotations
    image_data = []
    with open(gtsrb_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_data.append(row)
    
    print(f"Found {len(image_data)} GTSRB images")
    
    # Process each image
    image_files = []
    gtsrb_root = os.path.dirname(gtsrb_csv_path)  # Get the directory containing the CSV
    
    for row in image_data:
        class_id = row['ClassId']
        image_path = os.path.join(gtsrb_root, row['Path'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get unified class name
        original_key = f"gtsrb_{class_id}"
        if original_key not in unified_mapping:
            print(f"Warning: No mapping found for {original_key}")
            continue
        
        unified_class = unified_mapping[original_key]
        yolo_class_id = class_name_to_id[unified_class]
        
        # Bounding box coordinates (GTSRB provides ROI)
        x1 = int(row['Roi.X1'])
        y1 = int(row['Roi.Y1'])
        x2 = int(row['Roi.X2'])
        y2 = int(row['Roi.Y2'])
        
        # Get image dimensions
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            print(f"Warning: Could not get dimensions for {image_path}")
            continue
        
        # Convert to YOLO format (normalized coordinates)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Create YOLO annotation
        yolo_annotation = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
        image_files.append({
            'path': image_path,
            'annotation': yolo_annotation,
            'class_id': yolo_class_id,
            'unified_class': unified_class
        })
    
    # Split data with better distribution
    print(f"Splitting GTSRB data: {len(image_files)} images")
    random.shuffle(image_files)
    
    train_count = int(len(image_files) * split_ratios[0])
    val_count = int(len(image_files) * split_ratios[1])
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Ensure minimum sizes
    min_val_test = 50  # Minimum images for val and test
    if len(val_files) < min_val_test and len(train_files) > min_val_test * 2:
        # Take more from train for val
        needed = min_val_test - len(val_files)
        val_files.extend(train_files[-needed:])
        train_files = train_files[:-needed]
    
    if len(test_files) < min_val_test and len(train_files) > min_val_test * 2:
        # Take more from train for test
        needed = min_val_test - len(test_files)
        test_files.extend(train_files[-needed:])
        train_files = train_files[:-needed]
    
    # Copy files and create annotations
    def copy_split_files(files, split_name):
        for file_info in files:
            # Copy image
            filename = os.path.basename(file_info['path'])
            # Ensure unique filename
            unique_filename = f"gtsrb_{filename}"
            new_image_path = os.path.join(images_dir, split_name, unique_filename)
            shutil.copy2(file_info['path'], new_image_path)
            
            # Create label file
            label_filename = os.path.splitext(unique_filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, split_name, label_filename)
            with open(label_path, 'w') as f:
                f.write(file_info['annotation'])
    
    copy_split_files(train_files, 'train')
    copy_split_files(val_files, 'val')
    copy_split_files(test_files, 'test')
    
    print(f"GTSRB - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return len(image_files)

def process_pakistani_dataset(pakistani_root, output_dir, unified_mapping, class_name_to_id, split_ratios=(0.85, 0.10, 0.05)):
    """Process Pakistani dataset and convert to YOLOv8 format"""
    print("\nProcessing Pakistani dataset...")
    
    # Create directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Pakistani structure: root/class_folders/*.jpg
    if not os.path.exists(pakistani_root):
        print(f"Pakistani root directory not found: {pakistani_root}")
        return 0
    
    image_files = []
    
    # Find all class folders
    for class_dir in os.listdir(pakistani_root):
        class_path = os.path.join(pakistani_root, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        # Get unified class name
        original_key = f"pakistani_{class_dir}"
        if original_key not in unified_mapping:
            print(f"Warning: No mapping found for {original_key}")
            continue
        
        unified_class = unified_mapping[original_key]
        yolo_class_id = class_name_to_id[unified_class]
        
        # Process each image in class directory
        for image_file in os.listdir(class_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(class_path, image_file)
                
                # For Pakistani dataset, we'll assume the entire image is the object
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                    
                    # Use entire image as bounding box (centered, covering most of the image)
                    x_center = 0.5
                    y_center = 0.5
                    width = 0.8  # Cover 80% of image width
                    height = 0.8  # Cover 80% of image height
                    
                    yolo_annotation = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    
                    image_files.append({
                        'path': image_path,
                        'annotation': yolo_annotation,
                        'class_id': yolo_class_id,
                        'unified_class': unified_class
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not process {image_path}: {e}")
    
    print(f"Found {len(image_files)} Pakistani images")
    
    if len(image_files) == 0:
        print("No Pakistani images found! Check the directory structure.")
        return 0
    
    # Split data with better distribution
    print(f"Splitting Pakistani data: {len(image_files)} images")
    random.shuffle(image_files)
    
    train_count = int(len(image_files) * split_ratios[0])
    val_count = int(len(image_files) * split_ratios[1])
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Ensure minimum sizes
    min_val_test = 30  # Minimum images for val and test
    if len(val_files) < min_val_test and len(train_files) > min_val_test * 2:
        # Take more from train for val
        needed = min_val_test - len(val_files)
        val_files.extend(train_files[-needed:])
        train_files = train_files[:-needed]
    
    if len(test_files) < min_val_test and len(train_files) > min_val_test * 2:
        # Take more from train for test
        needed = min_val_test - len(test_files)
        test_files.extend(train_files[-needed:])
        train_files = train_files[:-needed]
    
    # Copy files and create annotations
    def copy_split_files(files, split_name):
        for file_info in files:
            # Copy image
            filename = os.path.basename(file_info['path'])
            # Ensure unique filename
            unique_filename = f"pakistani_{file_info['unified_class']}_{filename}"
            new_image_path = os.path.join(images_dir, split_name, unique_filename)
            shutil.copy2(file_info['path'], new_image_path)
            
            # Create label file
            label_filename = os.path.splitext(unique_filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, split_name, label_filename)
            with open(label_path, 'w') as f:
                f.write(file_info['annotation'])
    
    copy_split_files(train_files, 'train')
    copy_split_files(val_files, 'val')
    copy_split_files(test_files, 'test')
    
    print(f"Pakistani - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return len(image_files)

def create_yolov8_config(output_dir, class_name_to_id):
    """Create YOLOv8 configuration files"""
    print("\nCreating YOLOv8 configuration...")
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_name_to_id),
        'names': {class_id: class_name for class_name, class_id in class_name_to_id.items()}
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"test: {data_yaml['test']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for class_id, class_name in sorted(data_yaml['names'].items()):
            f.write(f"  {class_id}: '{class_name}'\n")
    
    # Create classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for class_name, class_id in sorted(class_name_to_id.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")
    
    print(f"Created data.yaml with {len(class_name_to_id)} classes")

def main():
    parser = argparse.ArgumentParser(description='Combine datasets for YOLOv8 training')
    parser.add_argument('--gtsrb_root', required=True, help='Path to GTSRB Train.csv file')
    parser.add_argument('--pakistani_root', required=True, help='Path to Pakistani dataset root directory with class folders')
    parser.add_argument('--mapping_file', required=True, help='Path to unified class mapping file')
    parser.add_argument('--output_dir', default='./combined_yolov8_dataset', help='Output directory')
    parser.add_argument('--split_ratios', nargs=3, type=float, default=[0.85, 0.10, 0.05], 
                       help='Train/val/test split ratios (default: 0.85 0.10 0.05)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMBINING DATASETS FOR YOLOv8 TRAINING")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse unified mapping
    unified_mapping, class_name_to_id = parse_unified_mapping(args.mapping_file)
    
    # Process datasets
    gtsrb_count = process_gtsrb_dataset(
        args.gtsrb_root, 
        args.output_dir, 
        unified_mapping, 
        class_name_to_id,
        args.split_ratios
    )
    
    pakistani_count = process_pakistani_dataset(
        args.pakistani_root, 
        args.output_dir, 
        unified_mapping, 
        class_name_to_id,
        args.split_ratios
    )
    
    # Create YOLOv8 configuration
    create_yolov8_config(args.output_dir, class_name_to_id)
    
    total_images = gtsrb_count + pakistani_count
    print("\n" + "=" * 60)
    print("DATASET COMBINATION COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    print(f"  - GTSRB: {gtsrb_count} images ({gtsrb_count/total_images*100:.1f}%)")
    print(f"  - Pakistani: {pakistani_count} images ({pakistani_count/total_images*100:.1f}%)")
    print(f"Total unified classes: {len(class_name_to_id)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: Train {args.split_ratios[0]*100}%, Val {args.split_ratios[1]*100}%, Test {args.split_ratios[2]*100}%")
    
    # Calculate actual split counts
    train_dir = os.path.join(args.output_dir, 'images', 'train')
    val_dir = os.path.join(args.output_dir, 'images', 'val')
    test_dir = os.path.join(args.output_dir, 'images', 'test')
    
    train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    val_count = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
    test_count = len(os.listdir(test_dir)) if os.path.exists(test_dir) else 0
    
    print(f"\nActual split counts:")
    print(f"  - Train: {train_count} images ({train_count/total_images*100:.1f}%)")
    print(f"  - Val: {val_count} images ({val_count/total_images*100:.1f}%)")
    print(f"  - Test: {test_count} images ({test_count/total_images*100:.1f}%)")
    
    print("\nYou can now train YOLOv8 using:")
    print(f"yolo task=detect mode=train model=yolov8n.pt data={args.output_dir}/data.yaml epochs=100 imgsz=640")

if __name__ == "__main__":
    main()