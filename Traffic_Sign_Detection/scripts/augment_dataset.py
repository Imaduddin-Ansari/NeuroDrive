import os
import cv2
import random
import numpy as np
from pathlib import Path

class TrafficSignAugmentor:
    def __init__(self, dataset_path, output_path=None):
        """
        Initialize the augmentor with dataset paths
        
        Args:
            dataset_path: Path to the combined_dataset folder
            output_path: Optional output path (if None, modifies in-place)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path) if output_path else self.dataset_path
        
        # Create output directories if they don't exist
        if output_path:
            self._create_output_dirs()
        
    def _create_output_dirs(self):
        """Create output directory structure"""
        dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_path in dirs:
            (self.output_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    def add_random_occlusions(self, image, num_boxes=3, max_box_size=0.3):
        """
        Add random black boxes to occlude parts of the image
        
        Args:
            image: Input image (numpy array)
            num_boxes: Number of random boxes to add
            max_box_size: Maximum box size as fraction of image dimensions
            
        Returns:
            Augmented image with occlusions
        """
        augmented = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_boxes):
            # Random box size (10% to max_box_size% of image dimensions)
            box_w = random.randint(int(w * 0.1), int(w * max_box_size))
            box_h = random.randint(int(h * 0.1), int(h * max_box_size))
            
            # Random position
            x1 = random.randint(0, w - box_w - 1)
            y1 = random.randint(0, h - box_h - 1)
            
            # Draw black rectangle
            cv2.rectangle(augmented, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
        
        return augmented
    
    def add_traffic_sign_like_occlusions(self, image, num_occlusions=2):
        """
        Add occlusions that might resemble real-world obstructions
        (mud, stickers, partial coverage)
        """
        augmented = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_occlusions):
            # Choose between different occlusion types
            occlusion_type = random.choice(['rectangle', 'circle', 'polygon'])
            
            if occlusion_type == 'rectangle':
                # Random rectangle
                box_w = random.randint(int(w * 0.05), int(w * 0.3))
                box_h = random.randint(int(h * 0.05), int(h * 0.3))
                x1 = random.randint(0, w - box_w - 1)
                y1 = random.randint(0, h - box_h - 1)
                cv2.rectangle(augmented, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
            
            elif occlusion_type == 'circle':
                # Random circle
                radius = random.randint(int(min(w, h) * 0.05), int(min(w, h) * 0.2))
                center_x = random.randint(radius, w - radius - 1)
                center_y = random.randint(radius, h - radius - 1)
                cv2.circle(augmented, (center_x, center_y), radius, (0, 0, 0), -1)
            
            elif occlusion_type == 'polygon':
                # Random polygon (triangle or quadrilateral)
                num_vertices = random.choice([3, 4])
                vertices = []
                for _ in range(num_vertices):
                    x = random.randint(0, w - 1)
                    y = random.randint(0, h - 1)
                    vertices.append([x, y])
                cv2.fillPoly(augmented, [np.array(vertices)], (0, 0, 0))
        
        return augmented
    
    def augment_dataset_split(self, split='train', num_augmentations=2):
        """
        Augment images in a specific split (train, val, or test)
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            num_augmentations: Number of augmented versions to create per image
        """
        images_dir = self.dataset_path / 'images' / split
        labels_dir = self.dataset_path / 'labels' / split
        
        output_images_dir = self.output_path / 'images' / split
        output_labels_dir = self.output_path / 'labels' / split
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist. Skipping {split} split.")
            return
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + \
                     list(images_dir.glob('*.jpeg'))
        
        print(f"Augmenting {split} split: {len(image_files)} images")
        
        for img_path in image_files:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Get corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            for aug_idx in range(num_augmentations):
                # Choose augmentation method randomly
                if random.random() < 0.7:  # 70% chance for random boxes
                    augmented_image = self.add_random_occlusions(
                        image, 
                        num_boxes=random.randint(1, 4),
                        max_box_size=random.uniform(0.2, 0.4)
                    )
                else:  # 30% chance for traffic-sign-like occlusions
                    augmented_image = self.add_traffic_sign_like_occlusions(
                        image,
                        num_occlusions=random.randint(1, 3)
                )
                
                # Save augmented image
                aug_img_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                aug_img_path = output_images_dir / aug_img_name
                cv2.imwrite(str(aug_img_path), augmented_image)
                
                # Copy corresponding label file
                if label_path.exists():
                    aug_label_name = f"{img_path.stem}_aug{aug_idx}.txt"
                    aug_label_path = output_labels_dir / aug_label_name
                    
                    import shutil
                    shutil.copy2(str(label_path), str(aug_label_path))
        
        print(f"Completed augmenting {split} split")
    
    def augment_all_splits(self, splits=None, num_augmentations=2):
        """
        Augment all dataset splits
        
        Args:
            splits: List of splits to augment (default: ['train'])
            num_augmentations: Number of augmented versions per image
        """
        if splits is None:
            splits = ['train']  # Usually only augment training data
        
        for split in splits:
            self.augment_dataset_split(split, num_augmentations)

def main():
    # Configuration
    DATASET_PATH = "../combined_dataset/"  # Relative to script location
    OUTPUT_PATH = "../combined_dataset_augmented"  # New augmented dataset
    
    # Create augmentor
    augmentor = TrafficSignAugmentor(DATASET_PATH, OUTPUT_PATH)
    
    # Augment only training data (common practice)
    print("Starting data augmentation with synthetic occlusions...")
    augmentor.augment_all_splits(
        splits=['train'],  # Only augment training data
        num_augmentations=3  # Create 3 augmented versions per image
    )
    
    # Copy validation and test splits without augmentation
    print("Copying validation and test splits...")
    import shutil
    for split in ['val', 'test']:
        # Copy images
        src_images = Path(DATASET_PATH) / 'images' / split
        dst_images = Path(OUTPUT_PATH) / 'images' / split
        
        if src_images.exists():
            shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
        
        # Copy labels
        src_labels = Path(DATASET_PATH) / 'labels' / split
        dst_labels = Path(OUTPUT_PATH) / 'labels' / split
        
        if src_labels.exists():
            shutil.copytree(src_labels, dst_labels, dirs_exist_ok=True)
    
    # Copy data.yaml and classes.txt
    for file in ['data.yaml', 'classes.txt']:
        src_file = Path(DATASET_PATH) / file
        dst_file = Path(OUTPUT_PATH) / file
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
    
    print("Data augmentation completed!")
    print(f"Original dataset: {DATASET_PATH}")
    print(f"Augmented dataset: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()