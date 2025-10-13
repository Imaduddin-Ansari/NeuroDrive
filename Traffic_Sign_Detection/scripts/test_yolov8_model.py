# test_yolov8_model.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime

class YOLOv8Tester:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize YOLOv8 Tester
        
        Args:
            model_path (str): Path to the trained model weights (.pt file)
            conf_threshold (float): Confidence threshold for predictions
            iou_threshold (float): IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLOv8 model"""
        print(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"Model loaded successfully!")
                print(f"Number of classes: {len(self.class_names)}")
                print(f"Classes: {list(self.class_names.values())}")
            else:
                print("Warning: Could not retrieve class names from model")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def test_single_image(self, image_path, save_dir=None, show_result=True):
        """
        Test the model on a single image
        
        Args:
            image_path (str): Path to the image file
            save_dir (str): Directory to save results (optional)
            show_result (bool): Whether to display the result
        
        Returns:
            dict: Prediction results
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        print(f"Processing image: {image_path}")
        
        try:
            # Run inference
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=640,
                save=save_dir is not None,
                project=save_dir,
                exist_ok=True
            )
            
            # Extract results
            result = results[0]
            boxes = result.boxes
            original_image = result.orig_img
            
            # Print detection information
            print(f"Number of detections: {len(boxes) if boxes is not None else 0}")
            
            detections = []
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls.item())
                    conf = box.conf.item()
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    class_name = self.class_names[cls_id] if self.class_names else f"Class_{cls_id}"
                    
                    detection_info = {
                        'class_id': cls_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': bbox,
                        'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]
                    }
                    detections.append(detection_info)
                    
                    print(f"  Detection {i+1}: {class_name} - Confidence: {conf:.4f}")
            
            # Display result if requested
            if show_result:
                self.display_result(original_image, detections, os.path.basename(image_path))
            
            return {
                'image_path': image_path,
                'detections': detections,
                'original_image': original_image,
                'results_object': result
            }
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def test_image_directory(self, images_dir, save_dir=None, max_images=None):
        """
        Test the model on all images in a directory
        
        Args:
            images_dir (str): Path to directory containing images
            save_dir (str): Directory to save results
            max_images (int): Maximum number of images to process (optional)
        
        Returns:
            list: List of results for all processed images
        """
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return []
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(images_dir, file))
        
        if not image_files:
            print(f"No image files found in: {images_dir}")
            return []
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} images in {images_dir}")
        print(f"Processing {len(image_files)} images...")
        
        all_results = []
        for i, image_path in enumerate(image_files):
            print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self.test_single_image(
                image_path, 
                save_dir=save_dir, 
                show_result=False
            )
            
            if result:
                all_results.append(result)
        
        # Generate summary statistics
        self.generate_summary_statistics(all_results)
        
        return all_results
    
    def evaluate_on_test_set(self, data_yaml_path, save_dir=None):
        """
        Evaluate model on the test set using YOLO's built-in validation
        
        Args:
            data_yaml_path (str): Path to data.yaml file
            save_dir (str): Directory to save evaluation results
        
        Returns:
            object: Validation metrics
        """
        if not os.path.exists(data_yaml_path):
            print(f"Data YAML file not found: {data_yaml_path}")
            return None
        
        print(f"Evaluating model on test set...")
        print(f"Using data config: {data_yaml_path}")
        
        try:
            # Run validation
            metrics = self.model.val(
                data=data_yaml_path,
                split='test',  # Use test split
                imgsz=640,
                batch=16,
                conf=0.001,
                iou=0.6,
                save_json=True,
                save_hybrid=True,
                project=save_dir,
                name='test_evaluation'
            )
            
            print("\n" + "="*50)
            print("TEST SET EVALUATION RESULTS")
            print("="*50)
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.mp:.4f}")
            print(f"Recall: {metrics.box.mr:.4f}")
            
            # Print per-class AP
            if hasattr(metrics.box, 'aps') and metrics.box.aps is not None:
                print("\nPer-class AP50:")
                for i, ap in enumerate(metrics.box.ap50):
                    class_name = self.class_names[i] if self.class_names else f"Class_{i}"
                    print(f"  {class_name}: {ap:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error during test set evaluation: {e}")
            return None
    
    def display_result(self, image, detections, title="Detection Result"):
        """Display the image with bounding boxes"""
        # Convert BGR to RGB for matplotlib
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(title)
        plt.axis('off')
        
        # Draw bounding boxes
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Create rectangle
            rect = plt.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                fill=False, 
                edgecolor='red', 
                linewidth=2
            )
            plt.gca().add_patch(rect)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            plt.text(
                bbox[0], bbox[1] - 5, 
                label, 
                color='red', 
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_statistics(self, all_results):
        """Generate summary statistics from all results"""
        if not all_results:
            print("No results to generate statistics from")
            return
        
        total_detections = 0
        class_counts = {}
        confidence_scores = []
        
        for result in all_results:
            detections = result['detections']
            total_detections += len(detections)
            
            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Count class occurrences
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
                
                confidence_scores.append(confidence)
        
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total images processed: {len(all_results)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/len(all_results):.2f}")
        
        if confidence_scores:
            print(f"Average confidence: {np.mean(confidence_scores):.4f}")
            print(f"Min confidence: {np.min(confidence_scores):.4f}")
            print(f"Max confidence: {np.max(confidence_scores):.4f}")
        
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def export_results_to_csv(self, all_results, output_path):
        """Export detection results to CSV file"""
        if not all_results:
            print("No results to export")
            return
        
        data = []
        for result in all_results:
            image_name = os.path.basename(result['image_path'])
            for detection in result['detections']:
                data.append({
                    'image_name': image_name,
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'x1': detection['x1'],
                    'y1': detection['y1'],
                    'x2': detection['x2'],
                    'y2': detection['y2'],
                    'bbox_width': detection['x2'] - detection['x1'],
                    'bbox_height': detection['y2'] - detection['y1']
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")

# ============================================
# MAIN TESTING SCRIPT
# ============================================
def main():
    # Configuration
    MODEL_PATH = "../Model/Train/Weights/best.pt"  # Update this path
    DATA_YAML = "../combined_dataset/data.yaml"
    TEST_IMAGES_DIR = "../combined_dataset/images/test"
    OUTPUT_DIR = "../Model/Test"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize tester
    tester = YOLOv8Tester(
        model_path=MODEL_PATH,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    print("="*60)
    print("YOLOv8 MODEL TESTING")
    print("="*60)
    
    # # Option 1: Test on single image
    # print("\n1. Testing on single image...")
    # single_image_path = os.path.join(TEST_IMAGES_DIR, "your_test_image.jpg")  # Replace with actual image name
    # if os.path.exists(single_image_path):
    #     single_result = tester.test_single_image(
    #         image_path=single_image_path,
    #         save_dir=os.path.join(OUTPUT_DIR, "single_image"),
    #         show_result=True
    #     )
    # else:
    #     print(f"Single test image not found: {single_image_path}")
    
    # Option 2: Test on entire test directory
    print("\n2. Testing on entire test directory...")
    directory_results = tester.test_image_directory(
        images_dir=TEST_IMAGES_DIR,
        save_dir=os.path.join(OUTPUT_DIR, "batch_test")
    )
    
    # # Option 3: Formal evaluation on test set
    # print("\n3. Formal test set evaluation...")
    # test_metrics = tester.evaluate_on_test_set(
    #     data_yaml_path=DATA_YAML,
    #     save_dir=os.path.join(OUTPUT_DIR, "formal_evaluation")
    # )
    
    # Export results to CSV
    print("\n4. Exporting results to CSV...")
    csv_output_path = os.path.join(OUTPUT_DIR, "detection_results.csv")
    tester.export_results_to_csv(directory_results, csv_output_path)
    
    print(f"\nTesting completed! Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()