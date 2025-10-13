# yolo_detection_display.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

class YOLOv8Detector:
    def __init__(self, model_path, conf_threshold=0.25):
        """
        Initialize YOLOv8 Detector for real-time detection display
        
        Args:
            model_path (str): Path to the trained model weights (.pt file)
            conf_threshold (float): Confidence threshold for predictions
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_names = None
        self.colors = None
        
        self.load_model()
        self.generate_colors()
    
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
    
    def generate_colors(self, num_colors=100):
        """Generate distinct colors for different classes"""
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.randint(0, 255, size=(num_colors, 3))
    
    def detect_and_display(self, image_path, output_path=None, show_result=True, save_result=True):
        """
        Detect objects in an image and display with bounding boxes and class labels
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the output image (optional)
            show_result (bool): Whether to display the result
            save_result (bool): Whether to save the result
        
        Returns:
            tuple: (annotated_image, detections_list)
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None, []
        
        print(f"Processing image: {image_path}")
        
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image from {image_path}")
                return None, []
            
            original_image = image.copy()
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                imgsz=640,
                verbose=False  # Set to True for detailed inference info
            )
            
            # Extract results
            result = results[0]
            boxes = result.boxes
            
            # Print detection information
            print(f"Number of detections: {len(boxes) if boxes is not None else 0}")
            
            detections = []
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls.item())
                    conf = box.conf.item()
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    
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
                    
                    # Draw bounding box and label on the image
                    image = self.draw_detection(image, detection_info, i)
            
            # Add detection summary to image
            image = self.add_summary_to_image(image, detections)
            
            # Save result if requested
            if save_result and output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                cv2.imwrite(output_path, image)
                print(f"Result saved to: {output_path}")
            
            # Display result if requested
            if show_result:
                self.display_image(image, os.path.basename(image_path))
            
            return image, detections
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, []
    
    def draw_detection(self, image, detection, detection_id):
        """
        Draw bounding box and label for a single detection
        
        Args:
            image: Input image
            detection: Detection dictionary
            detection_id: ID of the detection for color selection
        
        Returns:
            image: Image with drawn detection
        """
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get color for this detection
        color = self.colors[detection_id % len(self.colors)].tolist()
        
        # Draw bounding box
        cv2.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            3  # Thicker line for better visibility
        )
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (bbox[0], bbox[1] - text_height - 10),
            (bbox[0] + text_width, bbox[1]),
            color,
            -1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (bbox[0], bbox[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text
            2,
            cv2.LINE_AA
        )
        
        # Add detection ID
        cv2.putText(
            image,
            f"ID: {detection_id}",
            (bbox[0], bbox[3] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
        
        return image
    
    def add_summary_to_image(self, image, detections):
        """
        Add detection summary to the top of the image
        
        Args:
            image: Input image
            detections: List of detections
        
        Returns:
            image: Image with summary added
        """
        # Create summary text
        summary_text = f"Detections: {len(detections)}"
        
        # Add summary background
        h, w = image.shape[:2]
        cv2.rectangle(
            image,
            (10, 10),
            (300, 50),
            (0, 0, 0),
            -1
        )
        
        # Add summary text
        cv2.putText(
            image,
            summary_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return image
    
    def display_image(self, image, title="YOLOv8 Detection Results"):
        """
        Display the image using matplotlib with proper formatting
        
        Args:
            image: Image to display (BGR format from OpenCV)
            title: Title for the plot
        """
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def process_multiple_images(self, image_paths, output_dir=None, show_results=True):
        """
        Process multiple images and display/save results
        
        Args:
            image_paths (list): List of image paths to process
            output_dir (str): Directory to save results
            show_results (bool): Whether to display each result
        
        Returns:
            list: List of detection results for each image
        """
        all_results = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Generate output path
            if output_dir:
                output_path = os.path.join(
                    output_dir, 
                    f"detected_{os.path.basename(image_path)}"
                )
            else:
                output_path = None
            
            # Process image
            result_image, detections = self.detect_and_display(
                image_path=image_path,
                output_path=output_path,
                show_result=show_results,
                save_result=output_path is not None
            )
            
            all_results.append({
                'image_path': image_path,
                'detections': detections,
                'result_image': result_image
            })
        
        return all_results

# ============================================
# MAIN DETECTION AND DISPLAY SCRIPT
# ============================================
def main():
    # Configuration
    MODEL_PATH = "../Model/Train/Weights/best.pt"  # Update this path
    TEST_IMAGE = "../TestImages/test.jpg"  # Replace with your test image path
    OUTPUT_DIR = "../Model/TestImages"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize detector
    detector = YOLOv8Detector(
        model_path=MODEL_PATH,
        conf_threshold=0.8
    )
    
    print("="*60)
    print("YOLOv8 DETECTION AND DISPLAY")
    print("="*60)
    
    # Option 1: Process single image
    if os.path.exists(TEST_IMAGE):
        print(f"\nProcessing single image: {TEST_IMAGE}")
        
        output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(TEST_IMAGE)}")
        
        result_image, detections = detector.detect_and_display(
            image_path=TEST_IMAGE,
            output_path=output_path,
            show_result=True,
            save_result=True
        )
        
        if result_image is not None:
            print(f"\nDetection completed!")
            print(f"Found {len(detections)} objects in the image")
        else:
            print(f"\nFailed to process image: {TEST_IMAGE}")
    else:
        print(f"Test image not found: {TEST_IMAGE}")
    
    # Option 2: Process multiple images from a directory
    # TEST_IMAGES_DIR = "../combined_dataset/images/test"
    # if os.path.exists(TEST_IMAGES_DIR):
    #     print(f"\nProcessing multiple images from: {TEST_IMAGES_DIR}")
    #     
    #     # Get all image files
    #     image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    #     image_files = []
    #     
    #     for file in os.listdir(TEST_IMAGES_DIR):
    #         if any(file.lower().endswith(ext) for ext in image_extensions):
    #             image_files.append(os.path.join(TEST_IMAGES_DIR, file))
    #     
    #     if image_files:
    #         # Process first 5 images as example
    #         sample_images = image_files[:5]
    #         results = detector.process_multiple_images(
    #             image_paths=sample_images,
    #             output_dir=OUTPUT_DIR,
    #             show_results=True
    #         )
    #         
    #         print(f"\nProcessed {len(results)} images successfully!")
    #     else:
    #         print("No image files found in the directory")
    # else:
    #     print(f"Test images directory not found: {TEST_IMAGES_DIR}")

# ============================================
# SIMPLE USAGE EXAMPLE
# ============================================
def simple_example():
    """
    Simple example for quick testing
    """
    model_path = "../Model/Train/Weights/best.pt"
    test_image = "your_test_image.jpg"
    
    # Initialize detector
    detector = YOLOv8Detector(model_path)
    
    # Detect and display
    result_image, detections = detector.detect_and_display(test_image)
    
    return result_image, detections

if __name__ == "__main__":
    main()