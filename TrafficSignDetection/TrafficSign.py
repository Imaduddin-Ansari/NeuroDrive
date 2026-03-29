"""
Combined YOLO Detection + Traffic Sign Classification
Detects traffic signs in video and classifies them in real-time
Optimized for M2 MacBook with 1280x720 display output
"""

import cv2
from ultralytics import YOLO
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

class TrafficSignDetector:
    def __init__(self, model_path, class_names_path):
        """Initialize YOLO detector and classifier"""
        print("="*60)
        print("YOLO TRAFFIC SIGN DETECTION + CLASSIFICATION")
        print("="*60)
        
        # Load YOLO model
        print("\nLoading YOLOv8 nano model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Use Metal Performance Shaders on Mac M2
        try:
            self.yolo_model.to('mps')
            print("✓ YOLO loaded on Apple Metal GPU (MPS)!")
        except:
            print("⚠ MPS not available, using CPU for YOLO")
            self.yolo_model.to('cpu')
        
        # Load classification model
        print("\nLoading traffic sign classifier...")
        self.classifier = keras.models.load_model(model_path)
        self.class_names = np.load(class_names_path, allow_pickle=True)
        print(f"✓ Classifier loaded")
        print(f"✓ Classes: {', '.join(self.class_names)}\n")
        
        # Traffic sign keywords for YOLO detection
        self.traffic_sign_keywords = ['sign', 'stop', 'speed', 'warning', 'yield', 
                                      'parking', 'no entry', 'do not', 'one way', 
                                      'street', 'route']
        
        self.img_size = 64  # Classifier input size
        self.display_width = 1280  # Display output width
        self.display_height = 720  # Display output height
    
    def resize_with_aspect_ratio(self, frame, target_width=1280, target_height=720):
        """Resize frame to target dimensions while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        
        # Calculate scaling factors
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized frame on canvas
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas, scale, x_offset, y_offset
    
    def classify_sign(self, sign_image):
        """Classify a traffic sign image"""
        if sign_image is None or sign_image.size == 0:
            return None
        
        # Preprocess image
        resized = cv2.resize(sign_image, (self.img_size, self.img_size))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0)
        
        # Make prediction
        predictions = self.classifier.predict(input_img, verbose=0)
        confidence = np.max(predictions)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [(self.class_names[idx], predictions[0][idx]) for idx in top_3_indices]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'top_3': top_3
        }
    
    def process_frame(self, frame, resize_scale=0.5, yolo_conf=0.12, classifier_conf=0.5, draw_annotations=False):
        """
        Process a single frame and return annotated frame + detections
        (Designed for UI integration - no cv2.imshow)
        
        Args:
            frame: Input frame (BGR format from cv2)
            resize_scale: Resize input to this fraction (0.5 = half size)
            yolo_conf: YOLO confidence threshold (default: 0.12)
            classifier_conf: Classifier confidence threshold (default: 0.5)
            draw_annotations: If True, draw bounding boxes and labels (default: False)
        
        Returns:
            tuple: (annotated_frame, detected_signs_list)
            - annotated_frame: Frame with bounding boxes and labels drawn (if draw_annotations=True)
            - detected_signs_list: List of dicts with keys:
                'class', 'confidence', 'bbox', 'yolo_class', 'yolo_confidence'
        """
        # Create a copy for processing
        display_frame = frame.copy()
        
        # Resize for faster YOLO processing
        small_frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
        
        # Run YOLO detection with augmentation
        detections_all = []
        
        # Original frame
        results = self.yolo_model(small_frame, verbose=False, conf=yolo_conf)
        detections_all.extend(results)
        
        # Horizontal flip
        small_frame_flipped = cv2.flip(small_frame, 1)
        results_flipped = self.yolo_model(small_frame_flipped, verbose=False, conf=yolo_conf)
        detections_all.extend(results_flipped)
        
        # Brightness augmentation
        small_frame_bright = cv2.convertScaleAbs(small_frame, alpha=1.1, beta=10)
        results_bright = self.yolo_model(small_frame_bright, verbose=False, conf=yolo_conf)
        detections_all.extend(results_bright)
        
        detected_signs = []
        seen_detections = set()
        
        # Process all detections
        for result_idx, result in enumerate(detections_all):
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_confidence = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.yolo_model.names[cls]
                
                # Mirror x coordinates if from flipped detection
                if result_idx == 1:
                    frame_width = small_frame.shape[1]
                    x1 = frame_width - x1
                    x2 = frame_width - x2
                    x1, x2 = min(x1, x2), max(x1, x2)
                
                # Scale coordinates back to original size
                x1 = int(x1 / resize_scale)
                y1 = int(y1 / resize_scale)
                x2 = int(x2 / resize_scale)
                y2 = int(y2 / resize_scale)
                
                # Clamp to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(display_frame.shape[1], x2)
                y2 = min(display_frame.shape[0], y2)
                
                # Check for duplicates
                detection_key = (x1//10, y1//10, x2//10, y2//10)
                if detection_key in seen_detections:
                    continue
                seen_detections.add(detection_key)
                
                # Check if this is a traffic sign
                is_traffic_sign = any(keyword.lower() in class_name.lower() 
                                    for keyword in self.traffic_sign_keywords)
                
                if is_traffic_sign:
                    # Extract and classify
                    sign_crop = display_frame[y1:y2, x1:x2]
                    
                    if sign_crop.size > 0:
                        classification = self.classify_sign(sign_crop)
                        
                        if classification and classification['confidence'] >= classifier_conf:
                            detected_signs.append({
                                'class': classification['class'],
                                'confidence': classification['confidence'],
                                'bbox': (x1, y1, x2, y2),
                                'yolo_class': class_name,
                                'yolo_confidence': yolo_confidence
                            })
                            
                            # Only draw if annotations are enabled
                            if draw_annotations:
                                # Draw green box for classified signs
                                color = (0, 255, 0)
                                label = f"{classification['class']}: {classification['confidence']:.2%}"
                                
                                # Draw bounding box
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label background
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                            (x1 + label_size[0], y1), color, -1)
                                
                                # Draw label text
                                cv2.putText(display_frame, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif draw_annotations:
                            # Yellow box for detected but not confidently classified
                            color = (0, 255, 255)
                            label = f"{class_name}: {yolo_confidence:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        return display_frame, detected_signs

    def process_video(self, video_path, skip_frames=2, resize_scale=0.5, 
                     yolo_conf=0.12, classifier_conf=0.5):
        """
        Process video with YOLO detection and classification
        
        Args:
            video_path: Path to video file, or 0 for webcam
            skip_frames: Process every Nth frame (2 = process every 2nd frame)
            resize_scale: Resize input to this fraction (0.5 = half size)
            yolo_conf: YOLO confidence threshold (default: 0.12)
            classifier_conf: Classifier confidence threshold
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ ERROR: Could not open video: {video_path}")
            return
        
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        source_name = "Webcam" if video_path == 0 else video_path
        
        print(f"Video Source: {source_name}")
        print(f"Input Resolution: {width}x{height}")
        print(f"Display Resolution: {self.display_width}x{self.display_height}")
        print(f"FPS: {fps}")
        print(f"Processing settings: skip_frames={skip_frames}, resize={resize_scale}")
        print(f"YOLO confidence threshold: {yolo_conf:.2f}")
        print(f"Classifier confidence threshold: {classifier_conf:.2f}")
        if video_path != 0:
            print(f"Total Frames: {total_frames}")
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  Q - Quit")
        print("  SPACE - Pause/Resume")
        print("  S - Save current frame")
        print("="*60 + "\n")
        
        frame_count = 0
        traffic_sign_count = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✓ Video ended or no more frames")
                    break
                
                frame_count += 1
                
                # Skip frames for speed
                if frame_count % skip_frames != 0:
                    # Still resize for display
                    display_frame, _, _, _ = self.resize_with_aspect_ratio(
                        frame, self.display_width, self.display_height)
                    cv2.imshow('Traffic Sign Detection', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("\n🛑 Stopped by user")
                        break
                    continue
                
                # Resize frame for faster YOLO processing
                small_frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
                
                # Run YOLO detection with augmentation
                detections_all = []
                
                # Original frame
                results = self.yolo_model(small_frame, verbose=False, conf=yolo_conf)
                detections_all.extend(results)
                
                # Horizontal flip
                small_frame_flipped = cv2.flip(small_frame, 1)
                results_flipped = self.yolo_model(small_frame_flipped, verbose=False, conf=yolo_conf)
                detections_all.extend(results_flipped)
                
                # Brightness augmentation
                small_frame_bright = cv2.convertScaleAbs(small_frame, alpha=1.1, beta=10)
                results_bright = self.yolo_model(small_frame_bright, verbose=False, conf=yolo_conf)
                detections_all.extend(results_bright)
                
                detected_this_frame = False
                seen_detections = set()
                
                # Process all detections
                for result_idx, result in enumerate(detections_all):
                    boxes = result.boxes
                    
                    if len(boxes) > 0:
                        detected_this_frame = True
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        yolo_confidence = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.yolo_model.names[cls]
                        
                        # Mirror x coordinates if from flipped detection
                        if result_idx == 1:
                            frame_width = small_frame.shape[1]
                            x1 = frame_width - x1
                            x2 = frame_width - x2
                            x1, x2 = min(x1, x2), max(x1, x2)
                        
                        # Scale coordinates back to original size
                        x1 = int(x1 / resize_scale)
                        y1 = int(y1 / resize_scale)
                        x2 = int(x2 / resize_scale)
                        y2 = int(y2 / resize_scale)
                        
                        # Clamp to frame boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        # Check for duplicates
                        detection_key = (x1//10, y1//10, x2//10, y2//10)
                        if detection_key in seen_detections:
                            continue
                        seen_detections.add(detection_key)
                        
                        # Check if this is a traffic sign
                        is_traffic_sign = any(keyword.lower() in class_name.lower() 
                                             for keyword in self.traffic_sign_keywords)
                        
                        # Extract and classify traffic sign
                        sign_crop = frame[y1:y2, x1:x2]
                        classification = None
                        
                        if is_traffic_sign and sign_crop.size > 0:
                            classification = self.classify_sign(sign_crop)
                            
                            if classification and classification['confidence'] >= classifier_conf:
                                traffic_sign_count += 1
                                print(f"🚦 Frame {frame_count}: {classification['class']} "
                                      f"({classification['confidence']:.2%}) - YOLO: {class_name}")
                                
                                # Draw green box for classified signs
                                color = (0, 255, 0)
                                label = f"{classification['class']}: {classification['confidence']:.2f}"
                            else:
                                # Yellow box for detected but not confidently classified
                                color = (0, 255, 255)
                                label = f"{class_name}: {yolo_confidence:.2f}"
                        else:
                            # Red box for other objects
                            color = (0, 0, 255)
                            label = f"{class_name}: {yolo_confidence:.2f}"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label background
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Status overlay
                status_color = (0, 255, 0) if detected_this_frame else (0, 0, 255)
                status_text = "DETECTING!" if detected_this_frame else "No Detection"
                
                cv2.putText(frame, status_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"Frame: {frame_count}", (20, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Signs Classified: {traffic_sign_count}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Resize frame for display (1280x720)
                display_frame, _, _, _ = self.resize_with_aspect_ratio(
                    frame, self.display_width, self.display_height)
            else:
                # If paused, keep showing the same frame
                display_frame, _, _, _ = self.resize_with_aspect_ratio(
                    frame, self.display_width, self.display_height)
            
            # Display frame
            cv2.imshow('Traffic Sign Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n🛑 Stopped by user")
                break
            elif key == ord(' '):
                paused = not paused
                status = "⏸ PAUSED" if paused else "▶ RESUMED"
                print(f"{status} at frame {frame_count}")
            elif key == ord('s') or key == ord('S'):
                filename = f'detection_frame_{frame_count}.jpg'
                cv2.imwrite(filename, display_frame)
                print(f"💾 Saved: {filename} (1280x720)")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Traffic signs classified: {traffic_sign_count}")
        print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMBINED TRAFFIC SIGN DETECTION + CLASSIFICATION")
    print("="*60 + "\n")
    
    # Configuration
    MODEL_PATH = "../Models/traffic_sign_model.h5"
    CLASS_NAMES_PATH = "../Models/class_names.npy"
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python combined_script.py video.mp4 [skip_frames] [resize] [yolo_conf]")
        print("  python combined_script.py 0  (for webcam)")
        print("\nExamples:")
        print("  python combined_script.py video.mp4 2 0.5 0.7")
        print("  python combined_script.py video.mp4 3 0.4 0.8")
        print("\nDefaults: skip_frames=2, resize=0.5, yolo_conf=0.12")
        print("Display output: 1280x720 (auto-resized with aspect ratio)\n")
        sys.exit(1)
    
    video_path = sys.argv[1]
    skip_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    resize_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    yolo_conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.12
    
    print(f"Video: {video_path}")
    print(f"Skip frames: {skip_frames}, Resize: {resize_scale}, YOLO Conf: {yolo_conf}\n")
    
    # Initialize detector and run
    detector = TrafficSignDetector(MODEL_PATH, CLASS_NAMES_PATH)
    detector.process_video(video_path, skip_frames, resize_scale, yolo_conf)