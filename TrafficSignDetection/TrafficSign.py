"""
Combined YOLO Detection + GTSRB Traffic Sign Classification
Detects traffic signs in video and classifies them using GTSRB model
Optimized for M2 MacBook - Now with alert system and external callable interface
"""

import cv2
from ultralytics import YOLO
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from collections import defaultdict

class TrafficSignDetector:
    def __init__(self, model_path, alert_threshold=0.90, verbose=True):
        """
        Initialize YOLO detector and GTSRB classifier
        
        Args:
            model_path: Path to the GTSRB model file
            alert_threshold: Minimum confidence for alerts (default: 0.90)
            verbose: Print messages during initialization (default: True)
        """
        self.verbose = verbose
        self.alert_threshold = alert_threshold
        
        if self.verbose:
            print("="*60)
            print("YOLO DETECTION + GTSRB CLASSIFICATION (43 classes)")
            print(f"Alert Threshold: {alert_threshold*100:.0f}%")
            print("="*60)
        
        # Load YOLO model
        if self.verbose:
            print("\nLoading YOLOv8 nano model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Use Metal Performance Shaders on Mac M2
        try:
            self.yolo_model.to('mps')
            if self.verbose:
                print("✓ YOLO loaded on Apple Metal GPU (MPS)!")
        except:
            if self.verbose:
                print("⚠ MPS not available, using CPU for YOLO")
            self.yolo_model.to('cpu')
        
        # Load GTSRB classification model
        if self.verbose:
            print("\nLoading GTSRB traffic sign classifier...")
        self.classifier = keras.models.load_model(model_path)
        
        # GTSRB class names (43 German traffic signs) - EXACT OFFICIAL MAPPING
        self.class_names = [
            'Speed limit 20 km/h',           # 0
            'Speed limit 30 km/h',           # 1
            'Speed limit 50 km/h',           # 2
            'Speed limit 60 km/h',           # 3
            'Speed limit 70 km/h',           # 4
            'Speed limit 80 km/h',           # 5
            'End of Speed limit 80 km/h',    # 6
            'Speed limit 100 km/h',          # 7
            'Speed limit 120 km/h',          # 8
            'No passing',                     # 9
            'No passing by vehicles over 3.5T',  # 10
            'Right of way at the next intersection',  # 11
            'Priority road',                  # 12
            'Yield',                          # 13
            'Stop',                           # 14
            'No vehicles',                    # 15
            'No vehicles over 3.5T',         # 16
            'No Entry',                       # 17
            'General caution',                # 18
            'Dangerous curve to the left',    # 19
            'Dangerous curve to the right',   # 20
            'Double curve',                   # 21
            'Bumpy road',                     # 22
            'Slippery road',                  # 23
            'Road narrows on the right',      # 24
            'Road work',                      # 25
            'Traffic signal',                 # 26
            'Pedestrians',                    # 27
            'Children crossing',              # 28
            'Bicycles crossing',              # 29
            'Beware of ice/snow',            # 30
            'Wild animals',                   # 31
            'End of all speed and passing restrictions',  # 32
            'Turn right ahead',               # 33
            'Turn left ahead',                # 34
            'Ahead only',                     # 35
            'Go straight or right',           # 36
            'Go straight or left',            # 37
            'Keep left',                      # 38
            'keep right',                     # 39
            'Roundabout mandatory',           # 40
            'End of No passing',              # 41
            'End of No passing by vehicles over 3.5T'  # 42
        ]
        
        if self.verbose:
            print(f"✓ GTSRB Classifier loaded")
            print(f"✓ Number of classes: {len(self.class_names)}")
            print(f"✓ Model trained on German traffic signs")
        
        # Traffic sign keywords for YOLO detection
        self.traffic_sign_keywords = ['sign', 'stop', 'speed', 'warning', 'yield', 
                                      'parking', 'no entry', 'do not', 'one way', 
                                      'street', 'route']
        
        # Auto-detect image size from model input shape
        model_input_shape = self.classifier.input_shape
        self.img_size = model_input_shape[1]
        
        if self.verbose:
            print(f"✓ Model expects input size: {self.img_size}x{self.img_size}\n")
    
    def classify_sign(self, sign_image):
        """Classify a traffic sign image using GTSRB model"""
        if sign_image is None or sign_image.size == 0:
            return None
        
        # Preprocess image to match GTSRB training pipeline
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
            'class_idx': predicted_class_idx,
            'confidence': confidence,
            'top_3': top_3,
            'is_alert': confidence >= self.alert_threshold
        }
    
    def process_video(self, video_path, skip_frames=2, resize_scale=0.5, 
                     yolo_conf=0.12, classifier_conf=0.5, display=True,
                     save_alerts=False, alert_output_dir='alerts'):
        """
        Process video with YOLO detection and GTSRB classification
        
        Args:
            video_path: Path to video file, or 0 for webcam
            skip_frames: Process every Nth frame (2 = process every 2nd frame)
            resize_scale: Resize input to this fraction (0.5 = half size)
            yolo_conf: YOLO confidence threshold
            classifier_conf: Classifier confidence threshold for displaying
            display: Show video window (default: True)
            save_alerts: Save frames with high-confidence detections (default: False)
            alert_output_dir: Directory to save alert frames
        
        Returns:
            dict: Complete results including all detections and alerts
        """
        # Initialize results dictionary
        results = {
            'video_path': video_path,
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'alerts': [],  # High confidence detections (>= alert_threshold)
            'all_detections': [],  # All detections above classifier_conf
            'class_counts': defaultdict(int),
            'alert_class_counts': defaultdict(int),
            'start_time': datetime.now(),
            'end_time': None
        }
        
        # Create alert directory if needed
        if save_alerts:
            import os
            os.makedirs(alert_output_dir, exist_ok=True)
        
        # Open video or webcam
        if video_path == "0" or video_path == 0:
            video_path = 0
            if self.verbose:
                print("Attempting to open default webcam (index 0)...")
        
        cap = cv2.VideoCapture(video_path)
        
        # For webcam, try additional backends if default fails
        if video_path == 0 and not cap.isOpened():
            if self.verbose:
                print("Default backend failed, trying AVFoundation (macOS)...")
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if video_path == 0 and not cap.isOpened():
            if self.verbose:
                print("Trying different webcam indices...")
            for i in range(1, 5):
                if self.verbose:
                    print(f"  Trying webcam index {i}...")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    if self.verbose:
                        print(f"✓ Webcam found at index {i}")
                    break
                cap.release()
        
        if not cap.isOpened():
            error_msg = f"Could not open video source: {video_path}"
            if self.verbose:
                print(f"❌ ERROR: {error_msg}")
                if video_path == 0:
                    print("\n🔧 Webcam Troubleshooting:")
                    print("1. Check if another app is using the webcam")
                    print("2. On macOS, grant Terminal camera permissions:")
                    print("   System Preferences → Security & Privacy → Camera")
            results['error'] = error_msg
            return results
        
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results['fps'] = fps
        results['resolution'] = (width, height)
        results['total_frames'] = total_frames if video_path != 0 else 0
        
        source_name = "Webcam" if video_path == 0 else video_path
        
        if self.verbose:
            print(f"Video Source: {source_name}")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps}")
            print(f"Processing settings: skip_frames={skip_frames}, resize={resize_scale}")
            if video_path != 0:
                print(f"Total Frames: {total_frames}")
            print("\n" + "="*60)
            print("ALERT SYSTEM ACTIVE")
            print(f"🚨 Alerts triggered for confidence ≥ {self.alert_threshold*100:.0f}%")
            print("="*60)
            if display:
                print("\nCONTROLS:")
                print("  Q - Quit")
                print("  SPACE - Pause/Resume")
                print("  S - Save current frame")
                print("="*60 + "\n")
        
        frame_count = 0
        traffic_sign_count = 0
        alert_count = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if self.verbose:
                        print("\n✓ Video ended or no more frames")
                    break
                
                frame_count += 1
                
                # Skip frames for speed
                if frame_count % skip_frames != 0:
                    if display:
                        cv2.imshow('GTSRB Traffic Sign Detection', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q'):
                            if self.verbose:
                                print("\n🛑 Stopped by user")
                            break
                    continue
                
                results['processed_frames'] += 1
                
                # Resize frame for faster YOLO processing
                small_frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
                
                # Run YOLO detection with augmentation
                detections_all = []
                
                # Original frame
                yolo_results = self.yolo_model(small_frame, verbose=False, conf=yolo_conf)
                detections_all.extend(yolo_results)
                
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
                frame_has_alert = False
                
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
                        
                        if is_traffic_sign and sign_crop.size > 0:
                            classification = self.classify_sign(sign_crop)
                            
                            if classification and classification['confidence'] >= classifier_conf:
                                traffic_sign_count += 1
                                
                                # Create detection record
                                detection_record = {
                                    'frame': frame_count,
                                    'timestamp': (frame_count / fps) if video_path != 0 and fps > 0 else None,
                                    'class': classification['class'],
                                    'class_idx': classification['class_idx'],
                                    'confidence': float(classification['confidence']),
                                    'bbox': (x1, y1, x2, y2),
                                    'top_3': classification['top_3'],
                                    'is_alert': classification['is_alert']
                                }
                                
                                results['all_detections'].append(detection_record)
                                results['class_counts'][classification['class']] += 1
                                
                                # Check if this is a high-confidence ALERT (>90%)
                                if classification['is_alert']:
                                    alert_count += 1
                                    frame_has_alert = True
                                    results['alerts'].append(detection_record)
                                    results['alert_class_counts'][classification['class']] += 1
                                    
                                    # ONLY print alerts with >90% confidence
                                    if self.verbose:
                                        print(f"\n{'='*60}")
                                        print(f"🚨 ALERT! High Confidence Detection")
                                        print(f"{'='*60}")
                                        print(f"Frame: {frame_count}")
                                        print(f"Sign: {classification['class']}")
                                        print(f"Confidence: {classification['confidence']:.2%}")
                                        print(f"Top 3: ", end="")
                                        for i, (cls_name, conf) in enumerate(classification['top_3']):
                                            print(f"{cls_name} ({conf:.2%})", end="")
                                            if i < 2:
                                                print(", ", end="")
                                        print(f"\n{'='*60}\n")
                                    
                                    # Save alert frame if requested (disabled by default)
                                    if save_alerts:
                                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = f"{alert_output_dir}/alert_frame{frame_count}_{classification['class'].replace(' ', '_')}_{timestamp_str}.jpg"
                                        # Draw bounding box on saved image
                                        alert_frame = frame.copy()
                                        cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                        cv2.putText(alert_frame, f"{classification['class']}: {classification['confidence']:.2%}", 
                                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        cv2.imwrite(filename, alert_frame)
                                        if self.verbose:
                                            print(f"💾 Alert frame saved: {filename}")
                
                # Status overlay for display
                if display:
                    status_color = (0, 255, 0) if detected_this_frame else (0, 0, 255)
                    status_text = "DETECTING!" if detected_this_frame else "No Detection"
                    
                    cv2.putText(frame, status_text, (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    cv2.putText(frame, f"Frame: {frame_count}", (20, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Signs: {traffic_sign_count}", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Alerts: {alert_count}", (20, 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Show alert indicator
                    if frame_has_alert:
                        cv2.putText(frame, "!!! ALERT !!!", (width - 200, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # Display frame
            if display:
                cv2.imshow('GTSRB Traffic Sign Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    if self.verbose:
                        print("\n🛑 Stopped by user")
                    break
                elif key == ord(' '):
                    paused = not paused
                    status = "⏸ PAUSED" if paused else "▶ RESUMED"
                    if self.verbose:
                        print(f"{status} at frame {frame_count}")
                elif key == ord('s') or key == ord('S'):
                    filename = f'detection_frame_{frame_count}.jpg'
                    cv2.imwrite(filename, frame)
                    if self.verbose:
                        print(f"💾 Saved: {filename}")
        
        # Cleanup
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        # Finalize results
        results['end_time'] = datetime.now()
        results['total_detections'] = traffic_sign_count
        results['total_alerts'] = alert_count
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # Print summary
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print detailed summary of processing results"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total frames processed: {results['processed_frames']}")
        print(f"Processing duration: {results['duration']:.2f} seconds")
        print(f"Total traffic signs detected: {results['total_detections']}")
        print(f"🚨 High-confidence alerts (≥{self.alert_threshold*100:.0f}%): {results['total_alerts']}")
        
        if results['alert_class_counts']:
            print(f"\nAlert Breakdown by Class:")
            for sign_class, count in sorted(results['alert_class_counts'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"  • {sign_class}: {count}")
        
        if results['class_counts']:
            print(f"\nAll Detections by Class:")
            for sign_class, count in sorted(results['class_counts'].items(), 
                                           key=lambda x: x[1], reverse=True)[:10]:
                print(f"  • {sign_class}: {count}")
        
        print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMBINED TRAFFIC SIGN DETECTION + CLASSIFICATION")
    print("Using GTSRB Model (43 German Traffic Sign Classes)")
    print("="*60 + "\n")
    
    # Configuration
    MODEL_PATH = "Models/gtsrb_kaggle_model.h5"
    ALERT_THRESHOLD = 0.90  # Alert on 90%+ confidence
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python combined_script.py video.mp4 [skip_frames] [resize]")
        print("  python combined_script.py 0  (for webcam)")
        print("  python combined_script.py 1  (for second webcam)")
        print("\nExamples:")
        print("  python combined_script.py video.mp4 2 0.5")
        print("  python combined_script.py 0 1 0.5  (webcam, no skip)")
        print("\nDefaults: skip_frames=2, resize=0.5")
        print(f"Alert threshold: {ALERT_THRESHOLD*100:.0f}%")
        print("\nRequired file:")
        print(f"  - {MODEL_PATH}")
        print("\n💡 Tip: For webcam on macOS, ensure Terminal has camera access\n")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Handle webcam indices
    try:
        video_path = int(video_path)
    except ValueError:
        pass  # It's a file path, keep as string
    
    skip_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    resize_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    if isinstance(video_path, int):
        print(f"Webcam index: {video_path}")
    else:
        print(f"Video: {video_path}")
    print(f"Skip frames: {skip_frames}, Resize: {resize_scale}\n")
    
    # Initialize detector with alert threshold
    detector = TrafficSignDetector(MODEL_PATH, alert_threshold=ALERT_THRESHOLD, verbose=True)
    
    # Process video and get results (save_alerts disabled by default)
    results = detector.process_video(
        video_path, 
        skip_frames=skip_frames, 
        resize_scale=resize_scale,
        display=True,
        save_alerts=False  # Disabled to avoid unnecessary computation
    )
    
    # Results are now available in the 'results' dictionary
    # You can access: results['alerts'], results['all_detections'], etc.