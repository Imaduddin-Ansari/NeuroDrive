import cv2
import numpy as np
import os
import sys
import hashlib

class BlindSpotMonitor:
    def __init__(self, side='left'):
        self.side = side.lower()
        if self.side not in ['left', 'right']:
            raise ValueError("Side must be 'left' or 'right'")
        
        # Create results folder if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        prototxt_path = 'Model/deploy.prototxt'
        caffemodel_path = 'Model/mobilenet_iter_73000.caffemodel'
        
        # Verify files exist
        if not os.path.exists(prototxt_path):
            raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
        if not os.path.exists(caffemodel_path):
            raise FileNotFoundError(f"Caffemodel file not found: {caffemodel_path}")
        
        # Check file sizes and hashes for diagnostics
        proto_size = os.path.getsize(prototxt_path)
        model_size = os.path.getsize(caffemodel_path)
        
        print(f"Loading MobileNet SSD model for {self.side} blind spot...")
        print(f"Prototxt: {prototxt_path} ({proto_size:,} bytes)")
        print(f"Caffemodel: {caffemodel_path} ({model_size:,} bytes)")
        
        # Expected caffemodel size is around 23MB
        if model_size < 20_000_000:
            print(f"WARNING: Caffemodel seems small ({model_size:,} bytes). Expected ~23MB")
            print("The file may be corrupted or incomplete.")
        
        # Compute MD5 hash for verification
        print("Computing file hash...")
        with open(caffemodel_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        print(f"Caffemodel MD5: {file_hash}")
        
        # Known good hash for mobilenet_iter_73000.caffemodel
        KNOWN_HASH = "994d30a8afaa9e754d17d2373b2d62a7"
        if file_hash == KNOWN_HASH:
            print("✓ Hash matches known good model!")
        else:
            print(f"⚠ Hash doesn't match expected: {KNOWN_HASH}")
            print("Model may be from a different source or corrupted.")
        
        try:
            # Try to load with better error handling
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            
            # Test if the network loaded properly by checking layer count
            layer_names = self.net.getLayerNames()
            print(f"✓ Network loaded successfully with {len(layer_names)} layers")
            
        except cv2.error as e:
            print("\n" + "="*70)
            print("ERROR: Failed to load model files!")
            print("="*70)
            print(f"\nOpenCV Error: {str(e)}")
            print("\nThis error typically means:")
            print("1. The prototxt and caffemodel don't match (different versions)")
            print("2. The caffemodel is corrupted")
            print("3. The files are incomplete")
            print("\nRECOMMENDED FIX:")
            print("Delete both files and re-download them together:")
            print("\ncd Model")
            print("rm deploy.prototxt MobileNetSSD_deploy.caffemodel")
            print("curl -L -o deploy.prototxt https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt")
            print("curl -L -o MobileNetSSD_deploy.caffemodel https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel")
            print("cd ..")
            print("\n" + "="*70)
            raise
        
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"]
        
        self.vehicle_classes = ["car", "bus", "motorbike", "bicycle"]
    
    def detect_lane_markings(self, image):
        """Detect lane markings using edge detection and Hough transform"""
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Define region of interest (bottom half of image for lane detection)
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(0, h), (0, h//2), (w, h//2), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, 
                                minLineLength=50, maxLineGap=150)
        
        if lines is None:
            print("No lane markings detected, using default zone")
            return None
        
        # Filter lines based on angle (looking for nearly vertical or angled lines)
        lane_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Filter for lane-like angles (30-90 degrees for left, -90 to -30 for right)
            if self.side == 'left' and 30 <= abs(angle) <= 90:
                lane_lines.append((x1, y1, x2, y2, angle))
            elif self.side == 'right' and 30 <= abs(angle) <= 90:
                lane_lines.append((x1, y1, x2, y2, angle))
        
        if not lane_lines:
            print("No suitable lane markings found, using default zone")
            return None
        
        # Find the most relevant lane marking based on side
        if self.side == 'left':
            # For left side, find leftmost lane marking
            relevant_line = min(lane_lines, key=lambda l: min(l[0], l[2]))
        else:
            # For right side, find rightmost lane marking
            relevant_line = max(lane_lines, key=lambda l: max(l[0], l[2]))
        
        x1, y1, x2, y2 = relevant_line[:4]
        
        # Find the topmost y-coordinate of the lane marking
        top_y = min(y1, y2)
        
        print(f"Lane marking detected at top_y={top_y}")
        return top_y
        
    def draw_detection_zone(self, img, lane_top_y=None):
        h, w = img.shape[:2]
        overlay = img.copy()
        
        zone_color = (255, 255, 0)
        
        # If lane marking detected, draw zone above it
        if lane_top_y is not None:
            # Zone from top of image to lane marking
            if self.side == 'left':
                cv2.rectangle(overlay, (10, 10), (w//2, lane_top_y), zone_color, 3)
            else:
                cv2.rectangle(overlay, (w//2, 10), (w-10, lane_top_y), zone_color, 3)
            
            # Draw lane marking indicator
            cv2.line(overlay, (10 if self.side == 'left' else w//2, lane_top_y), 
                    (w//2 if self.side == 'left' else w-10, lane_top_y), 
                    (0, 255, 0), 2)
            cv2.putText(overlay, "LANE", 
                       (20 if self.side == 'left' else w-100, lane_top_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Default zone (full area)
            cv2.rectangle(overlay, (10, 10), (w-10, h-10), zone_color, 3)
        
        label = f"{self.side.upper()} BLIND SPOT MONITOR"
        cv2.putText(overlay, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, zone_color, 2)
        
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        return img
    
    def detect_vehicles(self, image, lane_top_y=None):
        h, w = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            0.007843, 
            (300, 300), 
            127.5
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        vehicles = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.4:
                class_id = int(detections[0, 0, i, 1])
                class_name = self.classes[class_id]
                
                if class_name in self.vehicle_classes:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x_min, y_min, x_max, y_max = box.astype(int)
                    
                    # Check if vehicle is in blind spot zone (above lane marking)
                    if lane_top_y is not None:
                        # Vehicle center point
                        center_y = (y_min + y_max) // 2
                        
                        # Only consider vehicles above the lane marking
                        if center_y > lane_top_y:
                            continue
                        
                        # Check if vehicle is on correct side
                        center_x = (x_min + x_max) // 2
                        if self.side == 'left' and center_x > w//2:
                            continue
                        if self.side == 'right' and center_x < w//2:
                            continue
                    
                    vehicles.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': (x_min, y_min, x_max, y_max)
                    })
        
        return vehicles
    
    def process(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        result_img = image.copy()
        
        # Detect lane markings
        print("Detecting lane markings...")
        lane_top_y = self.detect_lane_markings(image)
        
        # Draw detection zone based on lane marking
        result_img = self.draw_detection_zone(result_img, lane_top_y)
        
        print(f"Scanning {self.side} blind spot...")
        vehicles = self.detect_vehicles(image, lane_top_y)
        
        alert = len(vehicles) > 0
        print(f"Found {len(vehicles)} vehicle(s) in {self.side} blind spot")
        
        for vehicle in vehicles:
            x_min, y_min, x_max, y_max = vehicle['box']
            label = vehicle['class']
            confidence = vehicle['confidence']
            
            color = (0, 0, 255)
            
            cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), color, 3)
            text = f"{label}: {confidence:.2f}"
            
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_img, (x_min, y_min - text_h - 10), 
                         (x_min + text_w, y_min), color, -1)
            cv2.putText(result_img, text, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            print(f"  WARNING: {label.upper()} detected (confidence: {confidence:.2f})")
        
        if alert:
            h, w = result_img.shape[:2]
            cv2.putText(result_img, "VEHICLE DETECTED", 
                       (w//2 - 200, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Save to results folder
        if output_path is None:
            output_path = f'results/output_{self.side}.jpg'
        elif not output_path.startswith('results/'):
            output_path = f'results/{output_path}'
        
        cv2.imwrite(output_path, result_img)
        print(f"✓ Processed image saved to {output_path}")
        
        return alert, len(vehicles)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        side = sys.argv[1]
        image_path = sys.argv[2]
    else:
        side = 'right'
        image_path = '../Pictures/right.jpeg'
    
    try:
        monitor = BlindSpotMonitor(side=side)
        
        output_name = f'output_{side}.jpg'
        alert, vehicle_count = monitor.process(image_path, output_name)
        
        print(f"\n{'='*50}")
        print(f"{side.upper()} BLIND SPOT DETECTION SUMMARY:")
        print(f"{'='*50}")
        print(f"Alert Status: {'WARNING' if alert else 'CLEAR'}")
        print(f"Vehicles Detected: {vehicle_count}")
        print(f"{'='*50}\n")
        
        sys.exit(1 if alert else 0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)