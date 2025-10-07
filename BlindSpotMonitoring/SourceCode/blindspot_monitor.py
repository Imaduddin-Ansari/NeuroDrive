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
            raise
        
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"]
        
        self.vehicle_classes = ["car", "bus", "motorbike", "bicycle"]
        
        # Cache for car edge position (so we don't recalculate for each frame)
        self.car_edge_x = None
    
    def detect_car_edge(self, image, debug=False):
        """
        Detects the edge of your own car in the blind spot camera image.
        Returns the x-coordinate of the car edge.
        """
        height, width = image.shape[:2]
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define search region based on camera side
        if self.side == 'left':
            # Left camera - car is on right side
            search_start = int(width * 0.50)
            search_end = int(width * 0.95)
        else:
            # Right camera - car is on left side
            search_start = int(width * 0.05)
            search_end = int(width * 0.50)
        
        print(f"Detecting car edge from {search_start} to {search_end} pixels...")
        
        # Method 1: Edge Detection - Find vertical structures (window frame/mirror)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for strong vertical edges
        vertical_kernel = np.ones((15, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Count vertical edge strength for each column
        vertical_scores = []
        for x in range(search_start, search_end):
            column = vertical_edges[:, x]
            score = np.sum(column > 0)
            vertical_scores.append(score)
        
        vertical_scores = np.array(vertical_scores)
        
        # Smooth scores
        if len(vertical_scores) > 20:
            kernel_size = min(20, len(vertical_scores) // 5)
            kernel = np.ones(kernel_size) / kernel_size
            smoothed_scores = np.convolve(vertical_scores, kernel, mode='same')
        else:
            smoothed_scores = vertical_scores
        
        # Method 2: Brightness discontinuity
        brightness = np.mean(gray, axis=0)
        brightness_gradient = np.abs(np.gradient(brightness))
        
        if len(brightness_gradient) > 30:
            brightness_gradient = np.convolve(brightness_gradient, np.ones(30)/30, mode='same')
        
        # Method 3: Texture analysis
        texture_scores = []
        window_size = 20
        
        for x in range(search_start, search_end):
            if x < window_size or x > width - window_size:
                texture_scores.append(0)
                continue
            
            patch = gray[:, max(0, x-window_size):min(width, x+window_size)]
            texture = np.std(patch)
            texture_scores.append(texture)
        
        texture_scores = np.array(texture_scores)
        texture_gradient = np.abs(np.gradient(texture_scores))
        
        if len(texture_gradient) > 20:
            texture_gradient = np.convolve(texture_gradient, np.ones(20)/20, mode='same')
        
        # Normalize and combine scores
        def normalize(arr):
            if len(arr) == 0:
                return arr
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if arr_max - arr_min == 0:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)
        
        norm_vertical = normalize(smoothed_scores)
        norm_brightness = normalize(brightness_gradient[search_start:search_end])
        norm_texture = normalize(texture_gradient)
        
        # Combined score with weights
        combined_score = (norm_vertical * 2.0 +
                         norm_brightness * 1.5 +
                         norm_texture * 1.0)
        
        # Find peaks in combined score
        car_edge_x = None
        
        if len(combined_score) > 0:
            threshold = np.mean(combined_score) + np.std(combined_score) * 0.8
            peaks = np.where(combined_score > threshold)[0]
            
            if len(peaks) > 0:
                # For left camera: take first (leftmost) strong peak
                # For right camera: take last (rightmost) strong peak
                if self.side == 'left':
                    best_peak = peaks[0]
                else:
                    best_peak = peaks[-1]
                
                car_edge_x = search_start + best_peak
                confidence = combined_score[best_peak] / np.max(combined_score) * 100
                print(f"✓ Car edge detected at x={car_edge_x} ({car_edge_x/width*100:.1f}%) with {confidence:.1f}% confidence")
            
            # If no strong peak found, use maximum score position
            if car_edge_x is None:
                best_peak = np.argmax(combined_score)
                car_edge_x = search_start + best_peak
                print(f"Using maximum score position at x={car_edge_x} ({car_edge_x/width*100:.1f}%)")
        
        # Fallback with side-aware default
        if car_edge_x is None:
            if self.side == 'left':
                car_edge_x = int(width * 0.65)
            else:
                car_edge_x = int(width * 0.35)
            print(f"⚠ Using default position: {car_edge_x} ({car_edge_x/width*100:.1f}%)")
        
        return car_edge_x
    
    def get_blindspot_zone(self, image_width, image_height, image=None):
        """
        Dynamically detect the blind spot zone based on car edge detection.
        Returns: (x_min, y_min, x_max, y_max) of the zone
        """
        # If car edge not yet detected and image provided, detect it
        if self.car_edge_x is None and image is not None:
            self.car_edge_x = self.detect_car_edge(image)
        
        # If still None (no image provided), use default
        if self.car_edge_x is None:
            if self.side == 'left':
                self.car_edge_x = int(image_width * 0.65)
            else:
                self.car_edge_x = int(image_width * 0.35)
            print(f"Using default car edge: {self.car_edge_x}")
        
        # Define zone based on camera side and detected car edge
        if self.side == 'left':
            # Left camera: blind spot is from left edge to car edge
            return (0, 0, self.car_edge_x, image_height)
        else:
            # Right camera: blind spot is from car edge to right edge
            return (self.car_edge_x, 0, image_width, image_height)
    
    def is_vehicle_in_zone(self, vehicle_box, zone):
        """
        Check if vehicle center point is within the blind spot zone
        """
        x_min, y_min, x_max, y_max = vehicle_box
        zone_x_min, zone_y_min, zone_x_max, zone_y_max = zone
        
        # Calculate vehicle center
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Check if center is within zone boundaries
        return (zone_x_min <= center_x <= zone_x_max and 
                zone_y_min <= center_y <= zone_y_max)
    
    def draw_detection_zone(self, img, zone, car_edge_x):
        """Draw the dynamically detected blind spot zone"""
        h, w = img.shape[:2]
        overlay = img.copy()
        
        zone_color = (255, 255, 0)  # Yellow
        x_min, y_min, x_max, y_max = zone
        
        # Draw zone rectangle
        padding = 10
        cv2.rectangle(overlay, 
                     (x_min + padding, y_min + padding), 
                     (x_max - padding, y_max - padding), 
                     zone_color, 3)
        
        # Draw car edge line in green
        cv2.line(overlay, (car_edge_x, 0), (car_edge_x, h), (0, 255, 0), 4)
        
        # Add labels with background
        def draw_label(text, pos, color=zone_color):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            
            # Background
            cv2.rectangle(overlay, (x-5, y-th-5), (x+tw+5, y+5), (0, 0, 0), -1)
            # Text
            cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)
        
        # Labels based on side
        percentage = f"{car_edge_x/w*100:.1f}%"
        
        if self.side == 'left':
            draw_label(f"LEFT BLIND SPOT ZONE", (20, 40))
            draw_label(f"Car Edge at {percentage}", (car_edge_x + 10, 40), (0, 255, 0))
            draw_label("YOUR CAR", (car_edge_x + 10, h - 30), (255, 255, 255))
        else:
            draw_label("YOUR CAR", (20, 40), (255, 255, 255))
            draw_label(f"Car Edge at {percentage}", (max(10, car_edge_x - 200), 40), (0, 255, 0))
            draw_label(f"RIGHT BLIND SPOT ZONE", (car_edge_x + 10, 40))
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        return img
    
    def detect_vehicles(self, image, zone):
        """Detect vehicles and filter only those in the blind spot zone"""
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
                    
                    # Only include vehicles IN the blind spot zone
                    if self.is_vehicle_in_zone((x_min, y_min, x_max, y_max), zone):
                        vehicles.append({
                            'class': class_name,
                            'confidence': confidence,
                            'box': (x_min, y_min, x_max, y_max)
                        })
                        center_x = (x_min + x_max) // 2
                        print(f"  ✓ {class_name} IN zone at x={center_x}")
                    else:
                        center_x = (x_min + x_max) // 2
                        print(f"  ✗ {class_name} OUTSIDE zone at x={center_x} - IGNORED")
        
        return vehicles
    
    def process(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        h, w = image.shape[:2]
        result_img = image.copy()
        
        # Detect car edge first (only once per image)
        self.car_edge_x = self.detect_car_edge(image)
        
        # Get blind spot zone based on detected car edge
        zone = self.get_blindspot_zone(w, h, image)
        
        print(f"\nBlind spot zone: x={zone[0]}-{zone[2]}, y={zone[1]}-{zone[3]}")
        print(f"Car edge at: {self.car_edge_x} ({self.car_edge_x/w*100:.1f}%)")
        print(f"Scanning {self.side} blind spot...")
        
        # Draw detection zone with car edge line
        result_img = self.draw_detection_zone(result_img, zone, self.car_edge_x)
        
        # Detect vehicles in the zone
        vehicles = self.detect_vehicles(image, zone)
        
        alert = len(vehicles) > 0
        print(f"\nFound {len(vehicles)} vehicle(s) IN {self.side} blind spot zone")
        
        # Draw detected vehicles
        for vehicle in vehicles:
            x_min, y_min, x_max, y_max = vehicle['box']
            label = vehicle['class']
            confidence = vehicle['confidence']
            
            color = (0, 0, 255)  # Red for vehicles in blind spot
            
            cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), color, 3)
            text = f"{label}: {confidence:.2f}"
            
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_img, (x_min, y_min - text_h - 10), 
                         (x_min + text_w, y_min), color, -1)
            cv2.putText(result_img, text, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            print(f"  ⚠ WARNING: {label.upper()} detected (confidence: {confidence:.2f})")
        
        # Add alert banner if vehicles detected
        if alert:
            cv2.putText(result_img, "VEHICLE DETECTED IN BLIND SPOT", 
                       (w//2 - 250, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Save to results folder
        if output_path is None:
            output_path = f'results/output_{self.side}.jpg'
        elif not output_path.startswith('results/'):
            output_path = f'results/{output_path}'
        
        cv2.imwrite(output_path, result_img)
        print(f"\n✓ Processed image saved to {output_path}")
        
        # Reset car edge for next image (if processing multiple images)
        # Comment this line if processing video frames
        # self.car_edge_x = None
        
        return alert, len(vehicles)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        side = sys.argv[1]
        image_path = sys.argv[2]
    else:
        side = 'left'
        image_path = '../Pictures/Testing.jpeg'
    
    try:
        monitor = BlindSpotMonitor(side=side)
        
        output_name = f'output_{side}.jpg'
        alert, vehicle_count = monitor.process(image_path, output_name)
        
        print(f"\n{'='*60}")
        print(f"{side.upper()} BLIND SPOT DETECTION SUMMARY:")
        print(f"{'='*60}")
        print(f"Alert Status: {'⚠ WARNING' if alert else '✓ CLEAR'}")
        print(f"Vehicles Detected: {vehicle_count}")
        print(f"Car Edge Position: {monitor.car_edge_x} pixels")
        print(f"{'='*60}\n")
        
        sys.exit(1 if alert else 0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)