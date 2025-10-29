#!/usr/bin/env python3
"""
Real-time Blind Spot Monitor with Depth Estimation
Optimized for performance - computes depth only when needed
Can be imported and used as a module
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import time
import sys
from typing import Optional, Tuple, List, Dict


class DepthEstimator:
    """Handles depth estimation using MiDaS models"""
    
    def __init__(self, model_name="midas_v21_small", models_dir="Model", verbose=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self.verbose = verbose
        
        if self.verbose:
            print(f"🔧 Initializing Depth Estimator")
            print(f"   Device: {self.device}")
            print(f"   Model: {model_name}")
        
        # Load model
        model_path = self.models_dir / f"{self.model_name}.pt"
        if not model_path.exists():
            if self.verbose:
                print(f"❌ Model not found: {model_path}")
                print("   Using fallback depth estimation")
            self.model = None
        else:
            try:
                self.model = torch.load(model_path, map_location=self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                if self.verbose:
                    print("✅ Depth model loaded")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error loading model: {e}")
                self.model = None
        
        # Setup transforms
        input_size = (256, 256) if "small" in model_name else (384, 384)
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def estimate_depth(self, frame):
        """Estimate depth for a single frame"""
        if self.model is None:
            # Fallback: simple edge-based depth approximation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth = cv2.GaussianBlur(gray, (21, 21), 0)
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            return depth
        
        try:
            # Convert to PIL and preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                prediction = self.model(input_tensor)
                
                if isinstance(prediction, (list, tuple)):
                    prediction = prediction[0]
                
                depth_map = prediction.squeeze().cpu().numpy()
            
            # Resize to original size
            h, w = frame.shape[:2]
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, 
                                           cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return depth_normalized
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Depth estimation error: {e}")
            # Fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


class VehicleDistanceCalculator:
    """Calculate distances using depth maps"""
    
    def __init__(self, focal_length=700):
        self.focal_length = focal_length
        
        # Vehicle dimensions (meters)
        self.vehicle_dimensions = {
            'car': {'width': 1.8, 'height': 1.5},
            'bus': {'width': 2.5, 'height': 3.0},
            'motorbike': {'width': 0.8, 'height': 1.2},
            'bicycle': {'width': 0.6, 'height': 1.1}
        }
    
    def calculate_distance(self, depth_map, bbox, vehicle_class='car'):
        """Calculate distance using depth map"""
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure bbox is within bounds
        h, w = depth_map.shape[:2]
        x_min = max(0, min(x_min, w-1))
        y_min = max(0, min(y_min, h-1))
        x_max = max(0, min(x_max, w-1))
        y_max = max(0, min(y_max, h-1))
        
        if x_max <= x_min or y_max <= y_min:
            return None, 0.0
        
        # Extract depth ROI - focus on center of vehicle
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        roi_size = min((x_max - x_min) // 3, (y_max - y_min) // 3, 50)
        
        roi_x_min = max(0, center_x - roi_size)
        roi_x_max = min(w, center_x + roi_size)
        roi_y_min = max(0, center_y - roi_size)
        roi_y_max = min(h, center_y + roi_size)
        
        depth_roi = depth_map[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        
        if depth_roi.size == 0:
            return None, 0.0
        
        # Calculate median depth (more robust than mean)
        median_depth = np.median(depth_roi)
        std_depth = np.std(depth_roi)
        mean_depth = np.mean(depth_roi)
        
        # Normalize if needed
        if median_depth > 1.0:
            median_depth = median_depth / 255.0
        
        if median_depth < 0.01:
            median_depth = 0.01
        
        # Distance estimation (calibration constant)
        k = 5.0
        distance = k / median_depth
        
        # Confidence based on depth consistency
        confidence = 1.0 / (1.0 + std_depth / (mean_depth + 0.01))
        confidence = min(1.0, max(0.0, confidence))
        
        # Clamp reasonable distances (1-30 meters for blind spots)
        distance = max(1.0, min(30.0, distance))
        
        return distance, confidence


class BlindSpotMonitor:
    """Main blind spot monitoring system"""
    
    def __init__(self, side='left', model_name="midas_v21_small", models_dir="Model", 
                 detection_interval=5, depth_interval=3, verbose=True):
        self.side = side.lower()
        if self.side not in ['left', 'right']:
            raise ValueError("Side must be 'left' or 'right'")
        
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚗 Initializing Blind Spot Monitor - {side.upper()} Side")
            print(f"{'='*60}\n")
        
        # Optimization settings
        self.detection_interval = detection_interval
        self.depth_interval = depth_interval
        self.frame_count = 0
        self.detection_count = 0
        
        if self.verbose:
            print(f"⚡ Performance settings:")
            print(f"   Detection interval: every {detection_interval} frames")
            print(f"   Depth calculation: when vehicles detected\n")
        
        # Initialize components
        self.depth_estimator = DepthEstimator(model_name, models_dir, verbose)
        self.distance_calculator = VehicleDistanceCalculator()
        
        # Load vehicle detection model
        prototxt_path = Path(models_dir) / 'deploy.prototxt'
        caffemodel_path = Path(models_dir) / 'mobilenet_iter_73000.caffemodel'
        
        if not prototxt_path.exists() or not caffemodel_path.exists():
            raise FileNotFoundError(f"Vehicle detection model files not found in {models_dir}/")
        
        if self.verbose:
            print("📦 Loading vehicle detection model...")
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        if self.verbose:
            print("✅ Vehicle detection model loaded\n")
        
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"]
        
        self.vehicle_classes = ["car", "bus", "motorbike", "bicycle"]
        
        # Car edge cache
        self.car_edge_x = None
        self.car_edge_detected = False
        
        # Cache for vehicles
        self.cached_vehicles = []
        
        # Performance metrics
        self.frame_times = []
    
    def detect_car_edge(self, image):
        """Detect the edge of the vehicle (side mirror/window)"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Define search region
        if self.side == 'left':
            search_start = int(width * 0.50)
            search_end = int(width * 0.95)
        else:
            search_start = int(width * 0.05)
            search_end = int(width * 0.50)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Vertical edge detection
        vertical_kernel = np.ones((15, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Score each column
        vertical_scores = []
        for x in range(search_start, search_end):
            score = np.sum(vertical_edges[:, x] > 0)
            vertical_scores.append(score)
        
        vertical_scores = np.array(vertical_scores)
        
        # Smooth scores
        if len(vertical_scores) > 20:
            kernel_size = 20
            kernel = np.ones(kernel_size) / kernel_size
            smoothed_scores = np.convolve(vertical_scores, kernel, mode='same')
        else:
            smoothed_scores = vertical_scores
        
        # Find peak
        if len(smoothed_scores) > 0 and np.max(smoothed_scores) > 0:
            threshold = np.mean(smoothed_scores) + np.std(smoothed_scores) * 0.8
            peaks = np.where(smoothed_scores > threshold)[0]
            
            if len(peaks) > 0:
                if self.side == 'left':
                    best_peak = peaks[0]
                else:
                    best_peak = peaks[-1]
                
                car_edge_x = search_start + best_peak
            else:
                car_edge_x = search_start + np.argmax(smoothed_scores)
        else:
            # Default fallback
            car_edge_x = int(width * 0.65) if self.side == 'left' else int(width * 0.35)
        
        return car_edge_x
    
    def get_blindspot_zone(self, width, height):
        """Get blind spot zone coordinates"""
        if self.side == 'left':
            return (0, 0, self.car_edge_x, height)
        else:
            return (self.car_edge_x, 0, width, height)
    
    def is_vehicle_in_zone(self, bbox, zone):
        """Check if vehicle center is in blind spot zone"""
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        zone_x_min, zone_y_min, zone_x_max, zone_y_max = zone
        
        return (zone_x_min <= center_x <= zone_x_max and 
                zone_y_min <= center_y <= zone_y_max)
    
    def detect_vehicles(self, frame, zone):
        """Detect vehicles in frame"""
        h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
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
                    
                    if self.is_vehicle_in_zone((x_min, y_min, x_max, y_max), zone):
                        vehicles.append({
                            'class': class_name,
                            'confidence': confidence,
                            'box': (x_min, y_min, x_max, y_max)
                        })
        
        return vehicles
    
    def draw_overlay(self, frame, zone, vehicles):
        """Draw all overlays on frame"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw blind spot zone
        zone_x_min, zone_y_min, zone_x_max, zone_y_max = zone
        cv2.rectangle(overlay, 
                     (zone_x_min + 10, zone_y_min + 10), 
                     (zone_x_max - 10, zone_y_max - 10), 
                     (255, 255, 0), 3)
        
        # Draw car edge line
        cv2.line(overlay, (self.car_edge_x, 0), (self.car_edge_x, h), 
                (0, 255, 0), 4)
        
        # Labels
        def draw_label(text, pos, color=(255, 255, 0), bg_color=(0, 0, 0)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            
            cv2.rectangle(overlay, (x-5, y-th-5), (x+tw+5, y+5), bg_color, -1)
            cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)
        
        # Zone labels
        percentage = f"{self.car_edge_x/w*100:.1f}%"
        
        if self.side == 'left':
            draw_label(f"LEFT BLIND SPOT", (20, 40))
            draw_label(f"Car Edge: {percentage}", (self.car_edge_x + 10, 40), (0, 255, 0))
        else:
            draw_label(f"RIGHT BLIND SPOT", (self.car_edge_x + 10, 40))
            draw_label(f"Car Edge: {percentage}", (max(10, self.car_edge_x - 200), 40), 
                      (0, 255, 0))
        
        # Draw detected vehicles with distances
        for vehicle in vehicles:
            x_min, y_min, x_max, y_max = vehicle['box']
            label = vehicle['class']
            confidence = vehicle['confidence']
            distance = vehicle.get('distance')
            
            # Box color: red for alert
            color = (0, 0, 255)
            
            # Draw box
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Label with distance
            if distance is not None and distance > 0:
                text = f"{label.upper()}: {distance:.1f}m ({distance*3.28:.1f}ft)"
            else:
                text = f"{label.upper()}"
            
            # Draw text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position text above box
            text_x = x_min
            text_y = y_min - 10
            
            # Ensure text is visible
            if text_y < text_h + 10:
                text_y = y_min + text_h + 10
            
            # Background rectangle
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - text_h - 5), 
                         (text_x + text_w + 5, text_y + 5), 
                         color, -1)
            
            # Text
            cv2.putText(overlay, text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        # Alert banner
        if len(vehicles) > 0:
            alert_text = f"WARNING: {len(vehicles)} VEHICLE(S) IN BLIND SPOT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            (tw, th), _ = cv2.getTextSize(alert_text, font, font_scale, thickness)
            
            # Draw flashing banner
            banner_color = (0, 0, 255) if (self.frame_count // 10) % 2 == 0 else (0, 100, 255)
            cv2.rectangle(overlay, (w//2 - tw//2 - 15, h - 70), 
                         (w//2 + tw//2 + 15, h - 10), banner_color, -1)
            cv2.putText(overlay, alert_text, (w//2 - tw//2, h - 30),
                       font, font_scale, (255, 255, 255), thickness)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        return frame
    
    def process_frame(self, frame, draw_overlay=True):
        """
        Process single frame with optimized computation
        
        Args:
            frame: Input frame (BGR image)
            draw_overlay: Whether to draw visualization overlay (default: True)
        
        Returns:
            tuple: (output_frame, vehicles_list)
                - output_frame: Processed frame with or without overlay
                - vehicles_list: List of detected vehicles with distances
        """
        start_time = time.time()
        
        h, w = frame.shape[:2]
        self.frame_count += 1
        
        # Detect car edge (only once)
        if not self.car_edge_detected:
            self.car_edge_x = self.detect_car_edge(frame)
            self.car_edge_detected = True
        
        # Get blind spot zone
        zone = self.get_blindspot_zone(w, h)
        
        # Run detection only every N frames
        if self.frame_count % self.detection_interval == 0:
            self.cached_vehicles = self.detect_vehicles(frame, zone)
            
            # Calculate distances only when vehicles are detected
            if len(self.cached_vehicles) > 0:
                depth_map = self.depth_estimator.estimate_depth(frame)
                
                for vehicle in self.cached_vehicles:
                    bbox = vehicle['box']
                    vehicle_class = vehicle['class']
                    distance, confidence = self.distance_calculator.calculate_distance(
                        depth_map, bbox, vehicle_class
                    )
                    vehicle['distance'] = distance
                    vehicle['distance_confidence'] = confidence
                    
                    # Debug print
                    if self.verbose and distance:
                        print(f"  {vehicle_class}: {distance:.1f}m (confidence: {confidence:.2f})")
        
        # Draw overlays if requested
        if draw_overlay:
            output = self.draw_overlay(frame, zone, self.cached_vehicles)
        else:
            output = frame.copy()
        
        # Performance metrics
        process_time = time.time() - start_time
        self.frame_times.append(process_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        avg_time = np.mean(self.frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Draw FPS if overlay is enabled
        if draw_overlay:
            cv2.putText(output, f"FPS: {fps:.1f}", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output, self.cached_vehicles
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display the video during processing (default: True)
        
        Returns:
            bool: True if processing completed successfully
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            if self.verbose:
                print(f"❌ Could not open video: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.verbose:
            print(f"📹 Video: {width}x{height} @ {fps} FPS")
            print(f"📊 Total frames: {total_frames}")
            print(f"⏱️  Duration: {total_frames/fps:.1f}s\n")
        
        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if self.verbose:
                print(f"💾 Saving output to: {output_path}")
        
        if self.verbose:
            print("▶️  Processing... (Press 'q' to quit)\n")
        
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_num += 1
                
                # Process frame
                output, vehicles = self.process_frame(frame)
                
                # Write to output file
                if writer:
                    writer.write(output)
                
                # Display
                if display:
                    cv2.imshow(f'Blind Spot Monitor - {self.side.upper()}', output)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        if self.verbose:
                            print("\n⏹️  Stopped by user")
                        break
                
                # Status update
                if self.verbose and frame_num % 30 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Frame {frame_num}/{total_frames} ({progress:.1f}%) - "
                          f"Vehicles in zone: {len(vehicles)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            if self.verbose:
                print(f"\n✅ Processing complete!")
                print(f"   Processed {frame_num} frames")
                if output_path:
                    print(f"   Output saved to: {output_path}")
        
        return True
    
    def reset(self):
        """Reset the monitor state (useful for processing multiple videos)"""
        self.car_edge_detected = False
        self.car_edge_x = None
        self.cached_vehicles = []
        self.frame_count = 0
        self.detection_count = 0
        self.frame_times = []


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Blind Spot Monitor with Depth Estimation"
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--side", default="left", choices=["left", "right"],
                       help="Camera side (left or right)")
    parser.add_argument("--model", default="midas_v21_small",
                       help="Depth model name")
    parser.add_argument("--models-dir", default="Model",
                       help="Directory with depth models")
    parser.add_argument("--output", help="Output video file (optional)")
    parser.add_argument("--detection-interval", type=int, default=5,
                       help="Run detection every N frames (default: 5)")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return
    
    # Initialize monitor
    try:
        monitor = BlindSpotMonitor(
            side=args.side,
            model_name=args.model,
            models_dir=args.models_dir,
            detection_interval=args.detection_interval
        )
        
        # Process video
        monitor.process_video(args.video, args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("Real-time Blind Spot Monitor with Depth Estimation")
        print("="*60)
        print("\nUsage:")
        print("  python blindspot_realtime.py --video input.mp4 --side left")
        print("\nOptions:")
        print("  --video PATH            Input video file (required)")
        print("  --side left/right       Camera side (default: left)")
        print("  --model NAME            Depth model (default: midas_v21_small)")
        print("  --output PATH           Save output video (optional)")
        print("  --detection-interval N  Run detection every N frames (default: 5)")
        print("\nExample:")
        print("  python blindspot_realtime.py --video dash_cam.mp4 --side left --output result.mp4")
        print("  python blindspot_realtime.py --video cam.mp4 --detection-interval 3")
        print("\nControls:")
        print("  Press 'q' to quit during playback")
        print("\nPerformance Tips:")
        print("  - Increase detection-interval for faster processing")
        print("  - Use 'midas_v21_small' model for best performance")
        print("="*60 + "\n")
    else:
        main()