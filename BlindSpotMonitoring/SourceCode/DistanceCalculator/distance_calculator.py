#!/usr/bin/env python3
"""
Calculate distance to detected vehicles using depth maps
Takes detection boxes and depth image as input
"""
import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class VehicleDistanceCalculator:
    def __init__(self, focal_length=700, sensor_width=6.17):
        """
        Initialize distance calculator
        
        Args:
            focal_length: Camera focal length in pixels (default 700)
            sensor_width: Camera sensor width in mm (default 6.17 for typical cameras)
        """
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        
        # Average real-world vehicle dimensions (meters)
        self.vehicle_dimensions = {
            'car': {'width': 1.8, 'height': 1.5, 'length': 4.5},
            'bus': {'width': 2.5, 'height': 3.0, 'length': 12.0},
            'motorbike': {'width': 0.8, 'height': 1.2, 'length': 2.2},
            'bicycle': {'width': 0.6, 'height': 1.1, 'length': 1.8}
        }
    
    def calculate_distance_from_depth(self, depth_map, bbox):
        """
        Calculate distance using depth map values in the bounding box
        
        Args:
            depth_map: Depth map (numpy array, 0-255 normalized)
            bbox: Bounding box tuple (x_min, y_min, x_max, y_max)
        
        Returns:
            distance_meters: Estimated distance in meters
            confidence: Confidence score (0-1)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure bbox is within image bounds
        h, w = depth_map.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Extract depth region of the vehicle
        depth_roi = depth_map[y_min:y_max, x_min:x_max]
        
        if depth_roi.size == 0:
            return None, 0.0
        
        # Calculate statistics
        median_depth = np.median(depth_roi)
        mean_depth = np.mean(depth_roi)
        std_depth = np.std(depth_roi)
        
        # MIDAS outputs inverse depth (higher values = closer)
        # Normalize to 0-1 range if needed
        if depth_roi.max() > 1.0:
            median_depth = median_depth / 255.0
            mean_depth = mean_depth / 255.0
        
        # Avoid division by zero
        if median_depth < 0.01:
            median_depth = 0.01
        
        # Distance estimation using calibrated formula
        # This is empirically derived - adjust based on your setup
        # Formula: distance = k / depth_value
        k = 3.5  # Calibration constant (adjust based on testing)
        
        distance = k / median_depth
        
        # Confidence based on depth consistency in ROI
        # Lower std = higher confidence
        confidence = 1.0 / (1.0 + std_depth / (mean_depth + 0.01))
        confidence = min(1.0, max(0.0, confidence))
        
        return distance, confidence
    
    def calculate_distance_from_size(self, bbox, vehicle_class, image_width):
        """
        Calculate distance using similar triangles (vehicle width method)
        
        Args:
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            vehicle_class: Vehicle type ('car', 'bus', etc.)
            image_width: Width of the image in pixels
        
        Returns:
            distance_meters: Estimated distance
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Vehicle width in pixels
        vehicle_width_pixels = x_max - x_min
        
        if vehicle_width_pixels <= 0:
            return None
        
        # Real-world vehicle width
        real_width = self.vehicle_dimensions.get(
            vehicle_class, 
            self.vehicle_dimensions['car']
        )['width']
        
        # Similar triangles formula: 
        # distance = (real_width * focal_length) / pixel_width
        distance = (real_width * self.focal_length) / vehicle_width_pixels
        
        return distance
    
    def calculate_combined_distance(self, depth_map, bbox, vehicle_class, image_width):
        """
        Combine depth-based and size-based distance estimates
        
        Returns:
            distance: Final distance estimate in meters
            method_used: Which method was used
            details: Dictionary with additional information
        """
        # Method 1: Depth-based
        dist_depth, confidence = self.calculate_distance_from_depth(depth_map, bbox)
        
        # Method 2: Size-based
        dist_size = self.calculate_distance_from_size(bbox, vehicle_class, image_width)
        
        details = {
            'depth_distance': dist_depth,
            'size_distance': dist_size,
            'depth_confidence': confidence,
            'vehicle_class': vehicle_class
        }
        
        # Decision logic: prioritize depth if confidence is high
        if dist_depth is not None and confidence > 0.6:
            final_distance = dist_depth
            method_used = 'depth'
        elif dist_size is not None:
            final_distance = dist_size
            method_used = 'size'
        elif dist_depth is not None:
            final_distance = dist_depth
            method_used = 'depth_low_conf'
        else:
            final_distance = None
            method_used = 'none'
        
        # If both available, use weighted average
        if dist_depth is not None and dist_size is not None and confidence > 0.3:
            weight_depth = confidence
            weight_size = 0.3
            final_distance = (dist_depth * weight_depth + dist_size * weight_size) / (weight_depth + weight_size)
            method_used = 'combined'
        
        details['final_distance'] = final_distance
        details['method_used'] = method_used
        
        return final_distance, method_used, details


def process_detection_image(original_image_path, depth_image_path, detection_data=None):
    """
    Process images with detections and calculate distances
    
    Args:
        original_image_path: Path to original image
        depth_image_path: Path to depth map image
        detection_data: Dictionary with detection info or None to parse from image
    
    Returns:
        results: List of detection results with distances
    """
    # Load images
    original_img = cv2.imread(str(original_image_path))
    depth_img = cv2.imread(str(depth_image_path), cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        print(f"Error: Could not load original image {original_image_path}")
        return None
    
    if depth_img is None:
        print(f"Error: Could not load depth image {depth_image_path}")
        return None
    
    h, w = original_img.shape[:2]
    
    # Initialize calculator
    calculator = VehicleDistanceCalculator()
    
    # If detection_data not provided, try to extract from image
    # (This assumes detection boxes are drawn with specific colors)
    if detection_data is None:
        detection_data = extract_detections_from_image(original_img)
    
    results = []
    
    for detection in detection_data:
        bbox = detection['box']
        vehicle_class = detection.get('class', 'car')
        confidence = detection.get('confidence', 0.0)
        
        # Calculate distance
        distance, method, details = calculator.calculate_combined_distance(
            depth_img, bbox, vehicle_class, w
        )
        
        result = {
            'vehicle_class': vehicle_class,
            'detection_confidence': confidence,
            'bounding_box': bbox,
            'distance_meters': distance,
            'distance_method': method,
            'details': details
        }
        
        results.append(result)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Vehicle: {vehicle_class.upper()}")
        print(f"Detection Confidence: {confidence:.2f}")
        print(f"Bounding Box: {bbox}")
        if distance is not None:
            print(f"Estimated Distance: {distance:.2f} meters ({distance * 3.28084:.2f} feet)")
            print(f"Method Used: {method}")
            print(f"Depth-based: {details['depth_distance']:.2f}m" if details['depth_distance'] else "Depth-based: N/A")
            print(f"Size-based: {details['size_distance']:.2f}m" if details['size_distance'] else "Size-based: N/A")
        else:
            print("Distance: Could not calculate")
        print(f"{'='*60}")
    
    # Visualize results
    visualize_distances(original_img, depth_img, results)
    
    return results


def extract_detections_from_image(img):
    """
    Extract detection boxes from annotated image
    Looks for red boxes (BGR: 0,0,255) which indicate detections
    """
    detections = []
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'box': (x, y, x+w, y+h),
                'class': 'car',  # Default to car
                'confidence': 0.8
            })
    
    return detections


def visualize_distances(original_img, depth_img, results):
    """
    Create visualization with distance annotations
    """
    output = original_img.copy()
    
    for result in results:
        bbox = result['bounding_box']
        distance = result['distance_meters']
        vehicle_class = result['vehicle_class']
        
        x_min, y_min, x_max, y_max = bbox
        
        # Draw box
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add distance label
        if distance is not None:
            label = f"{vehicle_class}: {distance:.1f}m"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output, (x_min, y_min - text_h - 10), 
                         (x_min + text_w + 10, y_min), (0, 255, 0), -1)
            cv2.putText(output, label, (x_min + 5, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save output
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "distance_output.jpg"
    cv2.imwrite(str(output_path), output)
    
    print(f"\n✓ Output saved to: {output_path}")
    
    # Also save a combined view
    depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_PLASMA)
    combined = np.hstack([output, depth_colored])
    combined_path = output_dir / "distance_combined.jpg"
    cv2.imwrite(str(combined_path), combined)
    
    print(f"✓ Combined view saved to: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate vehicle distances from depth maps")
    parser.add_argument("--original", required=True, help="Path to original/detection image")
    parser.add_argument("--depth", required=True, help="Path to depth map image")
    parser.add_argument("--detections", help="JSON file with detection data (optional)")
    
    args = parser.parse_args()
    
    # Load detection data if provided
    detection_data = None
    if args.detections:
        with open(args.detections, 'r') as f:
            detection_data = json.load(f)
    
    # Process
    results = process_detection_image(args.original, args.depth, detection_data)
    
    if results:
        print(f"\n✓ Processed {len(results)} detections")


if __name__ == "__main__":
    # Example usage without command line args
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python3 distance_calculator.py --original detection.jpg --depth depth_map.png")
        print("\nOr provide detection data:")
        print("  python distance_calculator.py --original image.jpg --depth depth.png --detections data.json")
        
        # Try to find images automatically
        results_dir = Path("results")
        if results_dir.exists():
            original_imgs = list(results_dir.glob("output_*.jpg"))
            if original_imgs:
                print(f"\nFound detection images: {[str(p) for p in original_imgs]}")
    else:
        main()