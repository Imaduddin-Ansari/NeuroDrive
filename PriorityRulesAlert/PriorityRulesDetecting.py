"""
Traffic Rules Detection System using YOLOv8 COCO Classes
Maps detected objects to traffic rules from knowledge base
Enhanced with alert persistence and duplicate prevention
"""

from ultralytics import YOLO
import cv2
import json
import numpy as np
import time

class TrafficRuleDetector:
    def __init__(self, knowledge_base_path='traffic_rules_knowledge_base.json'):
        """Initialize YOLOv8 and load traffic rules knowledge base"""
        
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')  # Automatically downloads if not present
        
        # Load traffic rules knowledge base
        with open(knowledge_base_path, 'r') as f:
            kb = json.load(f)
            self.rules = {rule['rule_id']: rule for rule in kb['rules']}
        
        # Enhanced COCO class to rule mapping
        self.coco_to_rules = {
            # Pedestrians - only for pedestrian crossing detection
            'person': ['PED_001'],  # Removed ZONE_001 - school zones need signs!
            
            # Cyclists and motorcycles
            'bicycle': ['CYC_001'],
            'motorcycle': ['MOTO_001'],
            
            # Vehicles
            'car': ['LANE_001'],
            'bus': ['HV_001', 'BUS_001'],
            'truck': ['HV_001'],
            
            # Traffic control
            'traffic light': ['LIGHT_001', 'LIGHT_002'],
            'stop sign': ['SIGN_001'],
            
            # Animals - NEW! Triggers animal crossing rule
            'cow': ['ANIMAL_001'],
            'sheep': ['ANIMAL_001'],
            'horse': ['ANIMAL_001'],
            'dog': ['ANIMAL_001'],
            'cat': ['ANIMAL_001'],
            'elephant': ['ANIMAL_001'],
            'bear': ['ANIMAL_001'],
            'zebra': ['ANIMAL_001'],
            'giraffe': ['ANIMAL_001'],
        }
        
        # Alert persistence system - keeps alerts visible for 2-3 seconds
        self.alert_persistence_time = 2.5  # seconds
        self.active_alerts = {}  # {rule_id: {'rule': rule_obj, 'expire_time': timestamp}}
        
        print("✓ System initialized successfully")
        print(f"✓ Loaded {len(self.rules)} traffic rules")
        print(f"✓ Monitoring {len(self.coco_to_rules)} object types")
        print(f"✓ Alert persistence: {self.alert_persistence_time}s")
    
    def detect_objects(self, frame, conf_threshold=0.5):
        """
        Detect objects in frame using YOLOv8
        
        Args:
            frame: OpenCV image (BGR format)
            conf_threshold: Minimum confidence threshold (0-1)
        
        Returns:
            List of detected objects with metadata
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = results.names[class_id]
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox,
                'center': [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ]
            })
        
        return detections
    
    def get_triggered_rules(self, detections):
        """
        Map detected objects to applicable traffic rules with persistence
        
        Args:
            detections: List of detected objects
        
        Returns:
            List of triggered rules with priorities
        """
        current_time = time.time()
        newly_triggered_rule_ids = set()
        
        # Process current detections
        for det in detections:
            class_name = det['class']
            
            # Check if this class triggers any rules
            if class_name in self.coco_to_rules:
                rule_ids = self.coco_to_rules[class_name]
                
                for rule_id in rule_ids:
                    newly_triggered_rule_ids.add(rule_id)
                    
                    # Add or update alert with new expiry time
                    if rule_id not in self.active_alerts:
                        # New alert - add it
                        rule = self.rules[rule_id].copy()
                        rule['detected_object'] = det
                        self.active_alerts[rule_id] = {
                            'rule': rule,
                            'expire_time': current_time + self.alert_persistence_time,
                            'last_seen': current_time
                        }
                    else:
                        # Alert already exists - extend its lifetime
                        self.active_alerts[rule_id]['expire_time'] = current_time + self.alert_persistence_time
                        self.active_alerts[rule_id]['last_seen'] = current_time
                        self.active_alerts[rule_id]['rule']['detected_object'] = det
        
        # Remove expired alerts
        expired_ids = [
            rule_id for rule_id, alert in self.active_alerts.items()
            if current_time > alert['expire_time']
        ]
        for rule_id in expired_ids:
            del self.active_alerts[rule_id]
        
        # Convert active alerts to list of rules
        triggered_rules = [alert['rule'] for alert in self.active_alerts.values()]
        
        # Sort by priority (critical > high > medium > low)
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        triggered_rules.sort(key=lambda r: priority_order.get(r['priority'], 4))
        
        return triggered_rules
    
    def draw_detections(self, frame, detections, triggered_rules):
        """
        Draw bounding boxes and rule information on frame
        
        Args:
            frame: OpenCV image
            detections: List of detected objects
            triggered_rules: List of triggered rules
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Define colors for different priorities
        priority_colors = {
            'critical': (0, 0, 255),    # Red
            'high': (0, 165, 255),      # Orange
            'medium': (0, 255, 255),    # Yellow
            'low': (0, 255, 0)          # Green
        }
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            conf = det['confidence']
            
            # Find highest priority rule for this detection
            priority = 'low'
            for rule in triggered_rules:
                if rule.get('detected_object', {}).get('class') == class_name:
                    priority = rule['priority']
                    break
            
            color = priority_colors.get(priority, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def display_active_rules(self, frame, triggered_rules):
        """
        Display active rules on the frame with persistence
        
        Args:
            frame: OpenCV image
            triggered_rules: List of triggered rules (includes persistent alerts)
        
        Returns:
            Frame with rule overlay
        """
        if not triggered_rules:
            return frame
            
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Calculate panel height based on number of rules
        num_rules = min(len(triggered_rules), 5)
        panel_height = 60 + (num_rules * 30)
        panel_height = min(panel_height, h // 3)
        
        # Create semi-transparent background for rules panel
        cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Display header
        y_offset = 40
        cv2.putText(frame, "🚨 ACTIVE TRAFFIC RULES", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display top 5 rules with priority colors
        for i, rule in enumerate(triggered_rules[:5]):
            rule_text = f"{rule['description'][:55]}"
            priority = rule['priority'].upper()
            
            # Color code by priority
            if priority == 'CRITICAL':
                color = (0, 0, 255)      # Red
                prefix = "🔴"
            elif priority == 'HIGH':
                color = (0, 165, 255)    # Orange
                prefix = "🟠"
            elif priority == 'MEDIUM':
                color = (0, 255, 255)    # Yellow
                prefix = "🟡"
            else:
                color = (0, 255, 0)      # Green
                prefix = "🟢"
            
            text = f"{prefix} [{priority}] {rule_text}"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y_offset += 30
        
        # Show alert count
        if len(triggered_rules) > 5:
            cv2.putText(frame, f"+ {len(triggered_rules) - 5} more...", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        return frame
    
    def process_frame(self, frame, show_rules=True):
        """
        Complete processing pipeline for a single frame
        
        Args:
            frame: OpenCV image
            show_rules: Whether to display active rules overlay
        
        Returns:
            Annotated frame, detections, triggered rules
        """
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Get triggered rules (with persistence)
        triggered_rules = self.get_triggered_rules(detections)
        
        # Draw detections
        annotated = self.draw_detections(frame, detections, triggered_rules)
        
        # Display active rules if requested
        if show_rules:
            annotated = self.display_active_rules(annotated, triggered_rules)
        
        return annotated, detections, triggered_rules
    
    def process_video(self, video_path, output_path=None, show_live=True):
        """
        Process video file or webcam feed
        
        Args:
            video_path: Path to video file or 0 for webcam
            output_path: Path to save output video (optional)
            show_live: Whether to display live video
        """
        cap = cv2.VideoCapture(video_path if video_path != 'camera' else 0)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info: {w}x{h} @ {fps}fps, {total_frames} frames")
        
        if output_path:
            out = cv2.VideoWriter(output_path, 
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))
        
        frame_count = 0
        start_time = time.time()
        print("\nProcessing video... Press 'q' to quit\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated, detections, rules = self.process_frame(frame)
            
            # Add statistics overlay
            stats_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(detections)} | Active Rules: {len(rules)}"
            cv2.putText(annotated, stats_text, 
                       (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Save to output
            if output_path:
                out.write(annotated)
            
            # Display live
            if show_live:
                cv2.imshow('Traffic Rules Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  User interrupted processing")
                    break
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} frames | "
                      f"FPS: {fps_actual:.1f} | "
                      f"Active alerts: {len(self.active_alerts)}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"✓ Processed {frame_count} frames in {elapsed:.1f}s")
        print(f"✓ Average FPS: {frame_count/elapsed:.1f}")
        if output_path:
            print(f"✓ Saved to: {output_path}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = TrafficRuleDetector('traffic_rules_knowledge_base.json')
    
    # Process video
    detector.process_video('TestPriority.mov', 
                          output_path='output.mp4',
                          show_live=True)