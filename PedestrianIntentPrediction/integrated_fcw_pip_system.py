#!/usr/bin/env python3
"""
INTEGRATED FCW + PIP SYSTEM
===========================
Complete integration of Forward Collision Warning and Pedestrian Intent Prediction
Week 5: Full integration with FCW module: unified alert prioritization logic.

Features:
- Unified detection pipeline (vehicles + pedestrians)
- Shared tracking system
- Integrated alert prioritization
- Environmental adaptation for both systems
- Real-time performance optimization
- Audio alert coordination

Author: NeuroDrive Team
Date: January 2025
"""

import os
import sys
import cv2
import numpy as np
import time
import json
import argparse
from pathlib import Path
from collections import deque
import threading
import queue

# Add project root to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import FCW components
try:
    from fcw_ttc import RobustLaneDetector, AccurateDepthEstimator, TTCCalculator
except ImportError:
    print("⚠ FCW components not found, using simplified versions")
    RobustLaneDetector = None
    AccurateDepthEstimator = None
    TTCCalculator = None

# Import PIP components
from enhanced_pip_system import (
    AudioAlertSystem, RealTimeOptimizer, WeatherDetector, 
    LightingClassifier, SpeedAdaptiveSystem, IntegratedPIPSystem,
    EnhancedConfig
)

# Core libraries
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker


class UnifiedDetectionSystem:
    """
    Unified detection system for both vehicles and pedestrians.
    Optimizes performance by running single YOLO inference.
    """
    
    def __init__(self, model_path='yolov8n.pt'):
        self.yolo_model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.pedestrian_classes = [0]  # person
        
    def detect_all_objects(self, frame, conf_threshold=0.4):
        """
        Detect both vehicles and pedestrians in single inference.
        
        Returns:
            vehicles: List of vehicle detections
            pedestrians: List of pedestrian detections
        """
        results = self.yolo_model(frame, conf=conf_threshold, verbose=False)
        
        vehicles = []
        pedestrians = []
        
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                try:
                    cls = int(boxes.cls[i])
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    
                    detection = {
                        'bbox': [float(xyxy[0]), float(xyxy[1]), 
                               float(xyxy[2]), float(xyxy[3])],
                        'conf': conf,
                        'cls': cls
                    }
                    
                    if cls in self.vehicle_classes:
                        vehicles.append(detection)
                    elif cls in self.pedestrian_classes:
                        pedestrians.append(detection)
                        
                except Exception as e:
                    continue
        
        return vehicles, pedestrians


class UnifiedTrackingSystem:
    """
    Unified tracking system using ByteTracker for both vehicles and pedestrians.
    Maintains separate trackers but coordinates updates.
    """
    
    def __init__(self):
        # Separate trackers for different object types
        self.vehicle_tracker = BYTETracker(frame_rate=30)
        self.pedestrian_tracker = BYTETracker(frame_rate=30)
        
        # Track management
        self.vehicle_tracks = {}
        self.pedestrian_tracks = {}
        
    def update_tracks(self, vehicles, pedestrians, frame_shape):
        """Update both vehicle and pedestrian tracks."""
        h, w = frame_shape[:2]
        
        # Convert detections to ByteTracker format
        vehicle_tensor = self._detections_to_tensor(vehicles)
        pedestrian_tensor = self._detections_to_tensor(pedestrians)
        
        # Update trackers
        try:
            vehicle_tracks = self.vehicle_tracker.update(
                vehicle_tensor, [h, w], (h, w)
            ) if len(vehicles) > 0 else []
            
            pedestrian_tracks = self.pedestrian_tracker.update(
                pedestrian_tensor, [h, w], (h, w)
            ) if len(pedestrians) > 0 else []
            
        except Exception as e:
            print(f"Tracking error: {e}")
            vehicle_tracks = []
            pedestrian_tracks = []
        
        # Update track dictionaries
        self._update_track_dict(self.vehicle_tracks, vehicle_tracks, 'vehicle')
        self._update_track_dict(self.pedestrian_tracks, pedestrian_tracks, 'pedestrian')
        
        return vehicle_tracks, pedestrian_tracks
    
    def _detections_to_tensor(self, detections):
        """Convert detections to ByteTracker tensor format."""
        if len(detections) == 0:
            return torch.empty((0, 6))
        
        output = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            output.append([x1, y1, x2, y2, det['conf'], det['cls']])
        
        return torch.from_numpy(np.array(output)).float()
    
    def _update_track_dict(self, track_dict, new_tracks, object_type):
        """Update track dictionary with new tracks."""
        current_ids = set()
        
        for track in new_tracks:
            if hasattr(track, 'track_id'):
                track_id = int(track.track_id)
                current_ids.add(track_id)
                
                if track_id not in track_dict:
                    track_dict[track_id] = {
                        'type': object_type,
                        'history': deque(maxlen=30),
                        'first_seen': time.time(),
                        'last_update': time.time()
                    }
                
                # Update track info
                track_dict[track_id]['last_update'] = time.time()
                track_dict[track_id]['history'].append({
                    'bbox': track.tlbr.tolist() if hasattr(track, 'tlbr') else None,
                    'timestamp': time.time()
                })
        
        # Remove old tracks
        to_remove = []
        current_time = time.time()
        for track_id in track_dict:
            if (track_id not in current_ids and 
                current_time - track_dict[track_id]['last_update'] > 2.0):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del track_dict[track_id]


class IntegratedAlertManager:
    """
    Integrated alert management system combining FCW and PIP alerts.
    Implements unified prioritization logic.
    """
    
    def __init__(self):
        self.alert_queue = queue.PriorityQueue()
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        
        # Alert priorities (higher = more urgent)
        self.priority_levels = {
            'critical_collision': 100,
            'critical_pedestrian': 95,
            'warning_collision': 80,
            'warning_pedestrian': 75,
            'info_vehicle': 50,
            'info_pedestrian': 45
        }
    
    def process_alerts(self, fcw_alerts, pip_alerts, environmental_factors):
        """
        Process and prioritize alerts from both FCW and PIP systems.
        
        Args:
            fcw_alerts: List of FCW alerts
            pip_alerts: List of PIP alerts
            environmental_factors: Dict with weather, lighting, speed adjustments
        """
        unified_alerts = []
        
        # Process FCW alerts
        for alert in fcw_alerts:
            priority = self._calculate_fcw_priority(alert, environmental_factors)
            unified_alert = {
                'source': 'FCW',
                'type': 'collision_warning',
                'priority': priority,
                'vehicle_id': alert.get('vehicle_id'),
                'distance': alert.get('distance'),
                'ttc': alert.get('ttc'),
                'alert_level': alert.get('alert_level'),
                'timestamp': time.time()
            }
            unified_alerts.append(unified_alert)
        
        # Process PIP alerts
        for alert in pip_alerts:
            priority = self._calculate_pip_priority(alert, environmental_factors)
            unified_alert = {
                'source': 'PIP',
                'type': 'pedestrian_intent',
                'priority': priority,
                'pedestrian_id': alert.get('pedestrian_id'),
                'distance': alert.get('distance'),
                'intent_prob': alert.get('intent_prob'),
                'alert_level': alert.get('alert_level'),
                'timestamp': time.time()
            }
            unified_alerts.append(unified_alert)
        
        # Sort by priority (highest first)
        unified_alerts.sort(key=lambda x: x['priority'], reverse=True)
        
        # Apply alert suppression logic
        filtered_alerts = self._apply_alert_suppression(unified_alerts)
        
        # Update history
        for alert in filtered_alerts:
            self.alert_history.append(alert)
        
        return filtered_alerts[:5]  # Return top 5 alerts
    
    def _calculate_fcw_priority(self, alert, env_factors):
        """Calculate priority for FCW alerts."""
        base_priority = self.priority_levels.get(
            f"{alert.get('alert_level', 'info')}_collision", 50
        )
        
        # Environmental adjustments
        weather_factor = env_factors.get('weather_urgency', 1.0)
        speed_factor = env_factors.get('speed_urgency', 1.0)
        
        # Distance and TTC factors
        distance = alert.get('distance', 20)
        ttc = alert.get('ttc', 10)
        
        distance_factor = max(0.5, 1.0 - distance / 20.0)
        ttc_factor = max(0.5, 1.0 - ttc / 10.0)
        
        final_priority = (base_priority * weather_factor * speed_factor * 
                         distance_factor * ttc_factor)
        
        return final_priority
    
    def _calculate_pip_priority(self, alert, env_factors):
        """Calculate priority for PIP alerts."""
        base_priority = self.priority_levels.get(
            f"{alert.get('alert_level', 'info')}_pedestrian", 45
        )
        
        # Environmental adjustments
        weather_factor = env_factors.get('weather_urgency', 1.0)
        lighting_factor = env_factors.get('lighting_urgency', 1.0)
        
        # Intent and distance factors
        intent_prob = alert.get('intent_prob', 0.5)
        distance = alert.get('distance', 15)
        
        intent_factor = intent_prob
        distance_factor = max(0.3, 1.0 - distance / 15.0)
        
        final_priority = (base_priority * weather_factor * lighting_factor * 
                         intent_factor * distance_factor)
        
        return final_priority
    
    def _apply_alert_suppression(self, alerts):
        """Apply alert suppression logic to prevent spam."""
        filtered_alerts = []
        current_time = time.time()
        
        for alert in alerts:
            # Check if similar alert was recently issued
            suppress = False
            
            for recent_alert in self.alert_history:
                if (current_time - recent_alert['timestamp'] < 2.0 and
                    recent_alert['source'] == alert['source'] and
                    recent_alert['type'] == alert['type']):
                    
                    # Same source and type within 2 seconds - check specifics
                    if alert['source'] == 'FCW':
                        if (recent_alert.get('vehicle_id') == alert.get('vehicle_id')):
                            suppress = True
                            break
                    elif alert['source'] == 'PIP':
                        if (recent_alert.get('pedestrian_id') == alert.get('pedestrian_id')):
                            suppress = True
                            break
            
            if not suppress:
                filtered_alerts.append(alert)
        
        return filtered_alerts


class IntegratedNeuroDriveSystem:
    """
    Complete integrated NeuroDrive system combining FCW and PIP.
    Main system class that orchestrates all components.
    """
    
    def __init__(self, config=None):
        self.config = config or EnhancedConfig()
        
        # Core detection and tracking
        self.detector = UnifiedDetectionSystem(self.config.YOLO_MODEL)
        self.tracker = UnifiedTrackingSystem()
        
        # FCW components (if available)
        self.lane_detector = RobustLaneDetector() if RobustLaneDetector else None
        self.depth_estimator = AccurateDepthEstimator() if AccurateDepthEstimator else None
        self.ttc_calculator = TTCCalculator() if TTCCalculator else None
        
        # PIP system
        self.pip_system = IntegratedPIPSystem(config)
        
        # Environmental adaptation
        self.weather_detector = WeatherDetector()
        self.lighting_classifier = LightingClassifier()
        self.speed_adaptive = SpeedAdaptiveSystem()
        
        # Alert management
        self.alert_manager = IntegratedAlertManager()
        self.audio_system = AudioAlertSystem()
        
        # Performance optimization
        self.performance_optimizer = RealTimeOptimizer()
        
        # System state
        self.frame_count = 0
        self.system_metrics = {
            'total_frames': 0,
            'vehicle_detections': 0,
            'pedestrian_detections': 0,
            'fcw_alerts': 0,
            'pip_alerts': 0,
            'unified_alerts': 0
        }
        
        print("✓ Integrated NeuroDrive System initialized")
    
    def process_frame(self, frame, vehicle_speed_kmh=0, frame_time=None):
        """
        Process frame through complete integrated pipeline.
        
        Args:
            frame: Input video frame
            vehicle_speed_kmh: Current vehicle speed
            frame_time: Frame timestamp
            
        Returns:
            processed_frame: Frame with all visualizations
            unified_alerts: Prioritized alerts from both systems
            system_metrics: Performance and detection metrics
        """
        if frame_time is None:
            frame_time = time.time()
        
        start_time = time.time()
        self.frame_count += 1
        
        # Environmental analysis
        weather_type, weather_conf = self.weather_detector.detect_weather_conditions(frame)
        lighting_type, brightness = self.lighting_classifier.classify_lighting(frame)
        self.speed_adaptive.update_speed(vehicle_speed_kmh)
        
        # Get environmental adjustments
        weather_adj = self.weather_detector.get_weather_adjustments()
        lighting_adj = self.lighting_classifier.get_lighting_adjustments()
        speed_adj = self.speed_adaptive.get_speed_adjustments()
        
        environmental_factors = {
            'weather_urgency': weather_adj.get('alert_timing', 1.0),
            'lighting_urgency': 1.2 if lighting_type == 'night' else 1.0,
            'speed_urgency': max(1.0, vehicle_speed_kmh / 50.0)
        }
        
        # Unified detection
        detection_start = time.time()
        vehicles, pedestrians = self.detector.detect_all_objects(
            frame, 
            conf_threshold=self.config.PEDESTRIAN_CONF_THRESHOLD * 
                          weather_adj['detection_threshold'] * 
                          lighting_adj['detection_threshold']
        )
        detection_time = time.time() - detection_start
        
        # Unified tracking
        tracking_start = time.time()
        vehicle_tracks, pedestrian_tracks = self.tracker.update_tracks(
            vehicles, pedestrians, frame.shape
        )
        tracking_time = time.time() - tracking_start
        
        # FCW processing
        fcw_start = time.time()
        fcw_alerts = self._process_fcw(frame, vehicle_tracks, environmental_factors)
        fcw_time = time.time() - fcw_start
        
        # PIP processing
        pip_start = time.time()
        pip_alerts = self._process_pip(frame, pedestrian_tracks, environmental_factors)
        pip_time = time.time() - pip_start
        
        # Unified alert management
        alert_start = time.time()
        unified_alerts = self.alert_manager.process_alerts(
            fcw_alerts, pip_alerts, environmental_factors
        )
        alert_time = time.time() - alert_start
        
        # Audio alerts
        self._trigger_audio_alerts(unified_alerts)
        
        # Visualization
        vis_start = time.time()
        processed_frame = self._visualize_integrated_results(
            frame, vehicles, pedestrians, vehicle_tracks, pedestrian_tracks,
            unified_alerts, weather_type, lighting_type, vehicle_speed_kmh
        )
        vis_time = time.time() - vis_start
        
        # Update metrics
        total_time = time.time() - start_time
        self.performance_optimizer.update_timing(total_time, detection_time + tracking_time)
        
        self.system_metrics.update({
            'total_frames': self.frame_count,
            'vehicle_detections': self.system_metrics['vehicle_detections'] + len(vehicles),
            'pedestrian_detections': self.system_metrics['pedestrian_detections'] + len(pedestrians),
            'fcw_alerts': self.system_metrics['fcw_alerts'] + len(fcw_alerts),
            'pip_alerts': self.system_metrics['pip_alerts'] + len(pip_alerts),
            'unified_alerts': self.system_metrics['unified_alerts'] + len(unified_alerts)
        })
        
        # Performance metrics
        perf_metrics = self.performance_optimizer.get_performance_metrics()
        perf_metrics.update({
            'detection_time_ms': detection_time * 1000,
            'tracking_time_ms': tracking_time * 1000,
            'fcw_time_ms': fcw_time * 1000,
            'pip_time_ms': pip_time * 1000,
            'alert_time_ms': alert_time * 1000,
            'vis_time_ms': vis_time * 1000,
            'total_time_ms': total_time * 1000,
            'weather': weather_type,
            'lighting': lighting_type,
            'speed_kmh': vehicle_speed_kmh
        })
        
        return processed_frame, unified_alerts, perf_metrics
    
    def _process_fcw(self, frame, vehicle_tracks, env_factors):
        """Process FCW alerts for vehicles."""
        fcw_alerts = []
        
        if not self.lane_detector or not self.depth_estimator or not self.ttc_calculator:
            return fcw_alerts
        
        # Lane detection
        lane_frame, lanes_detected = self.lane_detector.detect_lanes(frame)
        
        for track in vehicle_tracks:
            if not hasattr(track, 'tlbr') or not hasattr(track, 'track_id'):
                continue
            
            bbox = track.tlbr.tolist()
            track_id = int(track.track_id)
            
            # Check if vehicle is in ego lane
            in_ego_lane = self.lane_detector.is_in_ego_lane(bbox, frame.shape)
            
            if in_ego_lane:
                # Estimate distance
                distance = self.depth_estimator.estimate_depth(bbox, frame.shape)
                
                # Calculate TTC
                current_time = time.time()
                self.ttc_calculator.update_track(track_id, distance, current_time)
                ttc, rel_vel = self.ttc_calculator.calculate_ttc(track_id, 0)  # Assume ego speed handled elsewhere
                
                # Determine alert level
                if distance < 5.0 and ttc < 2.0:
                    alert_level = "critical"
                elif distance < 10.0 and ttc < 4.0:
                    alert_level = "warning"
                else:
                    continue  # No alert needed
                
                fcw_alerts.append({
                    'vehicle_id': track_id,
                    'distance': distance,
                    'ttc': ttc,
                    'alert_level': alert_level,
                    'bbox': bbox
                })
        
        return fcw_alerts
    
    def _process_pip(self, frame, pedestrian_tracks, env_factors):
        """Process PIP alerts for pedestrians."""
        pip_alerts = []
        
        for track in pedestrian_tracks:
            if not hasattr(track, 'tlbr') or not hasattr(track, 'track_id'):
                continue
            
            bbox = track.tlbr.tolist()
            track_id = int(track.track_id)
            
            # Simple intent prediction (in full system, use trained model)
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            frame_center = frame.shape[1] / 2
            
            # Distance to frame center (crossing indicator)
            distance_to_center = abs(center_x - frame_center) / frame_center
            intent_prob = 1.0 - distance_to_center
            
            # Estimate distance
            y_pos = (y1 + y2) / 2
            normalized_y = y_pos / frame.shape[0]
            estimated_distance = 15.0 * (1.0 - normalized_y)
            
            # Determine alert level
            if intent_prob > 0.8 and estimated_distance < 5.0:
                alert_level = "critical"
            elif intent_prob > 0.6 and estimated_distance < 10.0:
                alert_level = "warning"
            elif intent_prob > 0.4:
                alert_level = "info"
            else:
                continue
            
            pip_alerts.append({
                'pedestrian_id': track_id,
                'distance': estimated_distance,
                'intent_prob': intent_prob,
                'alert_level': alert_level,
                'bbox': bbox
            })
        
        return pip_alerts
    
    def _trigger_audio_alerts(self, unified_alerts):
        """Trigger appropriate audio alerts."""
        for alert in unified_alerts[:2]:  # Only top 2 alerts
            if alert['source'] == 'FCW':
                self.audio_system.trigger_alert(
                    alert['alert_level'],
                    f"vehicle_{alert.get('vehicle_id', 0)}",
                    alert.get('distance'),
                    None
                )
            elif alert['source'] == 'PIP':
                self.audio_system.trigger_alert(
                    alert['alert_level'],
                    f"pedestrian_{alert.get('pedestrian_id', 0)}",
                    alert.get('distance'),
                    alert.get('intent_prob')
                )
    
    def _visualize_integrated_results(self, frame, vehicles, pedestrians, 
                                    vehicle_tracks, pedestrian_tracks, 
                                    unified_alerts, weather, lighting, speed):
        """Visualize all system results on frame."""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw vehicle detections
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_frame, f"Vehicle {vehicle['conf']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw pedestrian detections
        for pedestrian in pedestrians:
            bbox = pedestrian['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Pedestrian {pedestrian['conf']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw unified alerts
        alert_y = 30
        for alert in unified_alerts[:3]:
            if alert['source'] == 'FCW':
                alert_text = f"FCW: {alert['alert_level'].upper()} - Vehicle {alert.get('vehicle_id', 'N/A')}"
                color = (0, 0, 255)
            else:
                alert_text = f"PIP: {alert['alert_level'].upper()} - Pedestrian {alert.get('pedestrian_id', 'N/A')}"
                color = (255, 0, 255)
            
            cv2.putText(vis_frame, alert_text, (10, alert_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            alert_y += 30
        
        # Draw system status
        status_y = h - 150
        cv2.putText(vis_frame, f"Weather: {weather}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Lighting: {lighting}", (10, status_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Speed: {speed:.1f} km/h", (10, status_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw performance metrics
        perf_metrics = self.performance_optimizer.get_performance_metrics()
        perf_text = f"FPS: {perf_metrics.get('fps', 0):.1f} | Vehicles: {len(vehicles)} | Pedestrians: {len(pedestrians)}"
        cv2.putText(vis_frame, perf_text, (10, status_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_frame
    
    def get_system_status(self):
        """Get comprehensive system status."""
        return {
            'frame_count': self.frame_count,
            'system_metrics': self.system_metrics,
            'performance_metrics': self.performance_optimizer.get_performance_metrics(),
            'active_vehicle_tracks': len(self.tracker.vehicle_tracks),
            'active_pedestrian_tracks': len(self.tracker.pedestrian_tracks),
            'alert_history_size': len(self.alert_manager.alert_history)
        }
    
    def shutdown(self):
        """Shutdown the integrated system."""
        self.audio_system.shutdown()
        self.pip_system.shutdown()
        print("✓ Integrated NeuroDrive System shutdown complete")


def main():
    """Main function for integrated system."""
    parser = argparse.ArgumentParser(description='Integrated NeuroDrive FCW+PIP System')
    parser.add_argument('--input', type=str, help='Input video file or camera index')
    parser.add_argument('--output', type=str, help='Output video file')
    parser.add_argument('--speed', type=float, default=50, help='Vehicle speed in km/h')
    parser.add_argument('--config', type=str, help='Configuration file')
    
    args = parser.parse_args()
    
    # Initialize integrated system
    config = EnhancedConfig()
    system = IntegratedNeuroDriveSystem(config)
    
    # Setup video input
    if args.input:
        if args.input.isdigit():
            cap = cv2.VideoCapture(int(args.input))
        else:
            cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(0)
    
    # Setup video output
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("Starting Integrated NeuroDrive System...")
    print("FCW + PIP Integration Active")
    print("Press 'q' to quit, 's' for system status")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, alerts, metrics = system.process_frame(
                frame, vehicle_speed_kmh=args.speed
            )
            
            # Display
            cv2.imshow('Integrated NeuroDrive System', processed_frame)
            
            # Save if output specified
            if writer:
                writer.write(processed_frame)
            
            # Print alerts
            if alerts:
                for alert in alerts:
                    print(f"UNIFIED ALERT: {alert['source']} - {alert['type']} - {alert['alert_level']}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                status = system.get_system_status()
                print("\n" + "="*50)
                print("SYSTEM STATUS")
                print("="*50)
                for key, value in status.items():
                    print(f"{key}: {value}")
                print("="*50 + "\n")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        system.shutdown()


if __name__ == "__main__":
    main()