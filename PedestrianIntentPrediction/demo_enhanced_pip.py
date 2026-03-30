#!/usr/bin/env python3
"""
ENHANCED PIP SYSTEM DEMO
========================
Interactive demonstration of the 6-week enhanced PIP system.
Shows all key features with real-time visualization.

Features Demonstrated:
- Multi-level audio alerts
- Environmental adaptation
- Multi-pedestrian handling
- FCW integration
- Performance monitoring

Author: NeuroDrive Team
Date: January 2025
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# Import enhanced PIP system
from enhanced_pip_system import IntegratedPIPSystem, EnhancedConfig
from integrated_fcw_pip_system import IntegratedNeuroDriveSystem


class PIPSystemDemo:
    """Interactive demo for the enhanced PIP system."""
    
    def __init__(self, use_integrated=True):
        self.config = EnhancedConfig()
        self.use_integrated = use_integrated
        
        if use_integrated:
            self.system = IntegratedNeuroDriveSystem(self.config)
            print("✓ Integrated FCW + PIP System loaded")
        else:
            self.system = IntegratedPIPSystem(self.config)
            print("✓ Enhanced PIP System loaded")
        
        # Demo parameters
        self.demo_speed = 60.0  # km/h
        self.frame_count = 0
        self.demo_scenarios = [
            "normal_driving",
            "rainy_weather",
            "night_driving",
            "multi_pedestrian",
            "high_speed"
        ]
        self.current_scenario = 0
        
        # Performance tracking
        self.fps_history = []
        self.alert_history = []
        
    def create_demo_frame(self, scenario="normal_driving"):
        """Create synthetic demo frames for different scenarios."""
        h, w = 480, 640
        
        if scenario == "normal_driving":
            # Clear day driving
            frame = np.random.randint(120, 180, (h, w, 3), dtype=np.uint8)
            # Add road markings
            cv2.line(frame, (w//4, h), (w//4, h//2), (255, 255, 255), 3)
            cv2.line(frame, (3*w//4, h), (3*w//4, h//2), (255, 255, 255), 3)
            
        elif scenario == "rainy_weather":
            # Rainy conditions
            frame = np.random.randint(80, 120, (h, w, 3), dtype=np.uint8)
            # Add rain effect
            for _ in range(100):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                cv2.line(frame, (x, y), (x+2, y+10), (200, 200, 200), 1)
            
        elif scenario == "night_driving":
            # Night driving
            frame = np.random.randint(20, 60, (h, w, 3), dtype=np.uint8)
            # Add headlight illumination
            center = (w//2, h)
            cv2.ellipse(frame, center, (w//3, h//3), 0, 180, 360, (100, 100, 100), -1)
            
        elif scenario == "multi_pedestrian":
            # Multiple pedestrians scenario
            frame = np.random.randint(100, 150, (h, w, 3), dtype=np.uint8)
            # Add multiple pedestrian-like rectangles
            for i in range(3):
                x = 200 + i * 100
                y = 300 + i * 20
                cv2.rectangle(frame, (x, y), (x+30, y+80), (0, 255, 0), -1)
            
        elif scenario == "high_speed":
            # High speed scenario with motion blur
            frame = np.random.randint(100, 160, (h, w, 3), dtype=np.uint8)
            # Add motion blur effect
            kernel = np.ones((1, 15), np.float32) / 15
            frame = cv2.filter2D(frame, -1, kernel)
            
        else:
            frame = np.random.randint(100, 150, (h, w, 3), dtype=np.uint8)
        
        # Add demo pedestrian
        ped_x = 300 + int(50 * np.sin(self.frame_count * 0.1))
        ped_y = 250 + int(20 * np.cos(self.frame_count * 0.05))
        cv2.rectangle(frame, (ped_x, ped_y), (ped_x+40, ped_y+100), (255, 0, 0), -1)
        
        return frame
    
    def run_demo(self, duration_seconds=60):
        """Run interactive demo."""
        print("\n" + "="*60)
        print("ENHANCED PIP SYSTEM - INTERACTIVE DEMO")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit demo")
        print("  's' - Switch scenario")
        print("  'p' - Show performance stats")
        print("  '+' - Increase speed")
        print("  '-' - Decrease speed")
        print("  'a' - Toggle audio alerts")
        print("="*60)
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_seconds:
                # Get current scenario
                scenario = self.demo_scenarios[self.current_scenario]
                
                # Create demo frame
                frame = self.create_demo_frame(scenario)
                
                # Process frame
                frame_start = time.time()
                
                if self.use_integrated:
                    processed_frame, alerts, metrics = self.system.process_frame(
                        frame, vehicle_speed_kmh=self.demo_speed
                    )
                else:
                    processed_frame, alerts, metrics = self.system.process_frame(
                        frame, vehicle_speed_kmh=self.demo_speed
                    )
                
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(fps)
                
                # Track alerts
                if alerts:
                    self.alert_history.extend(alerts)
                
                # Add demo information overlay
                self.add_demo_overlay(processed_frame, scenario, fps, alerts, metrics)
                
                # Display frame
                cv2.imshow('Enhanced PIP System Demo', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.current_scenario = (self.current_scenario + 1) % len(self.demo_scenarios)
                    print(f"Switched to scenario: {self.demo_scenarios[self.current_scenario]}")
                elif key == ord('p'):
                    self.show_performance_stats()
                elif key == ord('+'):
                    self.demo_speed = min(120, self.demo_speed + 10)
                    print(f"Speed increased to: {self.demo_speed} km/h")
                elif key == ord('-'):
                    self.demo_speed = max(20, self.demo_speed - 10)
                    print(f"Speed decreased to: {self.demo_speed} km/h")
                elif key == ord('a'):
                    print("Audio alert triggered manually")
                    if hasattr(self.system, 'audio_system'):
                        self.system.audio_system.trigger_alert("warning", "demo_ped", 5.0, 0.8)
                
                self.frame_count += 1
                
                # Scenario auto-switch every 10 seconds
                if self.frame_count % 300 == 0:  # ~10 seconds at 30 FPS
                    self.current_scenario = (self.current_scenario + 1) % len(self.demo_scenarios)
                    print(f"Auto-switched to scenario: {self.demo_scenarios[self.current_scenario]}")
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            self.system.shutdown()
            self.show_final_stats()
    
    def add_demo_overlay(self, frame, scenario, fps, alerts, metrics):
        """Add informative overlay to demo frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Top panel - scenario and system info
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Enhanced PIP System Demo", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current scenario
        cv2.putText(frame, f"Scenario: {scenario.replace('_', ' ').title()}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Speed and FPS
        cv2.putText(frame, f"Speed: {self.demo_speed:.0f} km/h | FPS: {fps:.1f}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # System type
        system_type = "Integrated FCW+PIP" if self.use_integrated else "Enhanced PIP"
        cv2.putText(frame, f"System: {system_type}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Alerts panel
        if alerts:
            alert_y = 140
            cv2.rectangle(frame, (0, 130), (w, 130 + len(alerts) * 25 + 10), (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.8, frame, 0.2, 0, frame)
            
            for i, alert in enumerate(alerts[:3]):  # Show max 3 alerts
                if self.use_integrated:
                    alert_text = f"{alert['source']}: {alert['type']} - {alert['alert_level']}"
                    color = (0, 0, 255) if alert['alert_level'] == 'critical' else (0, 165, 255)
                else:
                    alert_text = f"PIP: {alert['alert_level']} - Ped {alert['pedestrian_id']}"
                    color = (0, 0, 255) if alert['alert_level'] == 'critical' else (0, 165, 255)
                
                cv2.putText(frame, alert_text, (10, alert_y + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Performance metrics (bottom right)
        if metrics:
            perf_y = h - 80
            cv2.rectangle(frame, (w-250, h-90), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.8, frame, 0.2, 0, frame)
            
            weather = metrics.get('weather', 'clear')
            lighting = metrics.get('lighting', 'day')
            
            cv2.putText(frame, f"Weather: {weather}", (w-240, perf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Lighting: {lighting}", (w-240, perf_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            total_time = metrics.get('total_time_ms', 0)
            cv2.putText(frame, f"Latency: {total_time:.1f}ms", (w-240, perf_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controls reminder (bottom left)
        controls_y = h - 60
        cv2.rectangle(frame, (0, h-70), (300, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "Controls: q=quit, s=scenario, p=stats", (10, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "+/-=speed, a=audio test", (10, controls_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def show_performance_stats(self):
        """Display current performance statistics."""
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history[-30:])  # Last 30 frames
            min_fps = np.min(self.fps_history[-30:])
            max_fps = np.max(self.fps_history[-30:])
            
            print(f"FPS - Avg: {avg_fps:.1f}, Min: {min_fps:.1f}, Max: {max_fps:.1f}")
        
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Alerts Generated: {len(self.alert_history)}")
        print(f"Current Speed: {self.demo_speed} km/h")
        print(f"Current Scenario: {self.demo_scenarios[self.current_scenario]}")
        
        if hasattr(self.system, 'get_system_status'):
            status = self.system.get_system_status()
            print(f"Active Tracks: {status.get('active_pedestrian_tracks', 0)}")
        
        print("="*50)
    
    def show_final_stats(self):
        """Show final demo statistics."""
        print("\n" + "="*60)
        print("DEMO COMPLETED - FINAL STATISTICS")
        print("="*60)
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Total Frames: {len(self.fps_history)}")
        
        print(f"Total Alerts: {len(self.alert_history)}")
        
        # Alert breakdown
        if self.alert_history:
            alert_levels = {}
            for alert in self.alert_history:
                level = alert.get('alert_level', 'unknown')
                alert_levels[level] = alert_levels.get(level, 0) + 1
            
            print("Alert Breakdown:")
            for level, count in alert_levels.items():
                print(f"  {level.title()}: {count}")
        
        # Scenarios tested
        print(f"Scenarios Demonstrated: {len(self.demo_scenarios)}")
        for scenario in self.demo_scenarios:
            print(f"  - {scenario.replace('_', ' ').title()}")
        
        print("\n✅ Demo completed successfully!")
        print("="*60)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Enhanced PIP System Demo')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Demo duration in seconds (default: 60)')
    parser.add_argument('--integrated', action='store_true', 
                       help='Use integrated FCW+PIP system')
    parser.add_argument('--speed', type=float, default=60, 
                       help='Initial vehicle speed in km/h')
    
    args = parser.parse_args()
    
    print("Initializing Enhanced PIP System Demo...")
    
    try:
        # Create and run demo
        demo = PIPSystemDemo(use_integrated=args.integrated)
        demo.demo_speed = args.speed
        demo.run_demo(duration_seconds=args.duration)
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()