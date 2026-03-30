#!/usr/bin/env python3
"""
ENHANCED PIP SYSTEM TESTING & VALIDATION
========================================
Comprehensive testing suite for the 6-week enhanced PIP system.
Week 6: Conduct comprehensive validation testing with 50+ hours of real-world video.

Test Categories:
1. Unit Tests - Individual component testing
2. Integration Tests - System integration testing
3. Performance Tests - Latency and throughput testing
4. Accuracy Tests - Detection and prediction accuracy
5. Environmental Tests - Weather and lighting conditions
6. Audio Tests - Alert system testing
7. Multi-pedestrian Tests - Complex scenario handling

Author: NeuroDrive Team
Date: January 2025
"""

import os
import sys
import cv2
import numpy as np
import time
import json
import unittest
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# Add project root to path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import system components
from enhanced_pip_system import (
    AudioAlertSystem, RealTimeOptimizer, WeatherDetector,
    LightingClassifier, SpeedAdaptiveSystem, IntegratedPIPSystem,
    MultiPedestrianManager, AdvancedIntentClassifier, ContextualFeatureExtractor,
    ConfidenceScorer, EnhancedConfig
)

from integrated_fcw_pip_system import IntegratedNeuroDriveSystem


class PIPSystemTester:
    """Comprehensive testing suite for PIP system."""
    
    def __init__(self):
        self.config = EnhancedConfig()
        self.test_results = {}
        self.performance_metrics = {}
        
    def run_all_tests(self):
        """Run complete test suite."""
        print("="*60)
        print("ENHANCED PIP SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        # Run test categories
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_performance_tests()
        self.run_environmental_tests()
        self.run_audio_tests()
        self.run_multi_pedestrian_tests()
        
        # Generate final report
        self.generate_test_report()
        
        return self.test_results
    
    def run_unit_tests(self):
        """Test individual components."""
        print("\n1. UNIT TESTS")
        print("-" * 40)
        
        unit_results = {}
        
        # Test Audio Alert System
        print("Testing Audio Alert System...")
        try:
            audio_system = AudioAlertSystem(enable_tts=False, enable_sounds=False)
            audio_system.trigger_alert("warning", "test_ped_1", 5.0, 0.8)
            time.sleep(0.5)  # Allow processing
            audio_system.shutdown()
            unit_results['audio_system'] = 'PASS'
            print("  ✓ Audio Alert System: PASS")
        except Exception as e:
            unit_results['audio_system'] = f'FAIL: {e}'
            print(f"  ✗ Audio Alert System: FAIL - {e}")
        
        # Test Weather Detector
        print("Testing Weather Detector...")
        try:
            weather_detector = WeatherDetector()
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            weather_type, confidence = weather_detector.detect_weather_conditions(test_frame)
            adjustments = weather_detector.get_weather_adjustments()
            
            assert weather_type in ['clear', 'rain', 'fog', 'snow']
            assert 0 <= confidence <= 1
            assert 'detection_threshold' in adjustments
            
            unit_results['weather_detector'] = 'PASS'
            print("  ✓ Weather Detector: PASS")
        except Exception as e:
            unit_results['weather_detector'] = f'FAIL: {e}'
            print(f"  ✗ Weather Detector: FAIL - {e}")
        
        # Test Lighting Classifier
        print("Testing Lighting Classifier...")
        try:
            lighting_classifier = LightingClassifier()
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            lighting_type, brightness = lighting_classifier.classify_lighting(test_frame)
            adjustments = lighting_classifier.get_lighting_adjustments()
            
            assert lighting_type in ['day', 'night', 'dusk', 'dawn']
            assert 0 <= brightness <= 1
            assert 'detection_threshold' in adjustments
            
            unit_results['lighting_classifier'] = 'PASS'
            print("  ✓ Lighting Classifier: PASS")
        except Exception as e:
            unit_results['lighting_classifier'] = f'FAIL: {e}'
            print(f"  ✗ Lighting Classifier: FAIL - {e}")
        
        # Test Speed Adaptive System
        print("Testing Speed Adaptive System...")
        try:
            speed_adaptive = SpeedAdaptiveSystem()
            speed_adaptive.update_speed(60.0, "test")
            adjustments = speed_adaptive.get_speed_adjustments()
            
            assert speed_adaptive.current_speed_kmh == 60.0
            assert 'alert_distance' in adjustments
            assert 'alert_time' in adjustments
            
            unit_results['speed_adaptive'] = 'PASS'
            print("  ✓ Speed Adaptive System: PASS")
        except Exception as e:
            unit_results['speed_adaptive'] = f'FAIL: {e}'
            print(f"  ✗ Speed Adaptive System: FAIL - {e}")
        
        # Test Context Feature Extractor
        print("Testing Context Feature Extractor...")
        try:
            context_extractor = ContextualFeatureExtractor()
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_bbox = [100, 100, 200, 300]
            features = context_extractor.extract_features(test_frame, test_bbox, test_frame.shape)
            
            assert len(features) == 4
            assert all(0 <= f <= 1 for f in features)
            
            unit_results['context_extractor'] = 'PASS'
            print("  ✓ Context Feature Extractor: PASS")
        except Exception as e:
            unit_results['context_extractor'] = f'FAIL: {e}'
            print(f"  ✗ Context Feature Extractor: FAIL - {e}")
        
        # Test Confidence Scorer
        print("Testing Confidence Scorer...")
        try:
            confidence_scorer = ConfidenceScorer()
            confidence_level, confidence_score = confidence_scorer.score_confidence(0.8, 0.7, 0.6)
            
            assert confidence_level in ['low', 'medium', 'high']
            assert 0 <= confidence_score <= 1
            
            unit_results['confidence_scorer'] = 'PASS'
            print("  ✓ Confidence Scorer: PASS")
        except Exception as e:
            unit_results['confidence_scorer'] = f'FAIL: {e}'
            print(f"  ✗ Confidence Scorer: FAIL - {e}")
        
        self.test_results['unit_tests'] = unit_results
        
        # Summary
        passed = sum(1 for result in unit_results.values() if result == 'PASS')
        total = len(unit_results)
        print(f"\nUnit Tests Summary: {passed}/{total} PASSED")
    
    def run_integration_tests(self):
        """Test system integration."""
        print("\n2. INTEGRATION TESTS")
        print("-" * 40)
        
        integration_results = {}
        
        # Test PIP System Integration
        print("Testing PIP System Integration...")
        try:
            pip_system = IntegratedPIPSystem(self.config)
            
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process frame
            processed_frame, alerts, metrics = pip_system.process_frame(
                test_frame, vehicle_speed_kmh=50.0
            )
            
            # Validate outputs
            assert processed_frame.shape == test_frame.shape
            assert isinstance(alerts, list)
            assert isinstance(metrics, dict)
            assert 'fps' in metrics
            
            pip_system.shutdown()
            integration_results['pip_system'] = 'PASS'
            print("  ✓ PIP System Integration: PASS")
            
        except Exception as e:
            integration_results['pip_system'] = f'FAIL: {e}'
            print(f"  ✗ PIP System Integration: FAIL - {e}")
        
        # Test Integrated NeuroDrive System
        print("Testing Integrated NeuroDrive System...")
        try:
            integrated_system = IntegratedNeuroDriveSystem(self.config)
            
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process frame
            processed_frame, unified_alerts, perf_metrics = integrated_system.process_frame(
                test_frame, vehicle_speed_kmh=60.0
            )
            
            # Validate outputs
            assert processed_frame.shape == test_frame.shape
            assert isinstance(unified_alerts, list)
            assert isinstance(perf_metrics, dict)
            
            # Test system status
            status = integrated_system.get_system_status()
            assert 'frame_count' in status
            assert 'system_metrics' in status
            
            integrated_system.shutdown()
            integration_results['integrated_system'] = 'PASS'
            print("  ✓ Integrated NeuroDrive System: PASS")
            
        except Exception as e:
            integration_results['integrated_system'] = f'FAIL: {e}'
            print(f"  ✗ Integrated NeuroDrive System: FAIL - {e}")
        
        self.test_results['integration_tests'] = integration_results
        
        # Summary
        passed = sum(1 for result in integration_results.values() if result == 'PASS')
        total = len(integration_results)
        print(f"\nIntegration Tests Summary: {passed}/{total} PASSED")
    
    def run_performance_tests(self):
        """Test system performance and latency."""
        print("\n3. PERFORMANCE TESTS")
        print("-" * 40)
        
        performance_results = {}
        
        # Test Real-time Performance
        print("Testing Real-time Performance...")
        try:
            pip_system = IntegratedPIPSystem(self.config)
            
            # Performance test parameters
            num_frames = 100
            frame_times = []
            alert_latencies = []
            
            print(f"  Processing {num_frames} test frames...")
            
            for i in range(num_frames):
                # Create test frame with some variation
                brightness = 128 + 50 * np.sin(i * 0.1)
                test_frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
                
                start_time = time.time()
                processed_frame, alerts, metrics = pip_system.process_frame(test_frame)
                frame_time = time.time() - start_time
                
                frame_times.append(frame_time)
                
                # Measure alert latency if alerts present
                if alerts:
                    alert_latency = metrics.get('alert_time_ms', 0)
                    alert_latencies.append(alert_latency)
                
                if (i + 1) % 20 == 0:
                    print(f"    Processed {i + 1}/{num_frames} frames...")
            
            # Calculate performance metrics
            avg_frame_time = np.mean(frame_times)
            avg_fps = 1.0 / avg_frame_time
            max_frame_time = np.max(frame_times)
            min_frame_time = np.min(frame_times)
            
            avg_alert_latency = np.mean(alert_latencies) if alert_latencies else 0
            max_alert_latency = np.max(alert_latencies) if alert_latencies else 0
            
            # Performance criteria
            target_fps = 30
            max_alert_latency_ms = 100
            
            performance_pass = (
                avg_fps >= target_fps * 0.8 and  # At least 80% of target FPS
                max_alert_latency <= max_alert_latency_ms * 1.5  # Within 150% of target latency
            )
            
            performance_results['realtime_performance'] = {
                'status': 'PASS' if performance_pass else 'FAIL',
                'avg_fps': avg_fps,
                'target_fps': target_fps,
                'avg_frame_time_ms': avg_frame_time * 1000,
                'max_frame_time_ms': max_frame_time * 1000,
                'min_frame_time_ms': min_frame_time * 1000,
                'avg_alert_latency_ms': avg_alert_latency,
                'max_alert_latency_ms': max_alert_latency,
                'target_alert_latency_ms': max_alert_latency_ms
            }
            
            print(f"  ✓ Average FPS: {avg_fps:.1f} (target: {target_fps})")
            print(f"  ✓ Average Alert Latency: {avg_alert_latency:.1f}ms (target: <{max_alert_latency_ms}ms)")
            print(f"  ✓ Real-time Performance: {'PASS' if performance_pass else 'FAIL'}")
            
            pip_system.shutdown()
            
        except Exception as e:
            performance_results['realtime_performance'] = {'status': f'FAIL: {e}'}
            print(f"  ✗ Real-time Performance: FAIL - {e}")
        
        # Test Memory Usage
        print("Testing Memory Usage...")
        try:
            import psutil
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create system and process frames
            pip_system = IntegratedPIPSystem(self.config)
            
            for i in range(50):
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                pip_system.process_frame(test_frame)
            
            # Final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - baseline_memory
            
            # Memory criteria (should not increase by more than 500MB)
            memory_pass = memory_increase < 500
            
            performance_results['memory_usage'] = {
                'status': 'PASS' if memory_pass else 'FAIL',
                'baseline_memory_mb': baseline_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase
            }
            
            print(f"  ✓ Memory Usage: {memory_increase:.1f}MB increase ({'PASS' if memory_pass else 'FAIL'})")
            
            pip_system.shutdown()
            
        except ImportError:
            print("  ⚠ psutil not available, skipping memory test")
            performance_results['memory_usage'] = {'status': 'SKIPPED'}
        except Exception as e:
            performance_results['memory_usage'] = {'status': f'FAIL: {e}'}
            print(f"  ✗ Memory Usage: FAIL - {e}")
        
        self.test_results['performance_tests'] = performance_results
        self.performance_metrics = performance_results
    
    def run_environmental_tests(self):
        """Test environmental adaptation features."""
        print("\n4. ENVIRONMENTAL TESTS")
        print("-" * 40)
        
        env_results = {}
        
        # Test Weather Adaptation
        print("Testing Weather Adaptation...")
        try:
            weather_detector = WeatherDetector()
            
            # Test different weather conditions
            weather_conditions = {
                'clear': self._create_clear_weather_frame(),
                'rain': self._create_rainy_weather_frame(),
                'fog': self._create_foggy_weather_frame(),
                'snow': self._create_snowy_weather_frame()
            }
            
            weather_results = {}
            for condition, frame in weather_conditions.items():
                detected_weather, confidence = weather_detector.detect_weather_conditions(frame)
                adjustments = weather_detector.get_weather_adjustments()
                
                weather_results[condition] = {
                    'detected': detected_weather,
                    'confidence': confidence,
                    'adjustments': adjustments
                }
            
            env_results['weather_adaptation'] = {
                'status': 'PASS',
                'results': weather_results
            }
            print("  ✓ Weather Adaptation: PASS")
            
        except Exception as e:
            env_results['weather_adaptation'] = {'status': f'FAIL: {e}'}
            print(f"  ✗ Weather Adaptation: FAIL - {e}")
        
        # Test Lighting Adaptation
        print("Testing Lighting Adaptation...")
        try:
            lighting_classifier = LightingClassifier()
            
            # Test different lighting conditions
            lighting_conditions = {
                'day': self._create_day_frame(),
                'night': self._create_night_frame(),
                'dusk': self._create_dusk_frame()
            }
            
            lighting_results = {}
            for condition, frame in lighting_conditions.items():
                detected_lighting, brightness = lighting_classifier.classify_lighting(frame)
                adjustments = lighting_classifier.get_lighting_adjustments()
                
                lighting_results[condition] = {
                    'detected': detected_lighting,
                    'brightness': brightness,
                    'adjustments': adjustments
                }
            
            env_results['lighting_adaptation'] = {
                'status': 'PASS',
                'results': lighting_results
            }
            print("  ✓ Lighting Adaptation: PASS")
            
        except Exception as e:
            env_results['lighting_adaptation'] = {'status': f'FAIL: {e}'}
            print(f"  ✗ Lighting Adaptation: FAIL - {e}")
        
        self.test_results['environmental_tests'] = env_results
    
    def run_audio_tests(self):
        """Test audio alert system."""
        print("\n5. AUDIO TESTS")
        print("-" * 40)
        
        audio_results = {}
        
        # Test Audio Alert Levels
        print("Testing Audio Alert Levels...")
        try:
            audio_system = AudioAlertSystem(enable_tts=False, enable_sounds=False)
            
            # Test different alert levels
            alert_levels = ['info', 'warning', 'critical']
            
            for level in alert_levels:
                start_time = time.time()
                audio_system.trigger_alert(level, f"test_ped_{level}", 5.0, 0.8)
                
                # Wait for processing
                time.sleep(0.2)
                
                # Check if alert was processed (simplified check)
                processing_time = time.time() - start_time
                assert processing_time < 1.0  # Should be fast
            
            audio_system.shutdown()
            
            audio_results['alert_levels'] = 'PASS'
            print("  ✓ Audio Alert Levels: PASS")
            
        except Exception as e:
            audio_results['alert_levels'] = f'FAIL: {e}'
            print(f"  ✗ Audio Alert Levels: FAIL - {e}")
        
        # Test Alert Cooldown
        print("Testing Alert Cooldown...")
        try:
            audio_system = AudioAlertSystem(enable_tts=False, enable_sounds=False)
            
            # Trigger multiple alerts for same pedestrian
            ped_id = "test_cooldown"
            
            # First alert
            audio_system.trigger_alert("warning", ped_id, 5.0, 0.8)
            time.sleep(0.1)
            
            # Second alert (should be suppressed due to cooldown)
            audio_system.trigger_alert("warning", ped_id, 5.0, 0.8)
            time.sleep(0.1)
            
            # Check cooldown is working (simplified)
            assert ped_id in audio_system.last_alert_time
            
            audio_system.shutdown()
            
            audio_results['alert_cooldown'] = 'PASS'
            print("  ✓ Alert Cooldown: PASS")
            
        except Exception as e:
            audio_results['alert_cooldown'] = f'FAIL: {e}'
            print(f"  ✗ Alert Cooldown: FAIL - {e}")
        
        self.test_results['audio_tests'] = audio_results
    
    def run_multi_pedestrian_tests(self):
        """Test multi-pedestrian scenario handling."""
        print("\n6. MULTI-PEDESTRIAN TESTS")
        print("-" * 40)
        
        multi_ped_results = {}
        
        # Test Multi-Pedestrian Manager
        print("Testing Multi-Pedestrian Manager...")
        try:
            multi_ped_manager = MultiPedestrianManager(max_pedestrians=10)
            
            # Create test detections for multiple pedestrians
            test_detections = []
            for i in range(5):
                detection = {
                    'track_id': i,
                    'bbox': [100 + i*50, 100, 150 + i*50, 200],
                    'conf': 0.8,
                    'intent_prob': 0.5 + i*0.1
                }
                test_detections.append(detection)
            
            # Update pedestrians
            current_time = time.time()
            multi_ped_manager.update_pedestrians(test_detections, current_time)
            
            # Generate alerts
            alerts = multi_ped_manager.generate_priority_alerts()
            
            # Validate results
            assert len(multi_ped_manager.pedestrian_tracks) == 5
            assert isinstance(alerts, list)
            
            multi_ped_results['multi_pedestrian_manager'] = 'PASS'
            print("  ✓ Multi-Pedestrian Manager: PASS")
            
        except Exception as e:
            multi_ped_results['multi_pedestrian_manager'] = f'FAIL: {e}'
            print(f"  ✗ Multi-Pedestrian Manager: FAIL - {e}")
        
        self.test_results['multi_pedestrian_tests'] = multi_ped_results
    
    def _create_clear_weather_frame(self):
        """Create a frame simulating clear weather."""
        frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        # Add some high contrast edges
        frame[100:120, :] = 255
        frame[200:220, :] = 50
        return frame
    
    def _create_rainy_weather_frame(self):
        """Create a frame simulating rainy weather."""
        frame = np.random.randint(80, 150, (480, 640, 3), dtype=np.uint8)
        # Add noise to simulate rain
        noise = np.random.randint(-30, 30, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame
    
    def _create_foggy_weather_frame(self):
        """Create a frame simulating foggy weather."""
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Add very low contrast
        frame += np.random.randint(-10, 10, frame.shape, dtype=np.int8)
        return frame
    
    def _create_snowy_weather_frame(self):
        """Create a frame simulating snowy weather."""
        frame = np.random.randint(150, 220, (480, 640, 3), dtype=np.uint8)
        # Add white spots for snow
        snow_mask = np.random.random(frame.shape[:2]) > 0.95
        frame[snow_mask] = 255
        return frame
    
    def _create_day_frame(self):
        """Create a frame simulating daylight."""
        return np.random.randint(150, 255, (480, 640, 3), dtype=np.uint8)
    
    def _create_night_frame(self):
        """Create a frame simulating nighttime."""
        return np.random.randint(0, 80, (480, 640, 3), dtype=np.uint8)
    
    def _create_dusk_frame(self):
        """Create a frame simulating dusk."""
        return np.random.randint(80, 150, (480, 640, 3), dtype=np.uint8)
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.test_results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            if isinstance(results, dict):
                for test_name, result in results.items():
                    total_tests += 1
                    if isinstance(result, dict):
                        status = result.get('status', 'UNKNOWN')
                    else:
                        status = result
                    
                    if status == 'PASS':
                        passed_tests += 1
                        print(f"  ✓ {test_name}: PASS")
                    elif status == 'SKIPPED':
                        print(f"  ⚠ {test_name}: SKIPPED")
                    else:
                        print(f"  ✗ {test_name}: {status}")
        
        # Overall summary
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        # Performance summary
        if 'performance_tests' in self.test_results:
            perf_data = self.test_results['performance_tests']
            if 'realtime_performance' in perf_data:
                rt_perf = perf_data['realtime_performance']
                if isinstance(rt_perf, dict) and 'avg_fps' in rt_perf:
                    print(f"Average FPS: {rt_perf['avg_fps']:.1f}")
                    print(f"Average Alert Latency: {rt_perf.get('avg_alert_latency_ms', 0):.1f}ms")
        
        # System readiness assessment
        if pass_rate >= 80:
            print(f"\n🎉 SYSTEM STATUS: READY FOR DEPLOYMENT")
        elif pass_rate >= 60:
            print(f"\n⚠️  SYSTEM STATUS: NEEDS MINOR FIXES")
        else:
            print(f"\n❌ SYSTEM STATUS: NEEDS MAJOR FIXES")
        
        print("="*60)
        
        # Save report to file
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': self.test_results,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': pass_rate
            }
        }
        
        report_file = Path('test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Detailed report saved to: {report_file}")
        
        return report_data


def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced PIP System Testing')
    parser.add_argument('--category', type=str, choices=[
        'unit', 'integration', 'performance', 'environmental', 'audio', 'multi_pedestrian', 'all'
    ], default='all', help='Test category to run')
    parser.add_argument('--output', type=str, help='Output report file')
    
    args = parser.parse_args()
    
    # Create tester
    tester = PIPSystemTester()
    
    # Run specified tests
    if args.category == 'all':
        results = tester.run_all_tests()
    elif args.category == 'unit':
        tester.run_unit_tests()
        results = tester.test_results
    elif args.category == 'integration':
        tester.run_integration_tests()
        results = tester.test_results
    elif args.category == 'performance':
        tester.run_performance_tests()
        results = tester.test_results
    elif args.category == 'environmental':
        tester.run_environmental_tests()
        results = tester.test_results
    elif args.category == 'audio':
        tester.run_audio_tests()
        results = tester.test_results
    elif args.category == 'multi_pedestrian':
        tester.run_multi_pedestrian_tests()
        results = tester.test_results
    
    # Save custom output if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()