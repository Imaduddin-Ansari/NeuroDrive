"""
Live Video Preview with Lane Detection and Indicator Controls

Press K for left indicator, L for right indicator, Q to quit
"""

import cv2
import os
from main import FindLaneLines


def live_video_preview(video_path):
    """
    Display live video preview with lane detection and keyboard controls
    
    Controls:
        K - Toggle left indicator
        L - Toggle right indicator  
        Q - Quit
    
    Parameters:
        video_path (str): Path to the input video file
    """
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Open video first to get its properties
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"{'='*60}")
    
    # Initialize lane detection with video dimensions
    print("Initializing lane detection system...")
    lane_detector = FindLaneLines(
        camera_cal_path='../Images/camera_cal',
        show_overlay=False,
        img_size=(width, height)  # Pass actual video size
    )
    
    left_indicator = False
    right_indicator = False
    
    frame_count = 0
    warning_count = 0
    detection_failures = 0
    
    print("\n🎮 CONTROLS:")
    print("   K - Toggle LEFT indicator")
    print("   L - Toggle RIGHT indicator")
    print("   Q - Quit")
    print(f"\n{'='*60}")
    print("Starting live preview...\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached")
            break
        
        frame_count += 1
        
        # Update indicator states
        lane_detector.set_indicator(left=left_indicator, right=right_indicator)
        
        try:
            # Process frame
            output_frame, lane_data = lane_detector.process_frame(frame)
            
            # Check if lane detection actually worked
            detection_failed = lane_data.get('detection_failed', False)
            if detection_failed:
                detection_failures += 1
            
            # Create display frame
            display_frame = output_frame.copy()
            
            # Add indicator status at top
            indicator_text = f"Indicators: L={'ON' if left_indicator else 'OFF'} | R={'ON' if right_indicator else 'OFF'}"
            cv2.putText(display_frame, indicator_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Add frame counter
            frame_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(display_frame, frame_text, (width - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show detection status
            if detection_failed:
                status_text = "⚠️ Lane Detection: FAILED"
                cv2.putText(display_frame, status_text, (10, height - 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                status_text = "✓ Lane Detection: OK"
                cv2.putText(display_frame, status_text, (10, height - 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add lane info only if detection worked
            if not detection_failed and lane_data['direction']:
                dir_text = f"Direction: {lane_data['direction']}"
                pos_text = f"Position: {lane_data['position']:.2f}m from center"
                
                cv2.putText(display_frame, dir_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, pos_text, (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display warning if present
            if lane_data['deviation_warning'] and not detection_failed:
                warning_count += 1
                
                # Big red warning banner
                cv2.rectangle(display_frame, (0, height-100), (width, height), (0, 0, 255), -1)
                cv2.putText(display_frame, "⚠️  LANE DEPARTURE WARNING  ⚠️", 
                           (width//2 - 300, height - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(display_frame, lane_data['warning_message'], 
                           (width//2 - 250, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Console warning
                print(f"⚠️  Frame {frame_count}: {lane_data['warning_message']}")
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            display_frame = frame.copy()
            cv2.putText(display_frame, "ERROR: Lane detection failed", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Lane Detection - Live Preview', display_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n🛑 Stopping video preview...")
            break
        elif key == ord('k') or key == ord('K'):
            left_indicator = not left_indicator
            status = '🟢 ON' if left_indicator else '⚫ OFF'
            print(f"Left indicator: {status}")
        elif key == ord('l') or key == ord('L'):
            right_indicator = not right_indicator
            status = '🟢 ON' if right_indicator else '⚫ OFF'
            print(f"Right indicator: {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Warnings triggered: {warning_count}")
    print(f"Detection failures: {detection_failures} ({detection_failures/frame_count*100:.1f}%)")
    if frame_count > detection_failures:
        print(f"Warning rate: {(warning_count/(frame_count-detection_failures)*100):.1f}%")
    print(f"{'='*60}\n")
    
    if detection_failures > frame_count * 0.5:
        print("⚠️  WARNING: More than 50% detection failures!")
        print("   This video may not be suitable for lane detection because:")
        print("   • Resolution is too different from calibration")
        print("   • Lane lines are not clearly visible")
        print("   • Camera angle is significantly different")
        print("   • Lighting conditions are poor")


if __name__ == "__main__":
    # Get video path from user
    print("\n🎬 Lane Detection - Live Video Preview")
    print("="*60)
    
    video_path = input("\nEnter video path: ").strip()
    
    if video_path:
        live_video_preview(video_path)
    else:
        print("No video path provided. Exiting...")