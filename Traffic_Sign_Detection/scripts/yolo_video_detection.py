# video_detection_processor.py
import cv2
import os
import time
from Traffic_sign_detection import YOLOv8Detector

class VideoDetectionProcessor:
    def __init__(self, detector, video_source=0, output_size=(1280, 720)):
        """
        Initialize Video Detection Processor
        
        Args:
            detector: YOLOv8Detector instance
            video_source: Video source (0 for webcam, or video file path)
            output_size: Output display size (width, height)
        """
        self.detector = detector
        self.video_source = video_source
        self.output_size = output_size
        self.cap = None
        self.is_video_file = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.start_time = 0
        
    def initialize_video_capture(self):
        """Initialize video capture from source"""
        try:
            if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                self.cap = cv2.VideoCapture(self.video_source)
                self.is_video_file = True
                print(f"Loading video file: {self.video_source}")
            else:
                self.cap = cv2.VideoCapture(self.video_source)
                self.is_video_file = False
                print(f"Initializing camera: {self.video_source}")
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open video source: {self.video_source}")
                
            # Get video properties
            if self.is_video_file:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps_source = self.cap.get(cv2.CAP_PROP_FPS)
                print(f"Video properties: {self.total_frames} frames, {self.fps_source:.2f} FPS")
            else:
                self.total_frames = 0
                self.fps_source = self.cap.get(cv2.CAP_PROP_FPS)
                print(f"Camera FPS: {self.fps_source:.2f}")
                
            return True
            
        except Exception as e:
            print(f"Error initializing video capture: {e}")
            return False
    
    def process_frame(self, frame):
        """
        Process a single frame through YOLOv8 detector
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            processed_frame: Frame with detections drawn
            detections: List of detection results
        """
        try:
            # Run YOLOv8 inference
            results = self.detector.model.predict(
                source=frame,
                conf=self.detector.conf_threshold,
                imgsz=640,
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes
            
            detections = []
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls.item())
                    conf = box.conf.item()
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    
                    class_name = self.detector.class_names[cls_id] if self.detector.class_names else f"Class_{cls_id}"
                    
                    detection_info = {
                        'class_id': cls_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': bbox,
                        'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]
                    }
                    detections.append(detection_info)
                    
                    # Draw detection on frame
                    frame = self.draw_detection_on_frame(frame, detection_info, i)
            
            # Add performance and info overlay
            frame = self.add_overlay_to_frame(frame, detections)
            
            return frame, detections
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, []
    
    def draw_detection_on_frame(self, frame, detection, detection_id):
        """
        Draw bounding box and label on video frame
        
        Args:
            frame: Input frame
            detection: Detection dictionary
            detection_id: ID of the detection
            
        Returns:
            frame: Frame with drawn detection
        """
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get color for this detection
        color = self.detector.colors[detection_id % len(self.detector.colors)].tolist()
        
        # Draw bounding box
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            2
        )
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1] - text_height - 10),
            (bbox[0] + text_width, bbox[1]),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (bbox[0], bbox[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return frame
    
    def add_overlay_to_frame(self, frame, detections):
        """
        Add information overlay to the frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            frame: Frame with overlay added
        """
        h, w = frame.shape[:2]
        
        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 110), (0, 0, 0), -1)
        
        # Add transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add information text
        info_lines = [
            f"Detections: {len(detections)}",
            f"FPS: {self.fps:.1f}",
            f"Frame: {self.frame_count}",
            "Press 'q' to quit",
            "Press 'p' to pause"
        ]
        
        for i, line in enumerate(info_lines):
            y_position = 35 + i * 18
            cv2.putText(
                frame,
                line,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Add progress bar for video files
        if self.is_video_file and self.total_frames > 0:
            progress = (self.frame_count / self.total_frames) * (w - 20)
            cv2.rectangle(frame, (10, h - 20), (int(progress), h - 10), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, h - 20), (w - 10, h - 10), (255, 255, 255), 1)
        
        return frame
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        
        if self.frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.fps = 30 / elapsed
            self.start_time = current_time
    
    def process_video(self, output_file=None):
        """
        Main video processing loop
        
        Args:
            output_file: Path to save output video (optional)
        """
        if not self.initialize_video_capture():
            return
        
        # Video writer setup
        writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
        
        print("\nStarting video processing...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Save current frame")
        
        self.start_time = time.time()
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        if self.is_video_file:
                            print("Video ended")
                            break
                        else:
                            print("Failed to grab frame")
                            continue
                    
                    # Process frame
                    processed_frame, detections = self.process_frame(frame)
                    
                    # Resize for display
                    display_frame = cv2.resize(processed_frame, self.output_size)
                    
                    # Calculate FPS
                    self.calculate_fps()
                    
                    # Display frame
                    cv2.imshow('YOLOv8 Real-time Detection', display_frame)
                    
                    # Save frame if writer is available
                    if writer:
                        writer.write(processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('p'):  # Pause/Resume
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s') and not paused:  # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as: {filename}")
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcessing completed!")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Average FPS: {self.fps:.1f}")

def main():
    """Main function to demonstrate video processing"""
    # Configuration
    MODEL_PATH = "../Model/Train/Weights/best.pt"  # Update this path
    VIDEO_SOURCE = "../TestVideos/videotrafficsign.mp4"  # 0 for webcam, or "path/to/video.mp4" for video file
    OUTPUT_VIDEO = "../Model/TestVideos/Detection_video.mp4"  # "output_detection.avi" to save output video
    
    # Initialize YOLOv8 detector
    print("Loading YOLOv8 detector...")
    detector = YOLOv8Detector(
        model_path=MODEL_PATH,
        conf_threshold=0.5  # Adjust confidence threshold as needed
    )
    
    # Initialize video processor
    video_processor = VideoDetectionProcessor(
        detector=detector,
        video_source=VIDEO_SOURCE,
        output_size=(1280, 720)
    )
    
    # Start video processing
    video_processor.process_video(output_file=OUTPUT_VIDEO)

def process_video_file(video_path, model_path, output_file=None):
    """
    Convenience function to process a specific video file
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLOv8 model
        output_file: Path to save output video (optional)
    """
    detector = YOLOv8Detector(model_path, conf_threshold=0.5)
    processor = VideoDetectionProcessor(
        detector=detector,
        video_source=video_path,
        output_size=(1280, 720)
    )
    processor.process_video(output_file=output_file)

def process_webcam(model_path, camera_index=0):
    """
    Convenience function to process webcam feed
    
    Args:
        model_path: Path to YOLOv8 model
        camera_index: Camera index (usually 0 for default camera)
    """
    detector = YOLOv8Detector(model_path, conf_threshold=0.5)
    processor = VideoDetectionProcessor(
        detector=detector,
        video_source=camera_index,
        output_size=(1280, 720)
    )
    processor.process_video()

if __name__ == "__main__":
    main()