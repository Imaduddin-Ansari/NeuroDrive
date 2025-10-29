import tkinter as tk
from tkinter import ttk
import json
import os
from PIL import Image, ImageTk
import cv2
import time
import threading
import queue
import sys
from pathlib import Path

# Add the blind spot module path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TrafficSignDetection.TrafficSign import TrafficSignDetector


class TrafficSignProcessor:
    """Wrapper for traffic sign detection using your existing combined_script.py"""
    
    def __init__(self, video_source, model_path, alert_threshold=0.90):
        self.video_source = video_source
        self.model_path = model_path
        self.alert_threshold = alert_threshold
        self.is_running = False
        self.detection_status = False
        self.current_sign = None
        self.sign_confidence = 0.0
        self.latest_detections = []
        self.processed_frame = None
        self.process_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.detector = None
        
    def start(self):
        """Start traffic sign processing in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.process_thread = threading.Thread(target=self._process_video, daemon=True)
            self.process_thread.start()
    
    def stop(self):
        """Stop traffic sign processing"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
    
    def _process_video(self):
        """Process video through traffic sign detector"""
        try:
            # Initialize the detector from your combined_script.py
            self.detector = TrafficSignDetector(
                self.model_path,
                alert_threshold=self.alert_threshold,
                verbose=False  # Silent mode for UI
            )
            
            # Open video
            cap = cv2.VideoCapture(self.video_source)
            
            if not cap.isOpened():
                print(f"Failed to open video source: {self.video_source}")
                self.is_running = False
                return
            
            print(f"✓ Traffic sign detection started on front camera")
            
            frame_count = 0
            skip_frames = 2  # Process every 2nd frame for performance
            resize_scale = 0.5
            
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % skip_frames != 0:
                    # Still send frame to queue for display
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait(frame.copy())
                        except queue.Full:
                            pass
                    time.sleep(0.01)
                    continue
                
                # Resize for faster processing
                small_frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
                
                # Run YOLO detection
                yolo_results = self.detector.yolo_model(small_frame, verbose=False, conf=0.12)
                
                detected_signs = []
                
                for result in yolo_results:
                    boxes = result.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        yolo_confidence = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.detector.yolo_model.names[cls]
                        
                        # Scale coordinates back
                        x1 = int(x1 / resize_scale)
                        y1 = int(y1 / resize_scale)
                        x2 = int(x2 / resize_scale)
                        y2 = int(y2 / resize_scale)
                        
                        # Clamp to frame boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        # Check if traffic sign
                        is_traffic_sign = any(keyword.lower() in class_name.lower() 
                                             for keyword in self.detector.traffic_sign_keywords)
                        
                        if is_traffic_sign:
                            # Extract and classify
                            sign_crop = frame[y1:y2, x1:x2]
                            
                            if sign_crop.size > 0:
                                classification = self.detector.classify_sign(sign_crop)
                                
                                if classification and classification['confidence'] >= 0.5:
                                    detected_signs.append({
                                        'class': classification['class'],
                                        'confidence': classification['confidence'],
                                        'bbox': (x1, y1, x2, y2),
                                        'is_alert': classification['is_alert']
                                    })
                                    
                                    # Draw on frame
                                    color = (0, 0, 255) if classification['is_alert'] else (0, 255, 0)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Add label
                                    label = f"{classification['class']}: {classification['confidence']:.2%}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                                (x1 + label_size[0], y1), color, -1)
                                    cv2.putText(frame, label, (x1, y1 - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Update detection status
                self.detection_status = len(detected_signs) > 0
                self.latest_detections = detected_signs
                
                # Get highest confidence detection
                if detected_signs:
                    best_detection = max(detected_signs, key=lambda x: x['confidence'])
                    self.current_sign = best_detection['class']
                    self.sign_confidence = best_detection['confidence']
                else:
                    self.current_sign = None
                    self.sign_confidence = 0.0
                
                # Store processed frame in queue
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                
                time.sleep(0.03)  # ~30 FPS
            
            cap.release()
            print(f"✓ Traffic sign detection stopped")
            
        except Exception as e:
            print(f"❌ Traffic sign processing error: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
    
    def get_processed_frame(self):
        """Get the latest processed frame with annotations"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None


class BlindSpotProcessor:
    """Handles blind spot monitoring using the blindspot.py module"""
    
    def __init__(self, video_source, side='left'):
        self.video_source = video_source
        self.side = side
        self.is_running = False
        self.detection_status = False
        self.vehicle_count = 0
        self.vehicle_distance = None
        self.vehicle_details = []
        self.processed_frame = None
        self.process_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.monitor = None
        
    def start(self):
        """Start blind spot processing in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.process_thread = threading.Thread(target=self._process_video, daemon=True)
            self.process_thread.start()
    
    def stop(self):
        """Stop blind spot processing"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
    
    def _process_video(self):
        """Process video through blind spot monitor"""
        try:
            # Import the BlindSpotMonitor from your blindspot.py
            from BlindSpotMonitoring.SourceCode.blindspot import BlindSpotMonitor
            
            # Initialize monitor with verbose=False to avoid console spam
            self.monitor = BlindSpotMonitor(
                side=self.side,
                model_name="midas_v21_small",
                models_dir="../BlindSpotMonitoring/SourceCode/Model",
                detection_interval=5,
                verbose=False
            )
            
            # Open video
            cap = cv2.VideoCapture(self.video_source)
            
            if not cap.isOpened():
                print(f"Failed to open video source: {self.video_source}")
                self.is_running = False
                return
            
            print(f"✓ Blind spot monitoring started for {self.side} side")
            
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process frame through blind spot monitor
                # draw_overlay=False to get clean video without annotations
                processed_frame, vehicles = self.monitor.process_frame(frame, draw_overlay=False)
                
                # Update detection status
                self.detection_status = len(vehicles) > 0
                self.vehicle_count = len(vehicles)
                self.vehicle_details = vehicles
                
                # Get distance from closest vehicle if available
                if vehicles:
                    # Find vehicle with smallest distance
                    vehicles_with_distance = [v for v in vehicles if v.get('distance')]
                    if vehicles_with_distance:
                        closest = min(vehicles_with_distance, key=lambda v: v['distance'])
                        self.vehicle_distance = closest['distance']
                    else:
                        self.vehicle_distance = None
                else:
                    self.vehicle_distance = None
                
                # Store processed frame in queue (non-blocking)
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        pass
                
                time.sleep(0.03)  # ~30 FPS
            
            cap.release()
            print(f"✓ Blind spot monitoring stopped for {self.side} side")
            
        except ImportError as e:
            print(f"❌ Error importing BlindSpotMonitor: {e}")
            print("   Make sure blindspot.py is in the correct location")
            self.is_running = False
        except Exception as e:
            print(f"❌ Blind spot processing error: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
    
    def get_processed_frame(self):
        """Get the latest processed frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None


class NeuroDriveUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroDrive")
        self.root.geometry("1280x800")
        self.root.configure(bg='#0a0a0a')
        
        # Configuration file path
        self.config_file = "neurodrive_config.json"
        self.load_config()
        
        # Video capture objects
        self.video_captures = [None, None, None, None]
        self.video_labels = []
        self.is_updating = False
        
        # Traffic sign processor (for Front camera)
        self.front_traffic_processor = None
        self.front_traffic_enabled = False
        
        # Traffic sign history for persistence (keep signs visible longer)
        self.traffic_sign_history = []  # List of recent detections
        self.max_sign_history = 3  # Show up to 3 recent signs
        self.sign_display_duration = 5.0  # Keep each sign visible for 5 seconds
        
        # Blind spot processors (for Left and Right cameras)
        self.left_blindspot_processor = None
        self.right_blindspot_processor = None
        self.left_blindspot_enabled = False
        self.right_blindspot_enabled = False
        
        # Frame timing
        self.last_frame_time = [0, 0, 0, 0]
        self.target_fps = 30
        self.frame_delay = 1.0 / self.target_fps
        
        # Update IDs
        self.update_ids = [None, None, None, None]
        
        # Show loading screen
        self.show_loading_screen()
        
    def load_config(self):
        """Load configuration from file or create default"""
        default_config = {
            "Lane Departure Warning": True,
            "Blind Spot Monitoring": True,
            "Traffic Sign Detection": True,
            "Driver Distraction Detection": True,
            "Overtake Assistance": True,
            "Weather/Visibility Adaptation": True,
            "Priority-Based Rules Alert": True,
            "LLM-Based Risk Explanation": True,
            "Driving Style Feedback": True,
            "Forward Collision Warning": True,
            "Pedestrian Intent Prediction": True
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except:
                self.config = default_config
                self.save_config()
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def show_loading_screen(self):
        """Display loading screen with NeuroDrive branding"""
        self.loading_frame = tk.Frame(self.root, bg='#0a0a0a')
        self.loading_frame.pack(fill='both', expand=True)
        
        title_label = tk.Label(
            self.loading_frame,
            text="NeuroDrive",
            font=("Helvetica", 56, "bold"),
            fg="#00a8ff",
            bg="#0a0a0a"
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            self.loading_frame,
            text="Advanced Driver Assistance System",
            font=("Helvetica", 14),
            fg="#666666",
            bg="#0a0a0a"
        )
        subtitle_label.pack()
        
        self.loading_label = tk.Label(
            self.loading_frame,
            text="Initializing System...",
            font=("Helvetica", 12),
            fg="#00a8ff",
            bg="#0a0a0a"
        )
        self.loading_label.pack(pady=20)
        
        progress_frame = tk.Frame(self.loading_frame, bg='#1a1a1a', height=4, width=300)
        progress_frame.pack()
        progress_fill = tk.Frame(progress_frame, bg='#00a8ff', height=4)
        progress_fill.pack(side='left', fill='y')
        
        def animate_progress(width=0):
            if width <= 300:
                progress_fill.config(width=width)
                self.root.after(15, lambda: animate_progress(width + 5))
        
        animate_progress()
        self.root.after(2000, self.show_main_screen)
    
    def show_main_screen(self):
        """Display main interface with 4 video feeds"""
        self.loading_frame.destroy()
        
        self.main_frame = tk.Frame(self.root, bg='#0f0f0f')
        self.main_frame.pack(fill='both', expand=True)
        
        # Top bar
        top_bar = tk.Frame(self.main_frame, bg='#1a1a1a', height=60)
        top_bar.pack(fill='x', side='top')
        top_bar.pack_propagate(False)
        
        title_label = tk.Label(
            top_bar,
            text="NeuroDrive",
            font=("Helvetica", 18, "bold"),
            fg="#00a8ff",
            bg="#1a1a1a"
        )
        title_label.pack(side='left', padx=20)
        
        # Status indicator
        status_frame = tk.Frame(top_bar, bg='#1a1a1a')
        status_frame.pack(side='left', padx=20)
        
        status_dot = tk.Label(
            status_frame,
            text="●",
            font=("Helvetica", 16),
            fg="#00ff66",
            bg="#1a1a1a"
        )
        status_dot.pack(side='left')
        
        status_text = tk.Label(
            status_frame,
            text="System Active",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#1a1a1a"
        )
        status_text.pack(side='left', padx=5)
        
        # TRAFFIC SIGN INDICATOR (Now shows multiple recent signs)
        self.traffic_sign_frame = tk.Frame(top_bar, bg='#1a1a1a')
        self.traffic_sign_frame.pack(side='left', padx=15)
        
        self.traffic_sign_indicator = tk.Label(
            self.traffic_sign_frame,
            text="🚦",
            font=("Helvetica", 18),
            fg="#00ff66",
            bg="#1a1a1a"
        )
        self.traffic_sign_indicator.pack(side='left')
        
        # Container for multiple sign labels
        self.traffic_signs_container = tk.Frame(self.traffic_sign_frame, bg='#1a1a1a')
        self.traffic_signs_container.pack(side='left', padx=5)
        
        self.traffic_sign_labels = []
        for i in range(3):  # Support up to 3 concurrent signs
            label = tk.Label(
                self.traffic_signs_container,
                text="",
                font=("Helvetica", 8, "bold"),
                fg="#888888",
                bg="#1a1a1a"
            )
            label.pack(anchor='w')
            self.traffic_sign_labels.append(label)
        
        # LEFT BLIND SPOT INDICATOR
        self.left_blindspot_frame = tk.Frame(top_bar, bg='#1a1a1a')
        self.left_blindspot_frame.pack(side='left', padx=15)
        
        self.left_blindspot_indicator = tk.Label(
            self.left_blindspot_frame,
            text="●",
            font=("Helvetica", 20),
            fg="#00ff66",
            bg="#1a1a1a"
        )
        self.left_blindspot_indicator.pack(side='left')
        
        self.left_blindspot_label = tk.Label(
            self.left_blindspot_frame,
            text="LEFT: CLEAR",
            font=("Helvetica", 9, "bold"),
            fg="#00ff66",
            bg="#1a1a1a"
        )
        self.left_blindspot_label.pack(side='left', padx=5)
        
        self.left_blindspot_distance = tk.Label(
            self.left_blindspot_frame,
            text="",
            font=("Helvetica", 8),
            fg="#888888",
            bg="#1a1a1a"
        )
        self.left_blindspot_distance.pack(side='left', padx=2)
        
        # RIGHT BLIND SPOT INDICATOR
        self.right_blindspot_frame = tk.Frame(top_bar, bg='#1a1a1a')
        self.right_blindspot_frame.pack(side='left', padx=15)
        
        self.right_blindspot_indicator = tk.Label(
            self.right_blindspot_frame,
            text="●",
            font=("Helvetica", 20),
            fg="#00ff66",
            bg="#1a1a1a"
        )
        self.right_blindspot_indicator.pack(side='left')
        
        self.right_blindspot_label = tk.Label(
            self.right_blindspot_frame,
            text="RIGHT: CLEAR",
            font=("Helvetica", 9, "bold"),
            fg="#00ff66",
            bg="#1a1a1a"
        )
        self.right_blindspot_label.pack(side='left', padx=5)
        
        self.right_blindspot_distance = tk.Label(
            self.right_blindspot_frame,
            text="",
            font=("Helvetica", 8),
            fg="#888888",
            bg="#1a1a1a"
        )
        self.right_blindspot_distance.pack(side='left', padx=2)
        
        # Settings button
        settings_btn = tk.Button(
            top_bar,
            text="⚙  Settings",
            font=("Helvetica", 11),
            bg="#00a8ff",
            fg="#000000",
            activebackground="#0088cc",
            activeforeground="#000000",
            relief='flat',
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.show_settings
        )
        settings_btn.pack(side='right', padx=20, pady=10)
        
        # Video feeds container
        video_container = tk.Frame(self.main_frame, bg='#0f0f0f')
        video_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Create 4 video feed placeholders
        self.video_frames = []
        feed_labels = ["Front Camera (Traffic Signs)", "Rear Camera", "Left Camera (Blind Spot)", "Right Camera (Blind Spot)"]
        
        for i in range(4):
            row = i // 2
            col = i % 2
            
            outer_frame = tk.Frame(video_container, bg='#00a8ff', relief='flat')
            outer_frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
            
            feed_frame = tk.Frame(outer_frame, bg='#1a1a1a', relief='flat')
            feed_frame.pack(fill='both', expand=True, padx=2, pady=2)
            
            feed_header = tk.Frame(feed_frame, bg='#252525', height=35)
            feed_header.pack(fill='x', side='top')
            feed_header.pack_propagate(False)
            
            feed_title = tk.Label(
                feed_header,
                text=feed_labels[i],
                font=("Helvetica", 10, "bold"),
                fg="#00a8ff",
                bg="#252525"
            )
            feed_title.pack(side='left', padx=10, pady=5)
            
            # Add indicator for special feeds
            if i == 0:  # Front camera
                ts_indicator = tk.Label(
                    feed_header,
                    text="[TRAFFIC SIGNS]",
                    font=("Helvetica", 8, "bold"),
                    fg="#ffaa00",
                    bg="#252525"
                )
                ts_indicator.pack(side='right', padx=10)
            elif i == 2:  # Left camera
                bs_indicator = tk.Label(
                    feed_header,
                    text="[LEFT BLIND SPOT]",
                    font=("Helvetica", 8, "bold"),
                    fg="#ffaa00",
                    bg="#252525"
                )
                bs_indicator.pack(side='right', padx=10)
            elif i == 3:  # Right camera
                bs_indicator = tk.Label(
                    feed_header,
                    text="[RIGHT BLIND SPOT]",
                    font=("Helvetica", 8, "bold"),
                    fg="#ffaa00",
                    bg="#252525"
                )
                bs_indicator.pack(side='right', padx=10)
            
            content_frame = tk.Frame(feed_frame, bg='#1a1a1a')
            content_frame.pack(fill='both', expand=True)
            
            feed_label = tk.Label(
                content_frame,
                text=f"📹\n\nClick to maximize",
                font=("Helvetica", 12),
                fg="#555555",
                bg="#1a1a1a",
                cursor="hand2"
            )
            feed_label.pack(expand=True, fill='both')
            feed_label.bind("<Button-1>", lambda e, idx=i: self.maximize_feed(idx))
            
            self.video_labels.append(feed_label)
            
            def on_enter(e, label=feed_label):
                label.config(fg="#00a8ff")
            
            def on_leave(e, label=feed_label):
                label.config(fg="#555555")
            
            feed_label.bind("<Enter>", on_enter)
            feed_label.bind("<Leave>", on_leave)
            
            self.video_frames.append(outer_frame)
        
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_rowconfigure(1, weight=1)
        video_container.grid_columnconfigure(0, weight=1)
        video_container.grid_columnconfigure(1, weight=1)
        
        # Alert panel
        alert_frame = tk.Frame(self.main_frame, bg='#1a1a1a', height=140)
        alert_frame.pack(fill='x', side='bottom', padx=15, pady=(0, 15))
        alert_frame.pack_propagate(False)
        
        alert_header = tk.Frame(alert_frame, bg='#252525', height=35)
        alert_header.pack(fill='x')
        alert_header.pack_propagate(False)
        
        alert_title = tk.Label(
            alert_header,
            text="🔔  ALERTS & NOTIFICATIONS",
            font=("Helvetica", 11, "bold"),
            fg="#00a8ff",
            bg="#252525"
        )
        alert_title.pack(anchor='w', padx=15, pady=8)
        
        self.alert_text = tk.Text(
            alert_frame,
            font=("Consolas", 10),
            fg="#cccccc",
            bg='#1a1a1a',
            height=5,
            wrap='word',
            relief='flat',
            padx=10,
            pady=5
        )
        self.alert_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.alert_text.insert('1.0', "✓ System initialized successfully\n✓ Initializing camera feeds...")
        self.alert_text.config(state='disabled')
        
        # Start video feeds
        self.start_video_feeds()
        
        # Start indicator updates
        self.update_traffic_sign_indicator()
        self.update_blindspot_indicators()
    
    def start_video_feeds(self):
        """Initialize video captures for all camera feeds"""
        # Camera sources - UPDATE THESE PATHS
        camera_sources = [
            "../TrafficSignDetection/Videos/trafficsign1.mp4",  # Front - with traffic sign detection
            "",  # Rear
            "../BlindSpotMonitoring/Pictures/leftblindspot.mp4",  # Left - with blind spot
            "../BlindSpotMonitoring/Pictures/rightblindspot.mp4"   # Right - with blind spot
        ]
        
        # Initialize traffic sign detection on front camera (feed 0)
        if self.config.get("Traffic Sign Detection", True):
            if camera_sources[0] and os.path.exists(camera_sources[0]):
                self.front_traffic_enabled = True
                self.front_traffic_processor = TrafficSignProcessor(
                    video_source=camera_sources[0],
                    model_path="../TrafficSignDetection/Models/gtsrb_kaggle_model.h5",
                    alert_threshold=0.90
                )
                self.front_traffic_processor.start()
                self.add_alert("✓ Traffic Sign Detection initialized on front camera")
            else:
                self.add_alert(f"⚠ Front video file not found: {camera_sources[0]}")
        
        # Initialize blind spot monitoring
        if self.config.get("Blind Spot Monitoring", True):
            # Left blind spot (feed index 2)
            if camera_sources[2] and os.path.exists(camera_sources[2]):
                self.left_blindspot_enabled = True
                self.left_blindspot_processor = BlindSpotProcessor(
                    video_source=camera_sources[2],
                    side='left'
                )
                self.left_blindspot_processor.start()
                self.add_alert("✓ Left Blind Spot Monitoring initialized")
            else:
                self.add_alert(f"⚠ Left video file not found: {camera_sources[2]}")
            
            # Right blind spot (feed index 3)
            if camera_sources[3] and os.path.exists(camera_sources[3]):
                self.right_blindspot_enabled = True
                self.right_blindspot_processor = BlindSpotProcessor(
                    video_source=camera_sources[3],
                    side='right'
                )
                self.right_blindspot_processor.start()
                self.add_alert("✓ Right Blind Spot Monitoring initialized")
            else:
                self.add_alert(f"⚠ Right video file not found: {camera_sources[3]}")
        
        # Initialize other video captures (feed 1 - rear)
        for i, source in enumerate(camera_sources):
            # Skip feeds 0, 2, 3 as they're handled by special processors
            if i in [0, 2, 3]:
                continue
                
            if source and os.path.exists(str(source)):
                try:
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        self.video_captures[i] = cap
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps > 0:
                            self.frame_delay = 1.0 / min(fps, 30)
                        print(f"✓ Camera {i} initialized")
                    else:
                        print(f"✗ Failed to open camera {i}")
                except Exception as e:
                    print(f"✗ Error initializing camera {i}: {e}")
        
        # Start updating video frames
        self.is_updating = True
        for i in range(4):
            if i == 0 and self.front_traffic_enabled:
                self.update_traffic_sign_feed(i)
            elif i == 2 and self.left_blindspot_enabled:
                self.update_blindspot_feed(i, 'left')
            elif i == 3 and self.right_blindspot_enabled:
                self.update_blindspot_feed(i, 'right')
            elif self.video_captures[i] is not None:
                self.update_video_frame(i)
    
    def update_traffic_sign_feed(self, feed_index):
        """Update the traffic sign detection feed"""
        if not self.is_updating:
            return
        
        processor = self.front_traffic_processor
        
        if not self.front_traffic_enabled or not processor:
            return
        
        if processor.is_running:
            processed_frame = processor.get_processed_frame()
            
            if processed_frame is not None and feed_index < len(self.video_labels):
                frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                label_width = self.video_labels[feed_index].winfo_width()
                label_height = self.video_labels[feed_index].winfo_height()
                
                if label_width <= 1 or label_height <= 1:
                    label_width = 400
                    label_height = 280
                
                h, w = frame.shape[:2]
                aspect = w / h
                
                if label_width / label_height > aspect:
                    new_h = label_height
                    new_w = int(aspect * new_h)
                else:
                    new_w = label_width
                    new_h = int(new_w / aspect)
                
                frame = cv2.resize(frame, (new_w, new_h))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                if hasattr(self.video_labels[feed_index], 'imgtk'):
                    old_img = self.video_labels[feed_index].imgtk
                    del old_img
                
                self.video_labels[feed_index].imgtk = imgtk
                self.video_labels[feed_index].configure(image=imgtk, text="")
        
        if self.is_updating:
            self.root.after(30, lambda: self.update_traffic_sign_feed(feed_index))
    
    def update_traffic_sign_indicator(self):
        """Update traffic sign indicator in top bar with persistence"""
        if not hasattr(self, 'traffic_sign_indicator'):
            return
        
        current_time = time.time()
        
        # Add new detections to history
        if self.front_traffic_processor and self.front_traffic_processor.is_running:
            if self.front_traffic_processor.detection_status:
                sign_name = self.front_traffic_processor.current_sign
                confidence = self.front_traffic_processor.sign_confidence
                
                # Check if this is a new detection (not already in recent history)
                is_new_sign = True
                for sign_entry in self.traffic_sign_history:
                    if sign_entry['name'] == sign_name and (current_time - sign_entry['last_seen']) < 1.0:
                        # Update existing entry
                        sign_entry['last_seen'] = current_time
                        sign_entry['confidence'] = max(sign_entry['confidence'], confidence)
                        is_new_sign = False
                        break
                
                if is_new_sign:
                    # Add new sign to history
                    self.traffic_sign_history.append({
                        'name': sign_name,
                        'confidence': confidence,
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'alerted': False
                    })
        
        # Remove old signs from history (older than display duration)
        self.traffic_sign_history = [
            sign for sign in self.traffic_sign_history 
            if (current_time - sign['last_seen']) < self.sign_display_duration
        ]
        
        # Sort by confidence (highest first) and limit to max_sign_history
        self.traffic_sign_history.sort(key=lambda x: x['confidence'], reverse=True)
        self.traffic_sign_history = self.traffic_sign_history[:self.max_sign_history]
        
        # Update display
        if self.traffic_sign_history:
            # Determine indicator color based on highest confidence
            highest_confidence = max(sign['confidence'] for sign in self.traffic_sign_history)
            
            if highest_confidence >= 0.90:
                # High confidence - RED/YELLOW flashing
                flash_state = (int(current_time * 3) % 2 == 0)
                indicator_color = "#ff0000" if flash_state else "#ffaa00"
                self.traffic_sign_indicator.config(text="⚠", fg=indicator_color)
            elif highest_confidence >= 0.70:
                # Medium-high confidence - YELLOW
                self.traffic_sign_indicator.config(text="🚦", fg="#ffaa00")
            else:
                # Medium confidence - LIGHT YELLOW
                self.traffic_sign_indicator.config(text="🚦", fg="#cccc00")
            
            # Update each label with sign info
            for i, label in enumerate(self.traffic_sign_labels):
                if i < len(self.traffic_sign_history):
                    sign = self.traffic_sign_history[i]
                    
                    # Calculate age for fade effect
                    age = current_time - sign['last_seen']
                    
                    # Determine color based on confidence and age
                    if sign['confidence'] >= 0.90 and age < 0.5:
                        color = "#ff0000"  # Red for high confidence, recent
                        prefix = "⚠ "
                    elif sign['confidence'] >= 0.90:
                        color = "#ff6666"  # Lighter red as it ages
                        prefix = "⚠ "
                    elif sign['confidence'] >= 0.70:
                        color = "#ffaa00"  # Yellow for medium confidence
                        prefix = ""
                    else:
                        color = "#cccc00"  # Light yellow for lower confidence
                        prefix = ""
                    
                    # Shorten long sign names
                    display_name = sign['name']
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                    
                    label.config(
                        text=f"{prefix}{display_name} ({sign['confidence']:.0%})",
                        fg=color
                    )
                    
                    # Generate alert for high confidence signs (only once per sign)
                    if sign['confidence'] >= 0.90 and not sign['alerted']:
                        sign['alerted'] = True
                        self.add_alert(f"🚨 TRAFFIC SIGN: {sign['name']} ({sign['confidence']:.1%})")
                else:
                    label.config(text="", fg="#888888")
        else:
            # No signs detected
            self.traffic_sign_indicator.config(text="🚦", fg="#00ff66")
            for label in self.traffic_sign_labels:
                label.config(text="", fg="#888888")
            # Show "NO SIGNS" only in first label
            self.traffic_sign_labels[0].config(text="NO SIGNS", fg="#888888")
        
        self.root.after(100, self.update_traffic_sign_indicator)
    
    def update_blindspot_feed(self, feed_index, side):
        """Update a blind spot monitored feed"""
        if not self.is_updating:
            return
        
        processor = self.left_blindspot_processor if side == 'left' else self.right_blindspot_processor
        enabled = self.left_blindspot_enabled if side == 'left' else self.right_blindspot_enabled
        
        if not enabled or not processor:
            return
        
        if processor.is_running:
            processed_frame = processor.get_processed_frame()
            
            if processed_frame is not None and feed_index < len(self.video_labels):
                frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                label_width = self.video_labels[feed_index].winfo_width()
                label_height = self.video_labels[feed_index].winfo_height()
                
                if label_width <= 1 or label_height <= 1:
                    label_width = 400
                    label_height = 280
                
                h, w = frame.shape[:2]
                aspect = w / h
                
                if label_width / label_height > aspect:
                    new_h = label_height
                    new_w = int(aspect * new_h)
                else:
                    new_w = label_width
                    new_h = int(new_w / aspect)
                
                frame = cv2.resize(frame, (new_w, new_h))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                if hasattr(self.video_labels[feed_index], 'imgtk'):
                    old_img = self.video_labels[feed_index].imgtk
                    del old_img
                
                self.video_labels[feed_index].imgtk = imgtk
                self.video_labels[feed_index].configure(image=imgtk, text="")
        
        if self.is_updating:
            self.root.after(30, lambda: self.update_blindspot_feed(feed_index, side))
    
    def update_blindspot_indicators(self):
        """Update both blind spot indicators in top bar"""
        if not hasattr(self, 'left_blindspot_indicator'):
            return
        
        # Update LEFT indicator
        if self.left_blindspot_processor and self.left_blindspot_processor.is_running:
            if self.left_blindspot_processor.detection_status:
                # Vehicle detected - RED with flashing
                flash_state = (int(time.time() * 2) % 2 == 0)
                color = "#ff0000" if flash_state else "#ff6666"
                
                self.left_blindspot_indicator.config(fg=color)
                self.left_blindspot_label.config(
                    text=f"⚠ LEFT ({self.left_blindspot_processor.vehicle_count})",
                    fg=color
                )
                
                if self.left_blindspot_processor.vehicle_distance:
                    distance = self.left_blindspot_processor.vehicle_distance
                    self.left_blindspot_distance.config(
                        text=f"{distance:.1f}m",
                        fg="#ff6666"
                    )
                else:
                    self.left_blindspot_distance.config(text="")
                
                # Alert every 2 seconds
                current_time = time.time()
                if not hasattr(self, 'last_left_alert_time') or current_time - self.last_left_alert_time > 2:
                    self.last_left_alert_time = current_time
                    self.add_alert(f"⚠ WARNING: {self.left_blindspot_processor.vehicle_count} vehicle(s) in LEFT blind spot!")
            else:
                # No vehicle - GREEN
                self.left_blindspot_indicator.config(fg="#00ff66")
                self.left_blindspot_label.config(
                    text="LEFT: CLEAR ✓",
                    fg="#00ff66"
                )
                self.left_blindspot_distance.config(text="")
        
        # Update RIGHT indicator
        if self.right_blindspot_processor and self.right_blindspot_processor.is_running:
            if self.right_blindspot_processor.detection_status:
                # Vehicle detected - RED with flashing
                flash_state = (int(time.time() * 2) % 2 == 0)
                color = "#ff0000" if flash_state else "#ff6666"
                
                self.right_blindspot_indicator.config(fg=color)
                self.right_blindspot_label.config(
                    text=f"⚠ RIGHT ({self.right_blindspot_processor.vehicle_count})",
                    fg=color
                )
                
                if self.right_blindspot_processor.vehicle_distance:
                    distance = self.right_blindspot_processor.vehicle_distance
                    self.right_blindspot_distance.config(
                        text=f"{distance:.1f}m",
                        fg="#ff6666"
                    )
                else:
                    self.right_blindspot_distance.config(text="")
                
                # Alert every 2 seconds
                current_time = time.time()
                if not hasattr(self, 'last_right_alert_time') or current_time - self.last_right_alert_time > 2:
                    self.last_right_alert_time = current_time
                    self.add_alert(f"⚠ WARNING: {self.right_blindspot_processor.vehicle_count} vehicle(s) in RIGHT blind spot!")
            else:
                # No vehicle - GREEN
                self.right_blindspot_indicator.config(fg="#00ff66")
                self.right_blindspot_label.config(
                    text="RIGHT: CLEAR ✓",
                    fg="#00ff66"
                )
                self.right_blindspot_distance.config(text="")
        
        self.root.after(100, self.update_blindspot_indicators)
    
    def add_alert(self, message):
        """Add alert message to alert panel"""
        if hasattr(self, 'alert_text'):
            self.alert_text.config(state='normal')
            current_text = self.alert_text.get('1.0', 'end-1c')
            lines = current_text.split('\n')
            
            if len(lines) >= 3:
                lines = lines[-2:]
            
            timestamp = time.strftime('%H:%M:%S')
            lines.append(f"[{timestamp}] {message}")
            self.alert_text.delete('1.0', 'end')
            self.alert_text.insert('1.0', '\n'.join(lines))
            self.alert_text.config(state='disabled')
            self.alert_text.see('end')
    
    def stop_video_feeds(self):
        """Stop all video feeds and release resources"""
        self.is_updating = False
        
        if self.front_traffic_processor:
            self.front_traffic_processor.stop()
        
        if self.left_blindspot_processor:
            self.left_blindspot_processor.stop()
        
        if self.right_blindspot_processor:
            self.right_blindspot_processor.stop()
        
        for i in range(4):
            if self.update_ids[i] is not None:
                self.root.after_cancel(self.update_ids[i])
                self.update_ids[i] = None
        
        for cap in self.video_captures:
            if cap is not None:
                cap.release()
    
    def update_video_frame(self, feed_index):
        """Update a single video frame"""
        if not self.is_updating or feed_index >= len(self.video_captures):
            return
        
        if feed_index == 0 and self.front_traffic_enabled:
            return
        if feed_index == 2 and self.left_blindspot_enabled:
            return
        if feed_index == 3 and self.right_blindspot_enabled:
            return
        
        cap = self.video_captures[feed_index]
        
        if cap is not None and cap.isOpened() and feed_index < len(self.video_labels):
            current_time = time.time()
            time_since_last = current_time - self.last_frame_time[feed_index]
            
            if time_since_last >= self.frame_delay:
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    label_width = self.video_labels[feed_index].winfo_width()
                    label_height = self.video_labels[feed_index].winfo_height()
                    
                    if label_width <= 1 or label_height <= 1:
                        label_width = 400
                        label_height = 280
                    
                    h, w = frame.shape[:2]
                    aspect = w / h
                    
                    if label_width / label_height > aspect:
                        new_h = label_height
                        new_w = int(aspect * new_h)
                    else:
                        new_w = label_width
                        new_h = int(new_w / aspect)
                    
                    frame = cv2.resize(frame, (new_w, new_h))
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    if hasattr(self.video_labels[feed_index], 'imgtk'):
                        old_img = self.video_labels[feed_index].imgtk
                        del old_img
                    
                    self.video_labels[feed_index].imgtk = imgtk
                    self.video_labels[feed_index].configure(image=imgtk, text="")
                    self.last_frame_time[feed_index] = current_time
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            elapsed = time.time() - current_time
            next_delay = max(1, int((self.frame_delay - elapsed) * 1000))
            
            if self.is_updating:
                self.update_ids[feed_index] = self.root.after(next_delay, lambda: self.update_video_frame(feed_index))

    def maximize_feed(self, feed_index):
        """Maximize a specific video feed"""
        feed_labels = ["Front Camera (Traffic Signs)", "Rear Camera", "Left Camera (Blind Spot)", "Right Camera (Blind Spot)"]
        
        self.is_updating = False
        for i in range(4):
            if self.update_ids[i] is not None:
                self.root.after_cancel(self.update_ids[i])
                self.update_ids[i] = None
        
        self.main_frame.pack_forget()
        
        self.maximized_frame = tk.Frame(self.root, bg='#0f0f0f')
        self.maximized_frame.pack(fill='both', expand=True)
        
        top_bar = tk.Frame(self.maximized_frame, bg='#1a1a1a', height=60)
        top_bar.pack(fill='x', side='top')
        top_bar.pack_propagate(False)
        
        back_btn = tk.Button(
            top_bar,
            text="←  Back",
            font=("Helvetica", 11),
            bg="#00a8ff",
            fg="#000000",
            activebackground="#0088cc",
            activeforeground="#000000",
            relief='flat',
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.back_to_main
        )
        back_btn.pack(side='left', padx=20, pady=10)
        
        feed_title = tk.Label(
            top_bar,
            text=feed_labels[feed_index],
            font=("Helvetica", 16, "bold"),
            fg="#00a8ff",
            bg="#1a1a1a"
        )
        feed_title.pack(side='left', padx=20)
        
        video_container = tk.Frame(self.maximized_frame, bg='#00a8ff')
        video_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.maximized_video_label = tk.Label(
            video_container,
            text=f"📹\n\n{feed_labels[feed_index]}\n\nLoading...",
            font=("Helvetica", 18),
            fg="#555555",
            bg="#1a1a1a"
        )
        self.maximized_video_label.pack(fill='both', expand=True, padx=2, pady=2)
        
        self.current_maximized_feed = feed_index
        self.maximized_last_frame_time = 0
        self.maximized_update_id = None
        
        info_frame = tk.Frame(self.maximized_frame, bg='#1a1a1a', height=100)
        info_frame.pack(fill='x', side='bottom', padx=15, pady=(0, 15))
        info_frame.pack_propagate(False)
        
        if feed_index == 0:
            info_text = "Traffic Sign Detection Active"
        elif feed_index == 2:
            info_text = "Left Blind Spot Monitoring Active"
        elif feed_index == 3:
            info_text = "Right Blind Spot Monitoring Active"
        else:
            info_text = "Camera Feed"
        
        info_label = tk.Label(
            info_frame,
            text=f"{info_text}\nResolution: 1920x1080 | FPS: 30 | Status: Active",
            font=("Consolas", 10),
            fg="#888888",
            bg="#1a1a1a",
            justify='left'
        )
        info_label.pack(anchor='w', padx=15, pady=10)
        
        self.is_updating = True
        self.update_maximized_feed()
    
    def update_maximized_feed(self):
        """Update the maximized video feed"""
        if not self.is_updating or not hasattr(self, 'maximized_video_label'):
            return
        
        feed_index = self.current_maximized_feed
        
        # Handle front traffic sign feed
        if feed_index == 0 and self.front_traffic_enabled and self.front_traffic_processor:
            processed_frame = self.front_traffic_processor.get_processed_frame()
            
            if processed_frame is not None:
                frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                label_width = self.maximized_video_label.winfo_width()
                label_height = self.maximized_video_label.winfo_height()
                
                if label_width <= 1 or label_height <= 1:
                    label_width = 1000
                    label_height = 600
                
                h, w = frame.shape[:2]
                aspect = w / h
                
                if label_width / label_height > aspect:
                    new_h = label_height
                    new_w = int(aspect * new_h)
                else:
                    new_w = label_width
                    new_h = int(new_w / aspect)
                
                frame = cv2.resize(frame, (new_w, new_h))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                if hasattr(self.maximized_video_label, 'imgtk'):
                    old_img = self.maximized_video_label.imgtk
                    del old_img
                
                self.maximized_video_label.imgtk = imgtk
                self.maximized_video_label.configure(image=imgtk, text="")
            
            if self.is_updating and hasattr(self, 'maximized_video_label'):
                self.maximized_update_id = self.root.after(30, self.update_maximized_feed)
            return
        
        # Handle left blind spot feed
        if feed_index == 2 and self.left_blindspot_enabled and self.left_blindspot_processor:
            processed_frame = self.left_blindspot_processor.get_processed_frame()
            
            if processed_frame is not None:
                frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                label_width = self.maximized_video_label.winfo_width()
                label_height = self.maximized_video_label.winfo_height()
                
                if label_width <= 1 or label_height <= 1:
                    label_width = 1000
                    label_height = 600
                
                h, w = frame.shape[:2]
                aspect = w / h
                
                if label_width / label_height > aspect:
                    new_h = label_height
                    new_w = int(aspect * new_h)
                else:
                    new_w = label_width
                    new_h = int(new_w / aspect)
                
                frame = cv2.resize(frame, (new_w, new_h))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                if hasattr(self.maximized_video_label, 'imgtk'):
                    old_img = self.maximized_video_label.imgtk
                    del old_img
                
                self.maximized_video_label.imgtk = imgtk
                self.maximized_video_label.configure(image=imgtk, text="")
            
            if self.is_updating and hasattr(self, 'maximized_video_label'):
                self.maximized_update_id = self.root.after(30, self.update_maximized_feed)
            return
        
        # Handle right blind spot feed
        if feed_index == 3 and self.right_blindspot_enabled and self.right_blindspot_processor:
            processed_frame = self.right_blindspot_processor.get_processed_frame()
            
            if processed_frame is not None:
                frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                label_width = self.maximized_video_label.winfo_width()
                label_height = self.maximized_video_label.winfo_height()
                
                if label_width <= 1 or label_height <= 1:
                    label_width = 1000
                    label_height = 600
                
                h, w = frame.shape[:2]
                aspect = w / h
                
                if label_width / label_height > aspect:
                    new_h = label_height
                    new_w = int(aspect * new_h)
                else:
                    new_w = label_width
                    new_h = int(new_w / aspect)
                
                frame = cv2.resize(frame, (new_w, new_h))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                if hasattr(self.maximized_video_label, 'imgtk'):
                    old_img = self.maximized_video_label.imgtk
                    del old_img
                
                self.maximized_video_label.imgtk = imgtk
                self.maximized_video_label.configure(image=imgtk, text="")
            
            if self.is_updating and hasattr(self, 'maximized_video_label'):
                self.maximized_update_id = self.root.after(30, self.update_maximized_feed)
            return
        
        # Normal video feed handling
        cap = self.video_captures[feed_index]
        
        if cap is not None and cap.isOpened():
            current_time = time.time()
            time_since_last = current_time - self.maximized_last_frame_time
            
            if time_since_last >= self.frame_delay:
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    label_width = self.maximized_video_label.winfo_width()
                    label_height = self.maximized_video_label.winfo_height()
                    
                    if label_width <= 1 or label_height <= 1:
                        label_width = 1000
                        label_height = 600
                    
                    h, w = frame.shape[:2]
                    aspect = w / h
                    
                    if label_width / label_height > aspect:
                        new_h = label_height
                        new_w = int(aspect * new_h)
                    else:
                        new_w = label_width
                        new_h = int(new_w / aspect)
                    
                    frame = cv2.resize(frame, (new_w, new_h))
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    if hasattr(self.maximized_video_label, 'imgtk'):
                        old_img = self.maximized_video_label.imgtk
                        del old_img
                    
                    self.maximized_video_label.imgtk = imgtk
                    self.maximized_video_label.configure(image=imgtk, text="")
                    self.maximized_last_frame_time = current_time
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            elapsed = time.time() - current_time
            next_delay = max(1, int((self.frame_delay - elapsed) * 1000))
            
            if self.is_updating and hasattr(self, 'maximized_video_label'):
                self.maximized_update_id = self.root.after(next_delay, self.update_maximized_feed)
    
    def back_to_main(self):
        """Return to main 4-feed view"""
        self.is_updating = False
        if hasattr(self, 'maximized_update_id') and self.maximized_update_id is not None:
            self.root.after_cancel(self.maximized_update_id)
            self.maximized_update_id = None
        
        self.current_maximized_feed = None
        
        if hasattr(self, 'maximized_frame'):
            self.maximized_frame.destroy()
        
        self.main_frame.pack(fill='both', expand=True)
        
        # Restart the 4-feed updates
        self.is_updating = True
        self.last_frame_time = [0, 0, 0, 0]
        for i in range(4):
            if i == 0 and self.front_traffic_enabled:
                self.update_traffic_sign_feed(i)
            elif i == 2 and self.left_blindspot_enabled:
                self.update_blindspot_feed(i, 'left')
            elif i == 3 and self.right_blindspot_enabled:
                self.update_blindspot_feed(i, 'right')
            elif self.video_captures[i] is not None:
                self.update_video_frame(i)
    
    def show_settings(self):
        """Display settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings - NeuroDrive")
        settings_window.geometry("550x750")
        settings_window.configure(bg='#0f0f0f')
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Header
        header = tk.Frame(settings_window, bg='#1a1a1a', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="Alert Module Configuration",
            font=("Helvetica", 18, "bold"),
            fg="#00a8ff",
            bg="#1a1a1a"
        )
        title.pack(pady=15)
        
        subtitle = tk.Label(
            header,
            text="Enable or disable specific monitoring modules",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#1a1a1a"
        )
        subtitle.pack()
        
        # Scrollable frame
        canvas = tk.Canvas(settings_window, bg='#0f0f0f', highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#0f0f0f')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Vertical.TScrollbar", 
                       background="#2a2a2a",
                       troughcolor="#1a1a1a",
                       bordercolor="#1a1a1a",
                       arrowcolor="#00a8ff")
        
        # Create checkboxes
        self.check_vars = {}
        for i, module in enumerate(self.config.keys()):
            var = tk.BooleanVar(value=self.config[module])
            self.check_vars[module] = var
            
            module_frame = tk.Frame(scrollable_frame, bg='#1a1a1a', relief='flat')
            module_frame.pack(fill='x', padx=20, pady=5)
            
            cb = tk.Checkbutton(
                module_frame,
                text=module,
                variable=var,
                font=("Helvetica", 11),
                fg="#ffffff",
                bg="#1a1a1a",
                selectcolor="#2a2a2a",
                activebackground="#1a1a1a",
                activeforeground="#00a8ff",
                cursor="hand2",
                relief='flat'
            )
            cb.pack(anchor='w', padx=15, pady=10)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Button frame
        button_frame = tk.Frame(settings_window, bg='#0f0f0f')
        button_frame.pack(fill='x', pady=20)
        
        save_btn = tk.Button(
            button_frame,
            text="Save Configuration",
            font=("Helvetica", 12, "bold"),
            bg="#00a8ff",
            fg="#000000",
            activebackground="#0088cc",
            activeforeground="#000000",
            relief='flat',
            padx=40,
            pady=12,
            cursor="hand2",
            command=lambda: self.save_settings(settings_window)
        )
        save_btn.pack(side='left', padx=(40, 10))
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            font=("Helvetica", 12),
            bg="#2a2a2a",
            fg="#cccccc",
            activebackground="#3a3a3a",
            activeforeground="#ffffff",
            relief='flat',
            padx=40,
            pady=12,
            cursor="hand2",
            command=settings_window.destroy
        )
        cancel_btn.pack(side='left', padx=10)
    
    def save_settings(self, window):
        """Save settings and close window"""
        for module, var in self.check_vars.items():
            self.config[module] = var.get()
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            if hasattr(self, 'alert_text'):
                self.add_alert("✓ Configuration saved successfully")
        except Exception as e:
            print(f"Error saving config: {e}")
        
        window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroDriveUI(root)
    
    def on_closing():
        app.stop_video_feeds()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()