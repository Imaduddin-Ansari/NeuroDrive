# Real-Time Facial Emotion Detection

A Python-based real-time facial emotion recognition system that uses computer vision and facial landmark detection to analyze emotions from webcam feed.

## Features

- **Real-time emotion detection** from webcam feed
- **7 emotion categories**: Happy, Sad, Surprised, Angry, Neutral, Disgusted, Fearful
- **Facial landmark visualization** with 68-point face detection
- **Automatic calibration** for personalized baseline measurements
- **Emotion history tracking** and data export
- **Live emotion scoring** with confidence percentages
- **Mirror mode** for natural user experience

## Demo

The system displays:
- Live webcam feed with facial landmarks
- Top 3 detected emotions with confidence percentages
- Real-time facial feature measurements (EAR, MAR, curvature, etc.)
- Emotion analysis metrics

## Requirements

### Dependencies

```bash
pip install opencv-python numpy dlib scipy matplotlib
```

### Required Files

You need to download the dlib facial landmark predictor:

1. Download `shape_predictor_68_face_landmarks.dat.bz2` from:
   ```
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   ```

2. Extract the `.dat` file and place it in:
   - Same directory as the script, OR
   - Parent directory of the script

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/emotion-detection.git
   cd emotion-detection
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the facial landmark predictor:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

## Usage

Run the emotion detection system:

```bash
python emotion_analyzer.py
```

### Controls

- **Look neutral** during the first few seconds for calibration
- **Press 'q'** to quit the application
- **Press 's'** to save emotion data to file

## How It Works

### Emotion Detection Algorithm

The system analyzes facial features to determine emotions:

1. **Face Detection**: Uses dlib's frontal face detector
2. **Landmark Detection**: Identifies 68 facial landmarks
3. **Feature Extraction**: Calculates various facial metrics:
   - Eye Aspect Ratio (EAR) for eye openness
   - Mouth Aspect Ratio (MAR) for mouth openness
   - Mouth curvature for smile/frown detection
   - Eyebrow angle and position
   - Nose flare detection
   - Mouth corner direction

4. **Emotion Classification**: Maps features to emotions:
   - **Happy**: Positive mouth curvature + normal eye opening
   - **Sad**: Negative mouth curvature + droopy eyes
   - **Surprised**: Wide eyes OR open mouth + raised eyebrows
   - **Angry**: Furrowed brows + tense features + nose flare
   - **Disgusted**: Nose wrinkle + slight frown
   - **Fearful**: Wide eyes + open mouth + raised eyebrows
   - **Neutral**: Default when no strong emotion detected

### Calibration Process

The system automatically calibrates during the first 15 frames to establish personal baselines for:
- Eye aspect ratio
- Mouth aspect ratio
- Mouth width
- Eyebrow positioning

## Output Files

When you press 's', the system generates an emotion report:
- Filename: `emotion_data_YYYYMMDD_HHMMSS.txt`
- Contains emotion distribution statistics
- Frame-by-frame analysis summary

## Technical Details

### Key Classes and Methods

- `EmotionAnalyzer`: Main class handling emotion detection
- `eye_aspect_ratio()`: Calculates eye openness
- `mouth_aspect_ratio()`: Measures mouth opening
- `mouth_curvature()`: Detects smile/frown
- `eyebrow_angle()`: Measures eyebrow position
- `analyze_emotion()`: Main emotion classification logic

### Facial Landmark Points

Uses dlib's 68-point facial landmark model:
- Points 36-47: Eyes
- Points 48-67: Mouth
- Points 17-26: Eyebrows
- Points 27-35: Nose
- Points 0-16: Jaw line

## Troubleshooting

### Common Issues

1. **"shape_predictor_68_face_landmarks.dat not found"**
   - Download the file from the provided link
   - Ensure it's in the correct directory

2. **Camera not detected**
   - Check if camera is connected and working
   - Try changing camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

3. **Poor emotion detection**
   - Ensure good lighting conditions
   - Look directly at camera during calibration
   - Maintain consistent distance from camera

4. **Installation issues with dlib**
   - On Windows: Install Visual Studio Build Tools
   - On Mac: `brew install cmake`
   - On Linux: `sudo apt-get install cmake`

## Customization

### Adjusting Sensitivity

Modify threshold values in `analyze_emotion()` method:
- Increase thresholds for stricter detection
- Decrease for more sensitive detection

### Adding New Emotions

1. Add emotion to `emotion_scores` dictionary
2. Implement detection logic in `analyze_emotion()`
3. Add color mapping in `draw_emotion_info()`

## Performance

- **FPS**: ~15-30 depending on hardware
- **CPU Usage**: Moderate (single-threaded)
- **Memory**: ~50-100MB

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Acknowledgments

- [dlib](http://dlib.net/) for facial landmark detection
- [OpenCV](https://opencv.org/) for computer vision operations
- Facial landmark dataset from iBUG 300-W project

## Future Improvements

- [ ] GPU acceleration with CUDA
- [ ] Deep learning emotion recognition
- [ ] Multi-face detection
- [ ] Emotion intensity measurement
- [ ] Export to CSV/JSON formats
- [ ] Web interface version
- [ ] Mobile app version

## Contact

For questions or suggestions, please open an issue or contact imad.ansarilol@gmail.com.
