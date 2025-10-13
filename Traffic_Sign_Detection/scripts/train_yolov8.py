# train_yolov8_colab.py 
import os
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

# ============================================
# CONFIGURATION - Modify these variables
# ============================================
DATA_YAML = '/combined_dataset/combined_dataset/data.yaml'  # Path to your data.yaml file
MODEL_SIZE = 'n'  # Model size: 'n', 's', 'm', 'l', or 'x'
EPOCHS = 50
IMGSZ = 416
BATCH_SIZE = 32
OUTPUT_DIR = '/Model'

def setup_training_environment():
    """Setup training environment and check requirements"""
    print("Setting up training environment...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        device = 'cuda'
    else:
        print("CUDA not available. Using CPU (training will be slow)")
        device = 'cpu'
    
    # Check ultralytics installation
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics._version_}")
    except ImportError:
        print("Ultralytics not installed. Installing now...")
        os.system('pip install ultralytics')
        import ultralytics
        print(f"Ultralytics version: {ultralytics._version_}")
    except AttributeError:
        print("Ultralytics installed (version info unavailable)")
    
    return device

def load_and_verify_dataset(data_yaml_path):
    """Load and verify the dataset configuration"""
    print(f"\nLoading dataset configuration from: {data_yaml_path}")
    
    if not os.path.exists(data_yaml_path):
        print(f"Data YAML file not found: {data_yaml_path}")
        return False
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Verify required paths exist
        required_dirs = ['train', 'val']
        for dir_type in required_dirs:
            dir_path = os.path.join(data_config['path'], data_config[dir_type])
            if not os.path.exists(dir_path):
                print(f"Directory not found: {dir_path}")
                return False
            print(f"Found {dir_type} directory: {dir_path}")
        
        # Check number of images
        train_dir = os.path.join(data_config['path'], data_config['train'])
        val_dir = os.path.join(data_config['path'], data_config['val'])
        
        train_images = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_images = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Training images: {train_images}")
        print(f"Validation images: {val_images}")
        print(f"Number of classes: {data_config['nc']}")
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset configuration: {e}")
        return False

def train_yolov8_model(data_yaml_path, model_size='n', epochs=100, imgsz=640, batch_size=16, output_dir='/content/runs'):
    """Train YOLOv8 model with the given parameters"""
    
    # Setup environment
    device = setup_training_environment()
    if device is None:
        return None
    
    # Verify dataset
    if not load_and_verify_dataset(data_yaml_path):
        return None
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nStarting YOLOv8 training...")
    print(f"Run directory: {run_dir}")
    
    try:
        # Load model
        model_name = f'yolov8{model_size}.pt'
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)
        
        # Training parameters (optimized for Colab)
        training_params = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'workers': 8,
            'patience': 20,
            'save': True,
            'project': run_dir,
            'optimizer': 'auto',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'verbose': True,
            'seed': 42,
            'plots': True
        }
        
        print(f"\nTraining Configuration:")
        print(f"   Model: YOLOv8{model_size.upper()}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")
        print(f"   Dataset: {data_yaml_path}")
        
        # Start training
        print(f"\nStarting training...")
        results = model.train(**training_params)
        
        print(f"\nTraining completed successfully!")
        return model, run_dir
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model, data_yaml_path, run_dir):
    """Evaluate the trained model"""
    try:
        print(f"\nEvaluating model...")
        
        # Load the best model for evaluation
        best_model_path = os.path.join(run_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            eval_model = YOLO(best_model_path)
            
            # Run validation
            metrics = eval_model.val(
                data=data_yaml_path,
                split='val',
                imgsz=640,
                batch=16,
                save_json=True,
                conf=0.001,
                iou=0.6
            )
            
            print(f"Evaluation completed!")
            print(f"   mAP50: {metrics.box.map50:.4f}")
            print(f"   mAP50-95: {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.mp:.4f}")
            print(f"   Recall: {metrics.box.mr:.4f}")
            
            return metrics
        else:
            print("Best model not found for evaluation")
            return None
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# MAIN EXECUTION
# ============================================
print("=" * 60)
print("YOLOv8 TRAINING SCRIPT FOR GOOGLE COLAB")
print("=" * 60)

# Train the model
result = train_yolov8_model(
    data_yaml_path=DATA_YAML,
    model_size=MODEL_SIZE,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch_size=BATCH_SIZE,
    output_dir=OUTPUT_DIR
)

if result is not None:
    model, run_dir = result
    
    # Evaluate the model
    metrics = evaluate_model(model, DATA_YAML, run_dir)
    
    print(f"\nTraining completed successfully!")
    print(f"Results saved in: {run_dir}")
    print(f"Best model: {run_dir}/weights/best.pt")
    
    print(f"\nTo use the trained model for inference:")
    print(f"   from ultralytics import YOLO")
    print(f"   model = YOLO('{run_dir}/weights/best.pt')")
    print(f"   results = model('path_to_image.jpg')")
    print(f"   results[0].show()  # Display results")
    
else:
    print("Training failed!")