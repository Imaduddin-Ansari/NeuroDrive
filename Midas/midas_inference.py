#!/usr/bin/env python3
"""
MIDAS Depth Estimation Inference Script
Runs depth estimation on images using locally downloaded models
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import json
import time

class DepthEstimationModel(nn.Module):
    """Generic depth estimation model wrapper"""
    
    def __init__(self, model_path, model_type):
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path
        
        # Load the actual model
        self.model = self._load_pretrained_model()
        
    def _load_pretrained_model(self):
        """Load pre-trained model"""
        try:
            # Try loading complete model first
            model = torch.load(self.model_path, map_location='cpu')
            
            # If it's a state dict, we need to create architecture
            if isinstance(model, dict):
                # This is a simplified approach - in practice you'd need
                # the exact architecture for each model type
                print(f"⚠️ Loaded state dict for {self.model_type}")
                print("   Note: Using simplified architecture, results may vary")
                return self._create_simple_model(model)
            else:
                print(f"✅ Loaded complete model for {self.model_type}")
                return model
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def _create_simple_model(self, state_dict):
        """Create a simple model architecture"""
        # This is a placeholder - you'd need actual architectures
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            print("⚠️ Could not load state dict completely")
        
        return model
    
    def forward(self, x):
        if self.model is None:
            return None
        return self.model(x)

class MidasInference:
    """MIDAS depth estimation inference engine"""
    
    def __init__(self, model_name="midas_v21", models_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = Path(models_dir)
        self.model_name = model_name
        
        print(f"🚀 Initializing MIDAS Inference")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_name}")
        
        # Load model info
        self.model_info = self._load_model_info()
        
        # Initialize model
        self.model = self._load_model()
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
        if self.model is None:
            print("❌ Model initialization failed")
        else:
            print("✅ Model loaded successfully")
    
    def _load_model_info(self):
        """Load model configuration"""
        info_file = self.models_dir / "models_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                return info.get(self.model_name, {})
            except:
                pass
        return {}
    
    def _load_model(self):
        """Load the depth estimation model"""
        model_path = self.models_dir / f"{self.model_name}.pt"
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            print("   Run the downloader script first:")
            print(f"   python midas_downloader.py --action download --model {self.model_name}")
            return None
        
        try:
            print(f"📂 Loading model from: {model_path}")
            
            # Load model
            model = torch.load(model_path, map_location=self.device)
            
            # Handle different model formats
            if hasattr(model, 'eval'):
                model.eval()
            elif isinstance(model, dict):
                # This is likely a state dict - create wrapper
                model = DepthEstimationModel(model_path, self.model_name)
                model.to(self.device)
                model.eval()
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        # Get input size from model info, or use defaults
        if self.model_info and 'config' in self.model_info:
            input_size = self.model_info['config'].get('input_size', (384, 384))
        else:
            # Default sizes based on model type
            size_map = {
                'dpt_large': (384, 384),
                'dpt_hybrid': (384, 384), 
                'midas_v21': (384, 384),
                'midas_v21_small': (256, 256)
            }
            input_size = size_map.get(self.model_name, (384, 384))
        
        print(f"🔧 Input size: {input_size}")
        
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"❌ Could not load image: {image_path}")
                    return None, None
            else:
                image = image_path
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Store original size
            original_size = image_pil.size
            
            # Apply transforms
            input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            return input_tensor, original_size
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None
    
    def predict_depth(self, input_tensor, original_size):
        """Run depth prediction"""
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                # Run inference
                start_time = time.time()
                prediction = self.model(input_tensor)
                inference_time = time.time() - start_time
                
                print(f"⏱️ Inference time: {inference_time:.3f}s")
                
                # Handle different output formats
                if isinstance(prediction, (list, tuple)):
                    prediction = prediction[0]
                
                # Move to CPU and convert to numpy
                depth_map = prediction.squeeze().cpu().numpy()
                
                # Resize to original image size
                if depth_map.shape != original_size[::-1]:
                    depth_map = cv2.resize(depth_map, original_size, 
                                         interpolation=cv2.INTER_LINEAR)
                
                return depth_map
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def estimate_depth(self, image_path):
        """Complete depth estimation pipeline"""
        print(f"\n🔍 Processing: {image_path}")
        
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image_path)
        if input_tensor is None:
            return None, None
        
        # Predict
        depth_map = self.predict_depth(input_tensor, original_size)
        if depth_map is None:
            return None, None
        
        # Normalize for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, 
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_map, depth_normalized

def visualize_results(image_path, depth_map, depth_normalized, save_output=True):
    """Visualize depth estimation results"""
    
    # Load original image
    original = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'MIDAS Depth Estimation - {Path(image_path).name}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Raw depth map
    im1 = axes[0, 1].imshow(depth_map, cmap='plasma')
    axes[0, 1].set_title("Raw Depth Map")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # Normalized depth map
    axes[1, 0].imshow(depth_normalized, cmap='plasma')
    axes[1, 0].set_title("Normalized Depth Map")
    axes[1, 0].axis('off')
    
    # Depth map with different colormap
    axes[1, 1].imshow(depth_normalized, cmap='viridis')
    axes[1, 1].set_title("Depth Map (Viridis)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save visualization
        vis_path = output_dir / f"{Path(image_path).stem}_depth_visualization.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        
        # Save depth maps
        depth_path = output_dir / f"{Path(image_path).stem}_depth.png"
        cv2.imwrite(str(depth_path), depth_normalized)
        
        # Save colorized version
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        colored_path = output_dir / f"{Path(image_path).stem}_depth_colored.png"
        cv2.imwrite(str(colored_path), depth_colored)
        
        print(f"💾 Outputs saved to:")
        print(f"   Visualization: {vis_path}")
        print(f"   Depth map: {depth_path}")
        print(f"   Colored depth: {colored_path}")
    
    plt.show()

def process_single_image(image_path, model_name, models_dir):
    """Process a single image"""
    
    # Initialize inference engine
    inference = MidasInference(model_name, models_dir)
    
    if inference.model is None:
        return False
    
    # Run depth estimation
    depth_map, depth_normalized = inference.estimate_depth(image_path)
    
    if depth_map is None:
        print("❌ Depth estimation failed")
        return False
    
    # Visualize results
    visualize_results(image_path, depth_map, depth_normalized)
    
    print("✅ Processing complete!")
    return True

def process_multiple_images(image_dir, model_name, models_dir):
    """Process multiple images in a directory"""
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images")
    
    # Initialize inference engine
    inference = MidasInference(model_name, models_dir)
    
    if inference.model is None:
        return
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        depth_map, depth_normalized = inference.estimate_depth(image_path)
        
        if depth_map is not None:
            # Save without showing plot
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            depth_path = output_dir / f"{image_path.stem}_depth.png"
            cv2.imwrite(str(depth_path), depth_normalized)
            
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
            colored_path = output_dir / f"{image_path.stem}_depth_colored.png"
            cv2.imwrite(str(colored_path), depth_colored)
            
            print(f"   ✅ Saved: {depth_path}")
        else:
            print(f"   ❌ Failed to process")
    
    print(f"\n🎉 Batch processing complete! Check 'outputs/' directory")

def main():
    parser = argparse.ArgumentParser(description="MIDAS Depth Estimation Inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--image-dir", type=str, help="Directory containing images")
    parser.add_argument("--model", type=str, default="midas_v21",
                       choices=["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21_small"],
                       help="Model to use for inference")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory containing downloaded models")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        models_dir = Path(args.models_dir)
        print("\n📋 Available Models:")
        print("=" * 40)
        
        model_files = list(models_dir.glob("*.pt"))
        if model_files:
            for model_file in model_files:
                model_name = model_file.stem
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"✅ {model_name} ({size_mb:.1f} MB)")
        else:
            print("❌ No models found")
            print("   Download models first:")
            print("   python midas_downloader.py --action download-all")
        
        print("=" * 40)
        return
    
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        process_single_image(args.image, args.model, args.models_dir)
    
    elif args.image_dir:
        if not Path(args.image_dir).exists():
            print(f"❌ Directory not found: {args.image_dir}")
            return
        
        process_multiple_images(args.image_dir, args.model, args.models_dir)
    
    else:
        print("❌ Please provide --image or --image-dir")
        print("\nUsage examples:")
        print("  python midas_inference.py --image photo.jpg --model midas_v21")
        print("  python midas_inference.py --image-dir ./photos --model midas_v21_small")
        print("  python midas_inference.py --list-models")

if __name__ == "__main__":
    main()

   #python midas_inference.py --image photo.jpg --model midas_v21
