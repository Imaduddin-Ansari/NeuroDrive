import requests
import os
from pathlib import Path
import hashlib
import json
from tqdm import tqdm
import argparse

class MidasModelDownloader:
    """Download and manage MIDAS models"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models_config = {
            "dpt_large": {
                "url": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
                "size": "1.4GB",
                "description": "Best quality, slowest inference",
                "input_size": (384, 384),
                "sha256": "2f21e586"  # partial hash from filename
            },
            "dpt_hybrid": {
                "url": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt", 
                "size": "470MB",
                "description": "Good quality, medium speed",
                "input_size": (384, 384),
                "sha256": "501f0c75"
            },
            "midas_v21": {
                "url": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
                "size": "350MB", 
                "description": "Good quality, faster than DPT",
                "input_size": (384, 384),
                "sha256": "f6b98070"
            },
            "midas_v21_small": {
                "url": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
                "size": "100MB",
                "description": "Lower quality, fastest inference", 
                "input_size": (256, 256),
                "sha256": "70d6b9c8"
            }
        }
        
        self.info_file = self.models_dir / "models_info.json"
    
    def test_connection(self):
        """Test internet connection"""
        try:
            response = requests.get("https://github.com", timeout=10)
            print(f"✅ Internet connection OK (GitHub accessible)")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Internet connection failed: {e}")
            return False
    
    def download_file_with_progress(self, url, destination):
        """Download file with progress bar"""
        try:
            print(f"\n📥 Downloading: {destination.name}")
            print(f"🔗 URL: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Successfully downloaded: {destination}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            if destination.exists():
                destination.unlink()  # Remove partial file
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if destination.exists():
                destination.unlink()
            return False
    
    def verify_file(self, file_path, expected_hash_part):
        """Verify downloaded file (basic check)"""
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        if file_size < 1000000:  # Less than 1MB is probably wrong
            return False
        
        # Basic verification - check if it's a pytorch file
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                # PyTorch files typically start with PK (zip format)
                if header[:2] == b'PK':
                    return True
        except:
            pass
        
        return False
    
    def list_available_models(self):
        """Display available models"""
        print("\n📋 Available MIDAS Models:")
        print("=" * 60)
        for model_name, config in self.models_config.items():
            status = "✅ Downloaded" if self.is_model_downloaded(model_name) else "⬜ Not downloaded"
            print(f"\n🎯 {model_name.upper()}")
            print(f"   Description: {config['description']}")
            print(f"   Size: {config['size']}")
            print(f"   Input size: {config['input_size']}")
            print(f"   Status: {status}")
        print("=" * 60)
    
    def is_model_downloaded(self, model_name):
        """Check if model is already downloaded"""
        model_path = self.models_dir / f"{model_name}.pt"
        return model_path.exists() and self.verify_file(model_path, self.models_config[model_name]["sha256"])
    
    def download_model(self, model_name):
        """Download a specific model"""
        if model_name not in self.models_config:
            print(f"❌ Unknown model: {model_name}")
            return False
        
        model_path = self.models_dir / f"{model_name}.pt"
        
        # Check if already downloaded
        if self.is_model_downloaded(model_name):
            print(f"✅ {model_name} already downloaded and verified")
            return True
        
        # Download
        config = self.models_config[model_name]
        success = self.download_file_with_progress(config["url"], model_path)
        
        if success:
            # Verify download
            if self.verify_file(model_path, config["sha256"]):
                print(f"✅ {model_name} downloaded and verified successfully")
                self.save_model_info(model_name)
                return True
            else:
                print(f"❌ {model_name} download verification failed")
                model_path.unlink()
                return False
        
        return False
    
    def download_all_models(self):
        """Download all available models"""
        print("📦 Downloading all MIDAS models...")
        
        success_count = 0
        for model_name in self.models_config.keys():
            if self.download_model(model_name):
                success_count += 1
        
        print(f"\n🎉 Download summary: {success_count}/{len(self.models_config)} models downloaded successfully")
        
        if success_count > 0:
            print("\n✅ You can now use the models with the inference script!")
            print("Example: python midas_inference.py --image your_image.jpg --model midas_v21")
    
    def save_model_info(self, model_name):
        """Save model information"""
        info = {}
        if self.info_file.exists():
            try:
                with open(self.info_file, 'r') as f:
                    info = json.load(f)
            except:
                info = {}
        
        info[model_name] = {
            "downloaded": True,
            "config": self.models_config[model_name]
        }
        
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def get_downloaded_models(self):
        """Get list of downloaded models"""
        downloaded = []
        for model_name in self.models_config.keys():
            if self.is_model_downloaded(model_name):
                downloaded.append(model_name)
        return downloaded
    
    def clean_incomplete_downloads(self):
        """Remove incomplete or corrupted downloads"""
        print("🧹 Cleaning incomplete downloads...")
        
        for model_name in self.models_config.keys():
            model_path = self.models_dir / f"{model_name}.pt"
            if model_path.exists() and not self.verify_file(model_path, self.models_config[model_name]["sha256"]):
                print(f"🗑️ Removing corrupted file: {model_path}")
                model_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="MIDAS Model Downloader")
    parser.add_argument("--action", 
                       choices=["list", "download", "download-all", "test-connection", "clean"], 
                       default="list",
                       help="Action to perform")
    parser.add_argument("--model", 
                       choices=["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21_small"],
                       help="Specific model to download")
    parser.add_argument("--models-dir", 
                       default="models",
                       help="Directory to save models (default: models)")
    
    args = parser.parse_args()
    
    downloader = MidasModelDownloader(args.models_dir)
    
    if args.action == "test-connection":
        downloader.test_connection()
    
    elif args.action == "list":
        downloader.list_available_models()
        downloaded = downloader.get_downloaded_models()
        if downloaded:
            print(f"\n✅ Downloaded models: {', '.join(downloaded)}")
        else:
            print("\n⬜ No models downloaded yet")
            print("\nTo download a model:")
            print("  python midas_downloader.py --action download --model midas_v21_small")
            print("  python midas_downloader.py --action download-all")
    
    elif args.action == "download":
        if not args.model:
            print("❌ Please specify --model")
            downloader.list_available_models()
        else:
            if not downloader.test_connection():
                print("❌ Cannot download without internet connection")
                return
            downloader.download_model(args.model)
    
    elif args.action == "download-all":
        if not downloader.test_connection():
            print("❌ Cannot download without internet connection")
            return
        downloader.download_all_models()
    
    elif args.action == "clean":
        downloader.clean_incomplete_downloads()

if __name__ == "__main__":
    main()