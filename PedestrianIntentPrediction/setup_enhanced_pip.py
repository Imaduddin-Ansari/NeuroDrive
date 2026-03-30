#!/usr/bin/env python3
"""
ENHANCED PIP SYSTEM SETUP SCRIPT
================================
Automated setup and validation for the Enhanced PIP System.
Checks dependencies, downloads models, and validates installation.

Author: NeuroDrive Team
Date: January 2025
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import json


class PIPSystemSetup:
    """Setup and validation for Enhanced PIP System."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_log = []
        
    def log(self, message, level="INFO"):
        """Log setup messages."""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.setup_log.append(log_entry)
    
    def run_setup(self):
        """Run complete setup process."""
        self.log("Starting Enhanced PIP System Setup", "INFO")
        self.log("="*50, "INFO")
        
        success = True
        
        # Step 1: Check Python version
        if not self.check_python_version():
            success = False
        
        # Step 2: Create directories
        if not self.create_directories():
            success = False
        
        # Step 3: Install dependencies
        if not self.install_dependencies():
            success = False
        
        # Step 4: Download models
        if not self.download_models():
            success = False
        
        # Step 5: Validate installation
        if not self.validate_installation():
            success = False
        
        # Step 6: Run basic tests
        if not self.run_basic_tests():
            success = False
        
        # Final report
        self.generate_setup_report(success)
        
        return success
    
    def check_python_version(self):
        """Check Python version compatibility."""
        self.log("Checking Python version...", "INFO")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            self.log(f"Python {version.major}.{version.minor} detected", "ERROR")
            self.log("Python 3.8+ required", "ERROR")
            return False
        
        self.log(f"Python {version.major}.{version.minor}.{version.micro} - OK", "INFO")
        return True
    
    def create_directories(self):
        """Create necessary directories."""
        self.log("Creating project directories...", "INFO")
        
        directories = [
            'models',
            'data',
            'data/PIE',
            'output',
            'logs',
            'audio'
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log(f"Created directory: {directory}", "INFO")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to create directories: {e}", "ERROR")
            return False
    
    def install_dependencies(self):
        """Install required dependencies."""
        self.log("Installing dependencies...", "INFO")
        
        try:
            # Install from requirements.txt
            requirements_file = self.project_root / 'requirements.txt'
            
            if requirements_file.exists():
                cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log("Dependencies installed successfully", "INFO")
                else:
                    self.log(f"Dependency installation failed: {result.stderr}", "ERROR")
                    return False
            else:
                self.log("requirements.txt not found, installing core dependencies", "WARN")
                
                # Install core dependencies manually
                core_deps = [
                    'torch', 'torchvision', 'ultralytics', 'opencv-python',
                    'numpy', 'scipy', 'scikit-learn', 'pyttsx3', 'pygame',
                    'filterpy', 'matplotlib', 'pandas'
                ]
                
                for dep in core_deps:
                    cmd = [sys.executable, '-m', 'pip', 'install', dep]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.log(f"Installed {dep}", "INFO")
                    else:
                        self.log(f"Failed to install {dep}: {result.stderr}", "WARN")
            
            return True
            
        except Exception as e:
            self.log(f"Dependency installation error: {e}", "ERROR")
            return False
    
    def download_models(self):
        """Download required models."""
        self.log("Downloading models...", "INFO")
        
        try:
            # YOLOv8n model (will be downloaded automatically by ultralytics)
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            self.log("YOLOv8n model ready", "INFO")
            
            # Copy to models directory
            import shutil
            yolo_source = Path.home() / '.ultralytics' / 'yolov8n.pt'
            yolo_dest = self.project_root / 'yolov8n.pt'
            
            if yolo_source.exists() and not yolo_dest.exists():
                shutil.copy2(yolo_source, yolo_dest)
                self.log("YOLOv8n model copied to project directory", "INFO")
            
            return True
            
        except Exception as e:
            self.log(f"Model download error: {e}", "ERROR")
            return False
    
    def validate_installation(self):
        """Validate installation by importing key modules."""
        self.log("Validating installation...", "INFO")
        
        modules_to_test = [
            ('torch', 'PyTorch'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('ultralytics', 'Ultralytics'),
            ('pyttsx3', 'Text-to-Speech'),
            ('pygame', 'Pygame (optional)'),
            ('filterpy', 'FilterPy'),
            ('sklearn', 'Scikit-learn')
        ]
        
        failed_imports = []
        
        for module_name, display_name in modules_to_test:
            try:
                __import__(module_name)
                self.log(f"{display_name} - OK", "INFO")
            except ImportError as e:
                self.log(f"{display_name} - FAILED: {e}", "WARN")
                failed_imports.append(module_name)
        
        if failed_imports:
            self.log(f"Some optional modules failed to import: {failed_imports}", "WARN")
            self.log("System may still work with reduced functionality", "WARN")
        
        return len(failed_imports) < 3  # Allow some optional failures
    
    def run_basic_tests(self):
        """Run basic functionality tests."""
        self.log("Running basic tests...", "INFO")
        
        try:
            # Test 1: Import main system
            sys.path.insert(0, str(self.project_root))
            
            try:
                from enhanced_pip_system import EnhancedConfig, AudioAlertSystem
                self.log("Enhanced PIP System import - OK", "INFO")
            except ImportError as e:
                self.log(f"Enhanced PIP System import - FAILED: {e}", "ERROR")
                return False
            
            # Test 2: Create config
            try:
                config = EnhancedConfig()
                self.log("Configuration creation - OK", "INFO")
            except Exception as e:
                self.log(f"Configuration creation - FAILED: {e}", "ERROR")
                return False
            
            # Test 3: Test audio system (without actual audio)
            try:
                audio_system = AudioAlertSystem(enable_tts=False, enable_sounds=False)
                audio_system.shutdown()
                self.log("Audio system test - OK", "INFO")
            except Exception as e:
                self.log(f"Audio system test - FAILED: {e}", "WARN")
            
            # Test 4: Test YOLO model
            try:
                from ultralytics import YOLO
                model = YOLO('yolov8n.pt')
                self.log("YOLO model test - OK", "INFO")
            except Exception as e:
                self.log(f"YOLO model test - FAILED: {e}", "ERROR")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Basic tests failed: {e}", "ERROR")
            return False
    
    def generate_setup_report(self, success):
        """Generate setup completion report."""
        self.log("="*50, "INFO")
        
        if success:
            self.log("🎉 SETUP COMPLETED SUCCESSFULLY!", "INFO")
            self.log("Enhanced PIP System is ready to use", "INFO")
            self.log("", "INFO")
            self.log("Next steps:", "INFO")
            self.log("1. Run demo: python demo_enhanced_pip.py", "INFO")
            self.log("2. Run tests: python test_enhanced_pip.py", "INFO")
            self.log("3. Start system: python enhanced_pip_system.py --input 0", "INFO")
        else:
            self.log("❌ SETUP FAILED", "ERROR")
            self.log("Please check the error messages above", "ERROR")
            self.log("You may need to manually install missing dependencies", "ERROR")
        
        self.log("="*50, "INFO")
        
        # Save setup log
        log_file = self.project_root / 'setup_log.txt'
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.setup_log))
        
        self.log(f"Setup log saved to: {log_file}", "INFO")
        
        return success


def main():
    """Main setup function."""
    print("Enhanced PIP System Setup")
    print("=" * 30)
    
    setup = PIPSystemSetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("You can now run the Enhanced PIP System.")
        
        # Ask if user wants to run demo
        try:
            response = input("\nWould you like to run the demo now? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("Starting demo...")
                os.system(f"{sys.executable} demo_enhanced_pip.py --duration 30")
        except KeyboardInterrupt:
            print("\nSetup completed.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()