"""Configuration management for NeuroDrive"""
import json
import os


class Config:
    """Manages application configuration"""
    
    DEFAULT_CONFIG = {
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
    
    def __init__(self, config_file="neurodrive_config.json"):
        self.config_file = config_file
        self.config = self.load()
    
    def load(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            config = self.DEFAULT_CONFIG.copy()
            self.save(config)
            return config
    
    def save(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates):
        """Update multiple configuration values"""
        self.config.update(updates)