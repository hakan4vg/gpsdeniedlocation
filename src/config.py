import yaml
import os
from pathlib import Path

def get_config_path(config_name="config.yaml"):
    """Get absolute path to config file"""
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir / config_name

def load_config(config_path=None):
    """Load config, create default if not exists"""
    if config_path is None:
        config_path = get_config_path()
    
    if not os.path.exists(config_path):
        create_default_config(config_path)
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config, config_path=None):
    """
    Saves configuration to the specified YAML file.
    """
    if config_path is None:
        config_path = get_config_path()
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def create_default_config(config_path):
    """Create and save default configuration"""
    default_config = {
        'feature_extraction': {
            'algorithm': 'ORB',
            'params': {
                'nfeatures': 1000,
                'scaleFactor': 1.2,
                'nlevels': 8,
            }
        },
        'geospatial_data': {
            'osm_data_path': 'data/osm_data',
            'landmark_types': ['building', 'road', 'waterway'],
        },
        'localization': {
            'matching_threshold': 0.7,
            'ransac_iterations': 1000,
            'pnp_algorithm': 'SOLVEPNP_RANSAC',
        },
        'runtime': {
            'device': 'RPi',
            'memory_limit_gb': 1.5,
            'latency_limit_sec': 2.0,
        }
    }
    
    with open(config_path, 'w') as file:
        yaml.dump(default_config, file)
    
    return default_config

# Example usage and default configuration structure
if __name__ == "__main__":
    save_config(create_default_config())
    loaded_config = load_config()
    print("Default Configuration:", loaded_config)
