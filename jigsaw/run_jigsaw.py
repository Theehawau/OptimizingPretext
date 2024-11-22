'''
Script for running JigSaw training 
<run_experiments> function should be replaced by the appropiate function

Emilio VC 
'''

import argparse
import yaml
import logging
import os
from datetime import datetime

from jigsaw_training import run_training_jigsaw

def setup_logging(log_dir, experiment_name):
    """Setup logging to a file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Clear any existing handlers if they exist and reconfigure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8', delay=False),  # Ensure immediate file writing
            logging.StreamHandler()  # Console output
        ],
        force=True  # Force logging configuration to reset if other loggers are set
    )
    
    logging.info(f"Logging setup completed. Logs saved to {log_path}")


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded from {config_path}")
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments with configuration.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='./logs', 
        help="Directory to save logs (default: ./logs)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load the configuration file
    config = load_config(args.config)
    
    # Setup logging
    experiment_name = config.get("experiment_name", "experiment")
    setup_logging(args.log_dir, experiment_name)
    
    # Run the experiment
    results = run_training_jigsaw(config)

    logging.info(results)