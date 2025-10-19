#!/usr/bin/env python3
"""
Script to download MiDaS model weights
"""

import requests
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL to filepath"""
    try:
        logger.info(f"Downloading {url} to {filepath}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Downloaded {filepath.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {str(e)}")
        return False

def download_midas_weights():
    """Download MiDaS model weights"""
    
    # Define weights directory
    weights_dir = Path("models/midas/weights")
    
    # Model URLs from MiDaS releases
    models = {
        "midas_v21_small_256.pt": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
        "midas_v21_384.pt": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt",
        "dpt_hybrid_384.pt": "https://github.com/isl-org/MiDaS/releases/download/v2_1/dpt_hybrid_384.pt",
        "dpt_large_384.pt": "https://github.com/isl-org/MiDaS/releases/download/v2_1/dpt_large_384.pt"
    }
    
    logger.info("Downloading MiDaS model weights...")
    
    downloaded_count = 0
    for model_name, url in models.items():
        filepath = weights_dir / model_name
        
        # Skip if already exists
        if filepath.exists():
            logger.info(f"⏭️  {model_name} already exists, skipping")
            continue
        
        if download_file(url, filepath):
            downloaded_count += 1
    
    if downloaded_count > 0:
        logger.info(f"✅ Successfully downloaded {downloaded_count} model weights")
    else:
        logger.info("ℹ️  All model weights already exist")
    
    logger.info("🎯 Ready to use MiDaS for depth estimation!")

if __name__ == "__main__":
    download_midas_weights()
