#!/usr/bin/env python3
"""
Installation script for MiDaS depth estimation
Run this to set up MiDaS for depth estimation
"""

import subprocess
import sys
import os
from pathlib import Path

def install_midas():
    """Install MiDaS and dependencies"""
    
    print("🚀 Installing MiDaS depth estimation...")
    
    try:
        # Install PyTorch dependencies
        print("Installing PyTorch dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "timm"
        ])
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Clone MiDaS repository
        midas_dir = models_dir / "midas"
        if not midas_dir.exists():
            print("Cloning MiDaS repository...")
            subprocess.check_call([
                "git", "clone", 
                "https://github.com/isl-org/MiDaS.git",
                str(midas_dir)
            ])
        else:
            print("MiDaS repository already exists")
        
        # Install additional dependencies
        print("Installing additional dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "opencv-python", "pillow", "numpy"
        ])
        
        print("MiDaS installation completed successfully!")
        print("You can now run depth estimation tests")
        
    except subprocess.CalledProcessError as e:
        print(f"x Installation failed: {e}")
        return False
    except Exception as e:
        print(f"x Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = install_midas()
    if success:
        print("\nInstallation complete! Next steps:")
        print("1. Run: python test_depth_estimation.py")
        print("2. Start server: python run.py")
    else:
        print("\nInstallation failed. Please check the error messages above.")
