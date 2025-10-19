#!/usr/bin/env python3
"""
Run script for the Nuu 3D Room Scanner backend
"""

import uvicorn
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

if __name__ == "__main__":
    print("🚀 Starting Nuu 3D Room Scanner Backend...")
    print("📁 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📋 Sample Videos: http://localhost:8000/api/samples")
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
