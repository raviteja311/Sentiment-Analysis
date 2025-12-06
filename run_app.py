#!/usr/bin/env python
"""
Wrapper script to run the Streamlit app from the project root.
This ensures proper module imports.
"""
import subprocess
import sys
from pathlib import Path

# Get the directory containing this script
project_root = Path(__file__).parent

# Run streamlit from the project root
app_path = project_root / "app" / "streamlit_app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], cwd=str(project_root))
