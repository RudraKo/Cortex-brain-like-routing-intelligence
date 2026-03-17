import sys
import os

# Ensure the app package can be imported correctly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.main import app
