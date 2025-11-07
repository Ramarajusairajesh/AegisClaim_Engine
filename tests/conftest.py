import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Set test environment variables
os.environ["ENV"] = "test"
os.environ["GEMINI_API_KEY"] = "test_key"  # Mock API key for testing

# Import any test fixtures here
# from tests.fixtures import *  # Uncomment if you create a fixtures directory
