"""
Configuration module for wekeo_plumes_post_process.

Loads environment variables from .env file and provides
directory paths for input/output data.
"""

from wekeo_plumes_post_process.hygeos_core.env import getdir

# Load directories
output_dir = getdir("OUTPUT_DIR") / "plumes_output"
output_dir.mkdir(parents=False, exist_ok=True)
