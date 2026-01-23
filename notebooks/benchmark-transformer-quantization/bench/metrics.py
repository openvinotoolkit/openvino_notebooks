import psutil
import os
from pathlib import Path

class MetricsCollector:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_rss = self.process.memory_info().rss

    def get_current_rss_mb(self):
        """Returns Resident Set Size (Physical RAM) in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def get_ram_growth_mb(self):
        """How much RAM has this process consumed since init?"""
        return self.get_current_rss_mb() - (self.baseline_rss / 1024 / 1024)

    @staticmethod
    def get_directory_size_mb(directory):
        """Calculates disk footprint of the exported model."""
        root_directory = Path(directory)
        if not root_directory.exists(): return 0
        return sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file()) / (1024 * 1024)