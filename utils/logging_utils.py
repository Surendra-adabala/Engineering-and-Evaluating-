"""
Logging utilities for email classification
"""
import logging
import os
import sys
from typing import Optional, Dict, Any


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level for console output
        log_file: Path to log file (if None, file logging is disabled)
        log_format: Format string for logging
        file_level: Logging level for file output

    Returns:
        Root logger
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all levels, filters handle the rest

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class PipelineLogger:
    """Logger class specifically designed for pipeline stages"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def start_stage(self, stage_name: str):
        """Log the start of a pipeline stage"""
        self.logger.info(f"Starting stage: {stage_name}")
        
    def end_stage(self, stage_name: str, elapsed_time: float):
        """Log the end of a pipeline stage with timing information"""
        self.logger.info(f"Completed stage: {stage_name} in {elapsed_time:.2f} seconds")
        
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log a dictionary of metrics"""
        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
                
            self.logger.info(f"Metric - {metric_name}: {formatted_value}")
    
    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
        
    def info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
        
    def debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message) 