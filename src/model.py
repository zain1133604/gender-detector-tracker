# model.py

import os
import logging
from ultralytics import YOLO

class CustomYOLOModel:
    """
    A wrapper around the Ultralytics YOLO model for consistent loading
    from pre-trained weights or existing checkpoints.
    """
    def __init__(self, model_path: str = 'yolov8l.pt'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = self._load_model(model_path)
        self.logger.info(f"Model initialized: {model_path}")

    def _load_model(self, path: str) -> YOLO:
        """
        Loads the YOLO model from a given path.
        If the path doesn't exist, it defaults to a pre-trained yolov8l.pt.
        """
        if os.path.exists(path):
            self.logger.info(f"Attempting to load model from: {path}")
            return YOLO(path)
        else:
            self.logger.warning(f"Model file not found at {path}. Loading default 'yolov8l.pt' instead.")
            return YOLO('yolov8l.pt')

    def get_model(self) -> YOLO:
        """
        Returns the loaded Ultralytics YOLO model instance.
        """
        return self.model