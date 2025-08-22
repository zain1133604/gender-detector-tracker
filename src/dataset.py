# dataset.py

import os
import logging

class DatasetManager:
    """
    Manages dataset paths and structure for Ultralytics YOLO models.
    Ultralytics handles data loading internally based on the data.yaml file.
    This class ensures the data.yaml path is correctly configured.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data_yaml_path = os.path.join(self.data_dir, 'data.yaml')
        self.logger = logging.getLogger(self.__class__.__name__)

    def verify_data_yaml(self):
        """
        Verifies if the data.yaml file exists and logs its path.
        """
        if not os.path.exists(self.data_yaml_path):
            self.logger.error(f"Error: data.yaml not found at {self.data_yaml_path}. "
                              "Please ensure your dataset is correctly configured.")
            raise FileNotFoundError(f"data.yaml not found: {self.data_yaml_path}")
        self.logger.info(f"Using data.yaml from: {self.data_yaml_path}")

    def get_data_yaml_path(self) -> str:
        """
        Returns the full path to the data.yaml file.
        """
        self.verify_data_yaml()
        return self.data_yaml_path