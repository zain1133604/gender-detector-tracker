# config.py

import os
import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Configuration settings for the model training process.
    """
    # --- Project Paths and Naming ---
    data_dir: str = r"A:\New folder\project5_human_detection"
    # Base directory for all model checkpoints and runs
    checkpoint_base_dir: str = r"A:\model_checkpoints"
    # Main project name, will be a subfolder in checkpoint_base_dir
    project_name: str = "gender_tracker"
    # Fixed run ID for all checkpoints. This ensures consistent saving/loading.
    run_id: str = "gender_detection_run63" 
    
    # --- Model Hyperparameters ---
    model_type: str = 'yolov8l.pt'  # Base YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt, yolov8l.pt)
    image_size: int = 640          # Input image size for the model
    batch_size: int = 4            # Batch size for training (set to 4 to prevent crashes)
    num_epochs: int = 40           # Total number of training epochs
    
    # --- Optimization and Learning ---
    freeze_backbone_epochs: int = 0 # <--- THIS IS THE ONLY CHANGE
    patience: int = 10               # Early stopping patience (epochs without improvement)
    
    # --- System Settings ---
    seed: int = 42                 # Random seed for reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # Automatically use GPU if available

    @property
    def project_dir(self) -> str:
        """Returns the full path to the project's checkpoint directory."""
        return os.path.join(self.checkpoint_base_dir, self.project_name)

    @property
    def run_dir(self) -> str:
        """Returns the full path to the current run's directory where weights are saved."""
        return os.path.join(self.project_dir, self.run_id)

    @property
    def last_checkpoint_path(self) -> str:
        """Returns the path to the last saved model checkpoint."""
        return os.path.join(self.run_dir, 'weights', 'last.pt')

    @property
    def best_checkpoint_path(self) -> str:
        """Returns the path to the best saved model checkpoint."""
        return os.path.join(self.run_dir, 'weights', 'best.pt')