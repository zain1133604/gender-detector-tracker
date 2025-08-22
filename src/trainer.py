# trainer.py


import logging
from ultralytics import YOLO
from config import TrainingConfig
from dataset import DatasetManager
from model import CustomYOLOModel

class Trainer:
    """
    Manages the training process for the YOLO model, including
    resuming from checkpoints and saving results.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_manager = DatasetManager(config.data_dir)
        
        self._setup_model()
        self.model.to(self.config.device) # Ensure model is on the correct device

    def _setup_model(self):
        """
        Initializes or resumes the model based on available checkpoints.
        This logic ensures training continues from the last point if possible.
        """
        # Try to load from the fixed checkpoint location first
        try:
            self.model_wrapper = CustomYOLOModel(model_path=self.config.last_checkpoint_path)
            self.model = self.model_wrapper.get_model()
            self.logger.info(f"Successfully loaded checkpoint from {self.config.last_checkpoint_path}. Starting new training session with these weights.")
        except Exception as e:
            self.logger.warning(f"Could not load checkpoint: {e}. Starting training from scratch with pre-trained weights.")
            self.model_wrapper = CustomYOLOModel(model_path=self.config.model_type)
            self.model = self.model_wrapper.get_model()

        self.logger.info(f"Model ready for training.")

    def train(self):
        """
        Starts or resumes the model training.
        """
        self.logger.info("Starting training process...")
        self.dataset_manager.verify_data_yaml() # Ensure data.yaml is valid

        # To completely override the old 'freeze' parameter, we must start a NEW training run
        # using the last checkpoint as the starting model. We explicitly set freeze=0.
        results = self.model.train(
            data=self.dataset_manager.get_data_yaml_path(),
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            project=self.config.project_dir,
            name=self.config.run_id,
            patience=self.config.patience,
            device=self.config.device,
            cache=False,
            epochs=self.config.num_epochs, # This restarts the training from epoch 0
            freeze=0 # <--- THIS IS THE DEFINITIVE FIX
        )
        self.logger.info("Training finished.")

    def validate(self):
        """
        Runs validation on the model.
        """
        self.logger.info("Running validation on the model...")
        metrics = self.model.val(
            data=self.dataset_manager.get_data_yaml_path(),
            split="val",
            device=self.config.device
        )
        self.logger.info("Validation completed.")
        return metrics

    def test(self):
        """
        Runs evaluation on the test set.
        This is typically done once after training is complete.
        """
        self.logger.info("Running final evaluation on the test set...")
        metrics = self.model.val(
            data=self.dataset_manager.get_data_yaml_path(),
            split="test",
            device=self.config.device
        )
        self.logger.info("Test evaluation completed.")
        return metrics