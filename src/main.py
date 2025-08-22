# main.py

import logging
from config import TrainingConfig
from trainer import Trainer
from tuning import HyperparameterTuner # Import the tuning placeholder

def main():
    """
    Main function to initialize configuration, set up logging,
    and orchestrate the training and evaluation process.
    """
    # Initialize configuration
    config = TrainingConfig()

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Gender Tracking Object Detection Project.")
    logger.info(f"Configuration for this run (ID: {config.run_id}): {config}")
    logger.info(f"Using device: {config.device}")

    # Initialize the Trainer
    detection_trainer = Trainer(config)

    # --- Training Phase ---
    logger.info("Initiating model training.")
    detection_trainer.train()
    logger.info("Model training completed.")

    # --- Validation Phase (Optional, but good for quick check) ---
    # The 'train' method already performs validation at each epoch,
    # but you can run a final dedicated validation here if needed.
    # logger.info("Performing final validation.")
    # validation_metrics = detection_trainer.validate()
    # logger.info(f"Final Validation Metrics: {validation_metrics}")

    # --- Test Phase ---
    logger.info("Performing final evaluation on the test set.")
    test_metrics = detection_trainer.test()
    logger.info(f"Final Test Metrics: {test_metrics}")
    
    # --- Hyperparameter Tuning (Optional - currently a placeholder) ---
    # tuner = HyperparameterTuner(config)
    # tuner.tune_hyperparameters()
    # logger.info("Hyperparameter tuning process completed.")

    logger.info("Project execution finished successfully.")

if __name__ == "__main__":
    main()










