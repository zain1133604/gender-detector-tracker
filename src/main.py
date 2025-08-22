import torch
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import io
import logging
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_L_Weights
import os
import tempfile
from typing import List, Dict, AsyncIterator
from contextlib import asynccontextmanager
from starlette.responses import FileResponse
import torch.nn as nn # Needed for nn.Linear

# Set up logging for the API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for Models and Paths ---
HUMAN_DETECTION_MODEL_PATH = "best.pt"
GENDER_CLASSIFICATION_MODEL_PATH = "best_modeln.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HUMAN_DETECTION_IMAGE_SIZE = 640

# ======================================================================================================
# !!! CRITICAL GENDER CLASSIFIER CONFIGURATION - PLEASE DOUBLE-CHECK THESE AGAINST YOUR TRAINING CODE !!!
# ======================================================================================================

# --- 1. IMPORTANT: Preprocessing Transforms ---
# Based on your test script, we are now using the EXACT transforms for inference.
# This ensures consistency with how your gender classifier was trained.
DEFAULT_NORM_MEAN = [0.485, 0.456, 0.406] # Verified from your test script
DEFAULT_NORM_STD = [0.229, 0.224, 0.225] # Verified from your test script
GENDER_CLASSIFICATION_IMAGE_SIZE = 224 # Verified from your test script

# --- 2. IMPORTANT: Verify this order matches your gender classifier's TRAINING class indices! ---
# This is the MOST COMMON reason for gender misclassification when the model is otherwise good.
# Example from your test script: class_names = ['female', 'male'] -> suggests index 0='female', index 1='male'
# So, the current setting below is likely correct IF your training used this mapping.
# If your training mapped index 0 to 'Male' and index 1 to 'Female',
# then you MUST change this to ["Male", "Female"].
GENDER_CLASS_NAMES = ["Female", "Male"] # Current assumption: Index 0 is Female, Index 1 is Male

# ======================================================================================================
# !!! END CRITICAL CONFIGURATION SECTION !!!
# ======================================================================================================

MAX_VIDEO_DURATION_SECONDS = 60
BOUNDING_BOX_EXPANSION_PIXELS = 20 # Pixels to expand bounding box on all sides
MIN_CROP_DIMENSION_FOR_GENDER = 20 # Minimum pixels for a cropped human dimension (width or height)

# --- Adaptive Crop & Uncertain Class Thresholds ---
# If human is taller than this ratio of frame height, consider them "close" for upper body/face crop
FACE_CROP_MIN_HEIGHT_RATIO = 0.4 # e.g., if bbox height > 40% of frame height, focus on upper body/face
UPPER_BODY_CROP_RATIO = 0.6 # If close, crop to top 60% of the expanded bbox height

# If gender classification confidence is below this, label as "Uncertain"
GENDER_CONFIDENCE_THRESHOLD_UNCERTAIN = 0.60
# --- END Adaptive Crop & Uncertain Class Thresholds ---

# --- Label Drawing Constants ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9 # Adjusted font size for better fit
FONT_THICKNESS = 2 # Adjusted font thickness
LABEL_COLOR = (255, 255, 255) # White text
BACKGROUND_COLOR = (0, 0, 0) # Black background for text
LABEL_PADDING = 3 # Adjusted padding around the text background


# --- Model Setup Functions ---
def setup_human_detection_model(model_path: str, device: str):
    """
    Initializes and loads the YOLOv8 model for human detection and tracking.
    """
    logger.info(f"Loading Human Detection (YOLOv8) model from {model_path} on {device}...")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Human Detection model not found at {model_path}")
        model = YOLO(model_path)
        model.to(device)
        model.eval()
        logger.info("Human Detection model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load Human Detection model from {model_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load human detection model: {e}")

# Global variable for gender inference transform
gender_inference_transform = None

def setup_gender_classification_model(model_path: str, num_classes: int, device: torch.device):
    """
    Initializes the EfficientNetV2-L model, loads its state dict, and sets up
    the gender inference transform using the model's weights' transforms.
    """
    global gender_inference_transform # Declare global to modify
    logger.info(f"Loading Gender Classification (EfficientNetV2-L) model from {model_path} on {device}...")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gender classification model not found at {model_path}")
        
        # Load the EfficientNetV2-L model without pre-trained weights initially
        # We will load our own trained weights.
        model = models.efficientnet_v2_l(weights=None) # No default weights here

        # Modify the classifier head to match your number of classes (2 for male/female)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
        # Load the state dict from your trained model, map to device
        state_dict = torch.load(model_path, map_location=device)
        # Load state dict with strict=False to allow for minor mismatches
        # (e.g., if you only saved the classifier head during training, or if you froze layers)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval() # Set to evaluation mode
        logger.info("Gender Classification model loaded successfully.")

        # --- Define the gender_inference_transform using the exact values from your test script ---
        # This is CRITICAL for matching training preprocessing
        gender_inference_transform = transforms.Compose([
            transforms.Resize((GENDER_CLASSIFICATION_IMAGE_SIZE, GENDER_CLASSIFICATION_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_NORM_MEAN, DEFAULT_NORM_STD)
        ])
        logger.info(f"Gender classification transforms explicitly defined: {gender_inference_transform}")
        # --- End Transform Definition ---

        return model
    except FileNotFoundError:
        logger.error(f"Gender classification model checkpoint not found at {model_path}.")
        raise HTTPException(status_code=500, detail=f"Gender classification model not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load Gender Classification model from {model_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load gender classification model: {e}")


# Global model variables
human_detection_model = None
gender_classification_model = None

# --- Lifespan Context Manager for FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Context manager to handle startup and shutdown events for the FastAPI application.
    Loads models on startup.
    """
    global human_detection_model, gender_classification_model

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Models will load and run on CPU, which will be significantly slower for video processing.")

    # --- Startup Log Critical Values ---
    logger.info(f"App Startup: GENDER_CONFIDENCE_THRESHOLD_UNCERTAIN set to {GENDER_CONFIDENCE_THRESHOLD_UNCERTAIN}")
    logger.info(f"App Startup: GENDER_CLASS_NAMES set to {GENDER_CLASS_NAMES}")
    logger.info(f"App Startup: DEFAULT_NORM_MEAN set to {DEFAULT_NORM_MEAN}")
    logger.info(f"App Startup: DEFAULT_NORM_STD set to {DEFAULT_NORM_STD}")
    logger.info(f"App Startup: GENDER_CLASSIFICATION_IMAGE_SIZE set to {GENDER_CLASSIFICATION_IMAGE_SIZE}")
    # --- End Startup Log Critical Values ---


    human_detection_model = setup_human_detection_model(HUMAN_DETECTION_MODEL_PATH, DEVICE)
    gender_classification_model = setup_gender_classification_model(GENDER_CLASSIFICATION_MODEL_PATH, len(GENDER_CLASS_NAMES), DEVICE)
    
    yield # Application runs
    
    # Clean up (optional, but good practice for larger models or if releasing GPU memory)
    logger.info("Shutting down API. Releasing models...")
    del human_detection_model
    del gender_classification_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear CUDA cache if on GPU


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Video Analysis API (Human Detection, Tracking & Gender Classification)",
    description="A FastAPI application to detect and track humans in videos, and classify their gender using YOLOv8 and EfficientNetV2-L. Supports videos up to 1 minute.",
    version="0.1.0",
    lifespan=lifespan # Register the lifespan context manager
)


# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """
    Checks if the API is running and both models are loaded.
    """
    return {
        "status": "API is running",
        "human_detection_model_loaded": human_detection_model is not None,
        "gender_classification_model_loaded": gender_classification_model is not None,
        "device": str(DEVICE)
    }

@app.post("/process_video/", summary="Process Video for Human Detection, Tracking & Gender Classification")
async def process_video(file: UploadFile = File(...)):
    """
    Receives a video file, processes it frame by frame for human detection,
    tracking, and gender classification. Returns the processed video.
    Maximum video duration: 1 minute.
    """
    if human_detection_model is None or gender_classification_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait for startup.")
    if gender_inference_transform is None:
        raise HTTPException(status_code=503, detail="Gender classification transforms not loaded yet. Please wait for startup.")

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    logger.info(f"Received video file: {file.filename}, Content-Type: {file.content_type}")

    temp_input_path = None
    temp_output_path = None
    cap = None
    out = None
    file_response_sent = False # Flag to track if FileResponse was successfully returned

    try:
        # Create a temporary input file
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
        with temp_input_file as f:
            file_contents = await file.read()
            f.write(file_contents)
        temp_input_path = temp_input_file.name
        logger.info(f"Video saved to temporary path: {temp_input_path}")

        # Open the video file with OpenCV to check duration
        cap_check_duration = cv2.VideoCapture(temp_input_path)
        if not cap_check_duration.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file for duration check.")

        frame_count = int(cap_check_duration.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_check = cap_check_duration.get(cv2.CAP_PROP_FPS)
        cap_check_duration.release() # Release the handle immediately after checking duration

        if fps_check > 0:
            duration = frame_count / fps_check
            logger.info(f"Video duration: {duration:.2f} seconds.")
            if duration > MAX_VIDEO_DURATION_SECONDS:
                raise HTTPException(status_code=400, detail=f"Video duration exceeds maximum allowed of {MAX_VIDEO_DURATION_SECONDS} seconds.")
        else:
            logger.warning("Could not determine video FPS, skipping duration check. Proceeding with caution.")


        # Re-open the video file for processing
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file for processing.")

        # Get video properties for output
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for output video (.mp4)

        # Create a temporary output file
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output_path = temp_output_file.name
        temp_output_file.close() # Close to allow VideoWriter to open it
        logger.info(f"Output video will be saved to temporary path: {temp_output_path}")

        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise HTTPException(status_code=500, detail="Could not create video writer.")
        
        frame_idx = 0
        
        gender_id_map = {}
        # Keep track of the last assigned ID for each gender to prevent assigning
        # the same ID to multiple people if they exit and re-enter.
        female_id_counter = 0
        male_id_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = human_detection_model.track(
                frame, 
                persist=True,
                imgsz=HUMAN_DETECTION_IMAGE_SIZE,
                conf=0.5,
                iou=0.45,
                tracker='bytetrack.yaml', # Ensure this file is available or compatible with your Ultralytics setup
                device=DEVICE
            )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                class_names_map = human_detection_model.names

                # A new dictionary to store the current frame's gender classifications
                current_frame_genders = {}

                for box_data in results[0].boxes:
                    bbox_xyxy = box_data.xyxy[0].cpu().numpy().astype(int)
                    track_id = int(box_data.id[0].item())
                    confidence = float(box_data.conf[0].item())
                    class_id = int(box_data.cls[0].item())
                    class_name = class_names_map.get(class_id, "unknown")

                    if class_name.lower() == 'person' or class_name.lower() == 'human':
                        x1_orig, y1_orig, x2_orig, y2_orig = bbox_xyxy # Original detection box

                        # --- Expand bounding box by N pixels ---
                        x1_exp = max(0, x1_orig - BOUNDING_BOX_EXPANSION_PIXELS)
                        y1_exp = max(0, y1_orig - BOUNDING_BOX_EXPANSION_PIXELS)
                        x2_exp = min(width, x2_orig + BOUNDING_BOX_EXPANSION_PIXELS)
                        y2_exp = min(height, y2_orig + BOUNDING_BOX_EXPANSION_PIXELS)
                        # --- End Expansion ---

                        gender_label = "N/A"
                        gender_confidence_str = "N/A"

                        # Check if expanded crop is valid and large enough for gender classification
                        if (x2_exp > x1_exp + MIN_CROP_DIMENSION_FOR_GENDER and
                            y2_exp > y1_exp + MIN_CROP_DIMENSION_FOR_GENDER):
                            
                            # --- ADAPTIVE CROP LOGIC ---
                            cropped_y1 = y1_exp
                            cropped_y2 = y2_exp
                            
                            current_bbox_height = y2_exp - y1_exp
                            
                            # Determine if human is 'close' enough to crop upper body/face
                            if current_bbox_height / height > FACE_CROP_MIN_HEIGHT_RATIO:
                                # If human is 'close' (tall in frame), crop upper body/face
                                cropped_y2 = y1_exp + int(current_bbox_height * UPPER_BODY_CROP_RATIO)
                                logger.debug(f"Track {track_id} Frame {frame_idx}: Cropping upper {UPPER_BODY_CROP_RATIO*100:.0f}% for close human.")
                            else:
                                # If human is 'far', use full expanded bbox height for more context
                                logger.debug(f"Track {track_id} Frame {frame_idx}: Using full expanded bbox for far human.")

                            # Ensure cropped_y2 doesn't go below cropped_y1 after calculation
                            cropped_y2 = max(cropped_y1 + MIN_CROP_DIMENSION_FOR_GENDER, cropped_y2)

                            cropped_human = frame[cropped_y1:cropped_y2, x1_exp:x2_exp]
                            # --- END ADAPTIVE CROP LOGIC ---

                            if cropped_human is not None and cropped_human.shape[0] > 0 and cropped_human.shape[1] > 0:
                                try:
                                    pil_image = Image.fromarray(cv2.cvtColor(cropped_human, cv2.COLOR_BGR2RGB))
                                    
                                    # Use the dynamically loaded gender_inference_transform
                                    gender_input_tensor = gender_inference_transform(pil_image).unsqueeze(0).to(DEVICE)
                                    
                                    with torch.no_grad():
                                        gender_outputs = gender_classification_model(gender_input_tensor)
                                        gender_probabilities = torch.softmax(gender_outputs, dim=1)[0]
                                        
                                        # Get the predicted class index (0 or 1)
                                        gender_predicted_idx = torch.argmax(gender_probabilities, 0)
                                        # Get the confidence for that specific predicted class
                                        gender_confidence_value = gender_probabilities[gender_predicted_idx.item()].item()


                                    # --- "UNCERTAIN" CLASS LOGIC ---
                                    if gender_confidence_value < GENDER_CONFIDENCE_THRESHOLD_UNCERTAIN:
                                        gender_label = "Uncertain"
                                        gender_confidence_str = f"Low ({gender_confidence_value:.2f})"
                                        logger.debug(f"Track {track_id} Frame {frame_idx}: Low confidence prediction, set to Uncertain.")
                                    else:
                                        gender_label = GENDER_CLASS_NAMES[gender_predicted_idx.item()]
                                        gender_confidence_str = f"{gender_confidence_value:.2f}"
                                    # --- END "UNCERTAIN" CLASS LOGIC ---

                                    # --- DEBUGGING LOG ---
                                    logger.info(f"Track ID: {track_id}, Frame: {frame_idx}, "
                                                f"Predicted Index: {gender_predicted_idx.item()}, "
                                                f"Confidence: {gender_confidence_value:.2f}, "
                                                f"Assigned Label: {gender_label}")
                                    # --- END DEBUGGING LOG ---

                                    # Check and update the gender ID based on the latest classification
                                    # This is the FIX for the bug where IDs get stuck
                                    if gender_label == "Female":
                                        if track_id not in gender_id_map:
                                            female_id_counter += 1
                                            gender_id_map[track_id] = f"Female{female_id_counter}"
                                        else:
                                            # If ID already exists, make sure it's not a generic ID
                                            # and update it if a confident classification is found.
                                            if not gender_id_map[track_id].startswith("Female"):
                                                gender_id_map[track_id] = f"Female{female_id_counter + 1}"
                                                female_id_counter += 1
                                    elif gender_label == "Male":
                                        if track_id not in gender_id_map:
                                            male_id_counter += 1
                                            gender_id_map[track_id] = f"Male{male_id_counter}"
                                        else:
                                            if not gender_id_map[track_id].startswith("Male"):
                                                gender_id_map[track_id] = f"Male{male_id_counter + 1}"
                                                male_id_counter += 1
                                    elif gender_label == "Uncertain":
                                        # For Uncertain, we don't assign a new gender ID, but we do update the map if the previous state was Female/Male
                                        if track_id not in gender_id_map or not gender_id_map[track_id].startswith(("Female", "Male")):
                                            gender_id_map[track_id] = f"UncertainID:{track_id}"
                                    else:
                                        # Fallback for errors or invalid classifications
                                        if track_id not in gender_id_map:
                                            gender_id_map[track_id] = f"ID: {track_id}"

                                except Exception as e:
                                    logger.warning(f"Failed to classify gender for track_id {track_id} (Frame {frame_idx}): {e}")
                                    gender_label = "Error"
                                    gender_confidence_str = "N/A"
                                    if track_id not in gender_id_map:
                                        gender_id_map[track_id] = f"ID: {track_id}"
                            else:
                                gender_label = "Crop Invalid"
                                gender_confidence_str = "N/A"
                                if track_id not in gender_id_map:
                                    gender_id_map[track_id] = f"ID: {track_id}"
                        else:
                            gender_label = "Too Small/Invalid"
                            gender_confidence_str = "N/A"
                            if track_id not in gender_id_map:
                                gender_id_map[track_id] = f"ID: {track_id}"


                        # Draw bounding box and labels on the frame (using original bbox for display)
                        color = (0, 255, 0) # Default Green
                        if gender_label == "Uncertain":
                            color = (0, 165, 255) # Orange for uncertain
                        elif gender_label == "Female":
                            color = (255, 0, 255) # Magenta for female
                        elif gender_label == "Male":
                            color = (255, 255, 0) # Cyan for male
                        
                        cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)

                        display_id = gender_id_map.get(track_id, f"ID:{track_id}")
                        label_text = f"{display_id} ({gender_confidence_str})"
                        
                        # --- Draw text with background for better visibility ---
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
                        
                        # Calculate position for the text background rectangle
                        text_x = x1_orig
                        text_y_for_text_line = y1_orig - LABEL_PADDING - text_height - baseline

                        # Adjust if text goes above the frame top
                        if text_y_for_text_line < 0:
                            text_y_for_text_line = y1_orig + text_height + LABEL_PADDING # Place below if not enough space above
                            text_y_for_text_line = min(height - 1 - baseline - LABEL_PADDING, text_y_for_text_line) # Clamp bottom

                        # Adjust text_x to prevent going off left/right edge
                        text_x = max(0, text_x) # Clamp left edge
                        if text_x + text_width + 2*LABEL_PADDING > width:
                            text_x = width - (text_width + 2*LABEL_PADDING)
                            text_x = max(0, text_x) # Re-clamp left if shifted too far right

                        # Draw background rectangle
                        cv2.rectangle(frame, 
                                      (text_x - LABEL_PADDING, text_y_for_text_line - text_height - LABEL_PADDING), 
                                      (text_x + text_width + LABEL_PADDING, text_y_for_text_line + baseline + LABEL_PADDING), 
                                      BACKGROUND_COLOR, 
                                      cv2.FILLED)
                        
                        # Draw the text itself
                        cv2.putText(frame, label_text, 
                                    (text_x, text_y_for_text_line + baseline), 
                                    FONT, FONT_SCALE, LABEL_COLOR, FONT_THICKNESS)
                        # --- END NEW ---
            
            out.write(frame)
            frame_idx += 1

        logger.info(f"Finished processing {frame_idx} frames. Output saved to {temp_output_path}")

        file_response_sent = True 
        return FileResponse(path=temp_output_path, media_type="video/mp4", filename=f"processed_{file.filename}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An error occurred during video processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
    finally:
        if cap is not None:
            cap.release()
            logger.info("Released video capture object.")
        if out is not None:
            out.release()
            logger.info("Released video writer object.")
        
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError as e:
                logger.warning(f"Could not remove temporary input file {temp_input_path}: {e}")
        
        if temp_output_path and os.path.exists(temp_output_path) and not file_response_sent:
            try:
                os.remove(temp_output_path)
            except OSError as e:
                logger.warning(f"Could not remove temporary output file {temp_output_path}: {e}")
