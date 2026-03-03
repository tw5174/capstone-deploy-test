print("BACKEND FILE LOADED")
import cv2
import numpy as np
import joblib
import os
import streamlit as st
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- CONFIGURATION ---
#SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
#RF_MODEL_PATH = "rf_peanut_maturity.joblib"
RF_MODEL_PATH = "Jacobtesting/rf_peanut_maturity.joblib"
#MODEL_TYPE = "vit_h"
MODEL_TYPE = "vit_b"

@st.cache_resource
def load_models():
    """
    Loads the SAM model and Random Forest model.
    Returns: (sam_model, rf_model) or (None, None) if failed.
    """
   # if not os.path.exists(SAM_CHECKPOINT_PATH):
   #     return None, "missing_sam"
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        import urllib.request
        with st.spinner("Downloading SAM model (first time only, ~375MB)..."):
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                SAM_CHECKPOINT_PATH
            )
    if not os.path.exists(RF_MODEL_PATH):
        return None, "missing_rf"

    try:
        # Load SAM
        print("LOADING SAM MODEL...")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        print("SAM MODEL CREATED")
        # Force CPU and float32 to avoid loose types causing interpolate errors on windows/cpu
        sam.to(device="cpu")
        print("SAM MOVED TO CPU")
        
        # Load Random Forest
        rf = joblib.load(RF_MODEL_PATH)
        
        return (sam, rf), None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_mask_generator(_sam_model):
    """
    Creates and caches the mask generator.
    """
    return SamAutomaticMaskGenerator(
        model=_sam_model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        points_per_batch=64
    )

def process_image_and_predict(image_rgb, models):
    """
    Main pipeline: Mask Generation -> Filtering -> Feature Extraction -> Prediction
    Returns: (overlay_image, prediction_days, count)
    """
    sam_model, rf_model = models
    mask_generator = get_mask_generator(sam_model)

    print("--- STARTING MASK GENERATION ---")
    # Generate Masks
    masks = mask_generator.generate(image_rgb)
    print(f"--- MASKS GENERATED: {len(masks)} found ---")
    masks = []
    
    # Thresholds (from original script)
    min_size = 300
    max_size = 5000
    min_circularity = 0.7

    overlay = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    valid_masks_count = 0
    
    # Filter masks
    for mask in masks:
        mask_size = mask['segmentation'].sum()
        if not (min_size <= mask_size <= max_size):
            continue

        mask_uint8 = mask['segmentation'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        keep_mask = False
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > min_circularity:
                keep_mask = True
                break

        if keep_mask:
            overlay = np.maximum(overlay, mask['segmentation'].astype(np.uint8) * 255)
            valid_masks_count += 1

    if valid_masks_count == 0:
        print("--- NO VALID MASKS FOUND AFTER FILTERING ---")
        return overlay, None, 0

    print(f"--- VALID MASKS: {valid_masks_count} ---")

    # Extract Features
    # Replicating the RGB split logic of the original script
    L_dummy, A, B = cv2.split(image_rgb)
    
    A_vals = A[overlay > 0]
    B_vals = B[overlay > 0]

    if len(A_vals) == 0:
        return overlay, None, 0

    features = [
        A_vals.mean(),
        np.median(A_vals),
        A_vals.std(),
        B_vals.mean(),
        np.median(B_vals),
        B_vals.std()
    ]
    
    # Predict
    prediction = rf_model.predict([features])[0]
    
    return overlay, prediction, valid_masks_count
