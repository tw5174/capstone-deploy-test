import os
import cv2
import numpy as np
import joblib
import streamlit as st
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- CONFIG ---
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
RF_MODEL_PATH = "Jacobtesting/rf_peanut_maturity.joblib"
MODEL_TYPE = "vit_b"

@st.cache_resource
def load_models():
    """
    Loads SAM (ViT-B) and RF model. Heavy code runs here, not at import.
    Returns: (sam_model, rf_model), or (None, error_string)
    """
    print("Loading SAM and RF models...")
    
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        import urllib.request
        with st.spinner("Downloading SAM model (~375MB, first time only)..."):
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                SAM_CHECKPOINT_PATH
            )
    
    if not os.path.exists(RF_MODEL_PATH):
        return None, "Random Forest model not found"

    try:
        # Load SAM safely
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device="cpu")  # Force CPU
        print("SAM loaded successfully")

        # Load RF
        rf = joblib.load(RF_MODEL_PATH)
        print("Random Forest loaded successfully")
        
        return (sam, rf), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_mask_generator(_sam_model):
    """
    Creates a cached mask generator with reduced memory usage.
    """
    return SamAutomaticMaskGenerator(
        model=_sam_model,
        points_per_side=16,          # Lower to reduce RAM usage
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        points_per_batch=32          # Reduce batch size for CPU
    )


def process_image_and_predict(image_rgb, models):
    """
    Pipeline: Mask generation -> filtering -> feature extraction -> RF prediction.
    """
    sam_model, rf_model = models
    mask_generator = get_mask_generator(sam_model)

    # Optional: Resize image to max 1024 to save memory
    h, w = image_rgb.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        image_rgb = cv2.resize(image_rgb, (int(w*scale), int(h*scale)))

    print("--- BEFORE MASK GENERATION ---")
    masks = mask_generator.generate(image_rgb)
    print(f"--- MASKS GENERATED: {len(masks)} ---")

    # --- Filter masks ---
    min_size, max_size, min_circularity = 300, 5000, 0.7
    overlay = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    valid_masks_count = 0

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
        print("--- NO VALID MASKS FOUND ---")
        return overlay, None, 0

    print(f"--- VALID MASKS: {valid_masks_count} ---")

    # --- Extract RGB features for RF ---
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

    prediction_days = rf_model.predict([features])[0]
    return overlay, prediction_days, valid_masks_count
    
    # Predict
    prediction = rf_model.predict([features])[0]
    
    return overlay, prediction, valid_masks_count
