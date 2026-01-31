from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
# import matplotlib.pyplot as plt
import numpy as np
#import os
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import LeaveOneOut, cross_val_score
# from sklearn.model_selection import cross_val_predict
import joblib


# ---- 1. Load the SAM model ----
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Download from SAM GitHub if you don't have it (link in readme)
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cpu")  # Use "cpu" if no GPU or "cuda" if nvidia gpu

# ---- 2. Load your image ----
# image_path = "images_before/peanuts1.jpeg"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- 3. Generate automatic masks ----
# mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,           # default grid resolution (higher → more detailed)
    pred_iou_thresh=0.88,         # default confidence threshold for masks
    stability_score_thresh=0.95,  # default stability filter (lower → more masks)
    points_per_batch=64           # batch size for processing points
)

def extract_lab_features(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

# ---- 4. Combine masks into a single overlay ----
# Set your size thresholds in pixels
    min_size = 800   # reject anything smaller than this
    max_size = 5000  # reject anything larger than this
    min_circularity = 0.7 # 1 is a perfect circle, 0 is a line

    overlay = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        mask_size = mask['segmentation'].sum()
        if not (min_size <= mask_size <= max_size):
            continue

        # Convert mask to uint8 for contour detection
        mask_uint8 = mask['segmentation'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute circularity for each contour
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

# ----- 7. Turn RGB values into LAB values
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(image)

    A_vals = A[overlay > 0]
    B_vals = B[overlay > 0]

    features = [
        A_vals.mean(),
        np.median(A_vals),
        A_vals.std(),
        B_vals.mean(),
        np.median(B_vals),
        B_vals.std()
    ]

    return np.array(features)




rf = joblib.load("rf_peanut_maturity.joblib")

features = extract_lab_features("images_before/peanuts1.jpeg") # Insert new image here
prediction = rf.predict([features])

print("Predicted maturity:", prediction[0], "Days till pick")