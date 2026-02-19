from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- 1. Load the SAM model ----
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Download from SAM GitHub if you don't have it (link in readme)
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cpu")  # Use "cpu" if no GPU or "cuda" if nvidia gpu

# ---- 2. Load your image ----
image_path = "images_before/peanuts4.jpeg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a predictor
predictor = SamPredictor(sam)
predictor.set_image(image)

# Draw points around the peanuts
plt.figure(figsize=(10, 10))
plt.imshow(image)
points = plt.ginput(n=-1, timeout=0)  # click on points, press 'd' to stop
points = np.array(points).astype(int)

# Use the points to guide the segmentation
masks, _, _ = predictor.predict(
    point_coords=points,
    point_labels=np.ones(len(points)),
    multimask_output=True,
)

# Convert masks to a single overlay
overlay = np.zeros(image.shape[:2], dtype=np.uint8)
for mask in masks:
    overlay = np.maximum(overlay, mask.astype(np.uint8) * 255)

# ---- 3. Display the result ----
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(overlay, alpha=0.5, cmap='Reds')  # Highlights all detected objects

# Draw contours around the detected peanuts
contours, _ = cv2.findContours(overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = image.copy()
for contour in contours:
    cv2.drawContours(image_with_contours, [contour], 0, (0, 255, 0), 2)

plt.imshow(image_with_contours)
plt.axis('off')
plt.show()

# ---- 4. Optional: Save output image ----
output_path = "peanuts_outlined.png"
outlined = image_with_contours.copy()
cv2.imwrite(output_path, cv2.cvtColor(outlined, cv2.COLOR_RGB2BGR))
print(f"Outlined image saved to {output_path}")


