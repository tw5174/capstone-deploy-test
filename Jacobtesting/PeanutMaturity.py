from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
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
image_path = "images_before/peanuts1.jpeg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- 3. Generate automatic masks ----
# mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,           # default grid resolution (higher → more detailed)
    pred_iou_thresh=0.88,         # default confidence threshold for masks
    stability_score_thresh=0.95,  # default stability filter (lower → more masks)
    points_per_batch=64           # batch size for processing points
)
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

# ---- 5. Display the result ----
#plt.figure(figsize=(10, 10))
#plt.imshow(image)
#plt.imshow(overlay, alpha=0.5, cmap='Reds')  # Highlights all detected objects
#plt.axis('off')
#plt.show()

# ---- 6. Optional: Save overlay image ----
#output_path = "peanuts_highlighted.png"
#highlighted = image.copy()
#highlighted[overlay > 0] = [255, 0, 0]  # Highlight in red
#cv2.imwrite(output_path, cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR))
#print(f"Highlighted image saved to {output_path}")



# ----- 7. Turn RGB values into LAB values
#lab = cv2.cvtColor(image_path, cv2.COLOR_RGB2LAB)
L, A, B = cv2.split(image)

L_vals = L[overlay > 0]
A_vals = A[overlay > 0]
B_vals = B[overlay > 0]

# plt.imshow(L * (overlay > 0), cmap="gray")
# plt.colorbar()
# plt.title("LAB L channel (masked)")
# plt.axis("off")
# plt.show()

# plt.imshow(A * (overlay > 0), cmap="gray")
# plt.colorbar()
# plt.title("LAB A channel (masked)")
# plt.axis("off")
# plt.show()

# plt.imshow(B * (overlay > 0), cmap="gray")
# plt.colorbar()
# plt.title("LAB B channel (masked)")
# plt.axis("off")
# plt.show()

# ----- 8. Extract Color Statistics
mean_L = np.mean(L_vals)
std_L = np.std(L_vals)

mean_A = np.mean(A_vals)
std_A  = np.std(A_vals)

mean_B = np.mean(B_vals)
std_B  = np.std(B_vals)

channels = ['L', 'A', 'B']
mean_vals = [mean_L, mean_A, mean_B]
std_vals = [std_L, std_A, std_B]

x = np.arange(len(channels))

plt.bar(x - 0.15, mean_vals, 0.3, label='Mean')
plt.bar(x + 0.15, std_vals, 0.3, label='Std')
plt.xticks(x, channels)
plt.ylabel('Value')
plt.title('LAB Channel Statistics')
plt.ylim(0, 150)  # fixed y-axis
plt.legend()
plt.show()