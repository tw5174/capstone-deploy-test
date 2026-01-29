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
image_path = "data/28/test3.png"
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
mean_L1 = np.mean(L_vals)
std_L1 = np.std(L_vals)
median_L1 = np.median(L_vals)

mean_A1 = np.mean(A_vals)
std_A1  = np.std(A_vals)
median_A1 = np.median(A_vals)

mean_B1 = np.mean(B_vals)
std_B1  = np.std(B_vals)
median_B1 = np.median(B_vals)

#--------------------------------------------------------------------------------------------------------------------------------

image_path = "data/14/test2.png"
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
mean_L2 = np.mean(L_vals)
std_L2 = np.std(L_vals)
median_L2 = np.median(L_vals)

mean_A2 = np.mean(A_vals)
std_A2  = np.std(A_vals)
median_A2 = np.median(A_vals)

mean_B2 = np.mean(B_vals)
std_B2  = np.std(B_vals)
median_B2 = np.median(B_vals)

#--------------------------------------------------------------------------------------------------------------------

image_path = "data/0/test1.png"
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
mean_L3 = np.mean(L_vals)
std_L3 = np.std(L_vals)
median_L3 = np.median(L_vals)

mean_A3 = np.mean(A_vals)
std_A3  = np.std(A_vals)
median_A3 = np.median(A_vals)

mean_B3 = np.mean(B_vals)
std_B3  = np.std(B_vals)
median_B3 = np.median(B_vals)

#----------------------------------------------------------------------------------------------------------------

# channels = ['L', 'A', 'B']
# mean_vals = [mean_L, mean_A, mean_B]
# std_vals = [std_L, std_A, std_B]

# x = np.arange(len(channels))

# plt.bar(x - 0.15, mean_vals, 0.3, label='Mean')
# plt.bar(x + 0.15, std_vals, 0.3, label='Std')
# plt.xticks(x, channels)
# plt.ylabel('Value')
# plt.title('LAB Channel Statistics peanuts3')
# plt.ylim(0, 150)  # fixed y-axis
# plt.legend()
# plt.show()

channels = ['L', 'A', 'B']
groups = ['Run1', 'Run2', 'Run3']

# Example: each row = a run, each column = channel mean
mean_vals = np.array([
    [mean_L1, mean_A1, mean_B1],
    [mean_L2, mean_A2, mean_B2],
    [mean_L3, mean_A3, mean_B3]
])

std_vals = np.array([
    [std_L1, std_A1, std_B1],
    [std_L2, std_A2, std_B2],
    [std_L3, std_A3, std_B3]
])

median_vals = np.array([
    [median_L1, median_A1, median_B1],
    [median_L2, median_A2, median_B2],
    [median_L3, median_A3, median_B3]
])

x = np.arange(len(channels))
width = 0.08  # width of each bar

for i in range(3):  # for each run
    offset = (i - 1) * 3 * width
    plt.bar(x + offset - width, mean_vals[i], width,label=f'{groups[i]} Mean')
    plt.bar(x + offset, std_vals[i], width,label=f'{groups[i]} Std')
    plt.bar(x + offset + width, median_vals[i], width,label=f'{groups[i]} Median')

plt.xticks(x, channels)
plt.ylabel('Value')
plt.title('LAB Channel Stats over 3 Runs')
plt.ylim(0, 150)
plt.legend()
plt.show()
