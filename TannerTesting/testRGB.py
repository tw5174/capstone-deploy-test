from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- 1. Load the SAM model ----
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cpu")  # or "cuda" if available

# ---- 2. Load your image ----
image_path = "images_before/peanuts2.jpeg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- 3. Generate automatic masks ----
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.7,
    points_per_batch=64
)
masks = mask_generator.generate(image)

# ---- 4. Combine masks into overlay & collect kept masks ----
min_size = 800
max_size = 5000
min_circularity = 0.63

overlay = np.zeros(image.shape[:2], dtype=np.uint8)
final_masks = []   # stores segmentation masks that pass filtering

for mask in masks:
    seg = mask['segmentation']
    mask_size = seg.sum()

    if not (min_size <= mask_size <= max_size):
        continue

    # Check circularity
    mask_uint8 = seg.astype(np.uint8) * 255
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
        overlay = np.maximum(overlay, seg.astype(np.uint8) * 255)
        final_masks.append(seg)

# ---- 5. Compute average RGB for each kept mask ----
avg_colors = []
for seg in final_masks:
    pixels = image[seg]            # all RGB pixels where mask = True
    mean_rgb = pixels.mean(axis=0) # compute average R,G,B
    avg_colors.append(mean_rgb)

# Print the results for debugging/verification
for i, color in enumerate(avg_colors):
    r, g, b = color
    print(f"Peanut {i+1}: R={r:.1f}, G={g:.1f}, B={b:.1f}")

# ---- 6. Create visualization with color swatches ----
vis = image.copy()
for seg, mean_rgb in zip(final_masks, avg_colors):
    r, g, b = mean_rgb.astype(int)
    # Find centroid of mask
    ys, xs = np.where(seg)
    cx, cy = int(xs.mean()), int(ys.mean())
    # Color swatch size
    patch_w, patch_h = 40, 40
    # Swatch location (offset to the right of centroid)
    x1 = cx + 10
    y1 = cy - patch_h // 2
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    # Bound the swatch inside the image
    y1 = max(0, y1)
    y2 = min(image.shape[0], y2)
    x1 = max(0, x1)
    x2 = min(image.shape[1], x2)

    # Draw the color swatch
    vis[y1:y2, x1:x2] = [r, g, b]

    # Optional: show centroid
    cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)

# ---- 7. Display final result ----
plt.figure(figsize=(10, 10))
plt.imshow(vis)
plt.axis("off")
plt.show()

# ---- 8. Save the visualization (optional) ----
output_path = "peanuts_avg_color_visualization.png"
cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"Visualization saved to {output_path}")





# ---- 7. Optional: Save overlay image ----
output_path = "peanuts_highlighted.png"
highlighted = image.copy()
highlighted[overlay > 0] = [255, 0, 0]
cv2.imwrite(output_path, cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR))
print(f"Highlighted image saved to {output_path}")
