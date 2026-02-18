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

mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,           # default grid resolution (higher → more detailed)
        pred_iou_thresh=0.88,         # default confidence threshold for masks
        stability_score_thresh=0.95,  # default stability filter (lower → more masks)
        points_per_batch=64           # batch size for processing points
    )

def extract_lab_features(image_path):

    #image_path = "images_before/peanuts1.jpeg"
    image = cv2.imread(image_path)
    # if image is None:
    #     print("Failed to read:", path)
    # else:
    #     print("this is what it sees: ", path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)

# def extract_lab_features(image_path):
#     img = cv2.imread(image_path)

#     if img is None:
#         raise ValueError(f"Could not load {image_path}")

#     # Convert BGR → LAB
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#     features = []
#     for i in range(3):  # L, A, B
#         channel = lab[:, :, i]
#         features.extend([
#             channel.mean(),
#             np.median(channel),
#             channel.std()
#         ])

#     return np.array(features)


# ---- 4. Combine masks into a single overlay ----
# Set your size thresholds in pixels
    min_size = 300   # reject anything smaller than this
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

    # L_vals = L[overlay > 0]
    # A_vals = A[overlay > 0]
    # B_vals = B[overlay > 0]


    # features = []
    # for i in range(3):  # L, A, B
    #     channel = lab[:, :, i]
    #     features.extend([
    #         channel.mean(),
    #         np.median(channel),
    #         channel.std()
    #     ])
    
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

X = []  #2d array, thus capital
y = []  #1d array, thus lowercase
i = 1
for class_name in ["0", "3", "5", "7", "10", "14", "17", "21", "24", "28", "31"]:
    label = int(class_name)  

    folder = os.path.join("data", class_name)
    for file in os.listdir(folder):
        if file.startswith('.'):
            continue

        path = os.path.join(folder, file)
        
        print(i, ":", path)
        i += 1
        
        features = extract_lab_features(path)

        X.append(features)
        y.append(label)


X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)
print(X[:1])
print(y[:10])


np.save("XsubLsubMin.npy", X)
np.save("ysubLsubMin.npy", y)