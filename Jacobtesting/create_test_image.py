
import cv2
import os

# Source images
folder = "images_before"
files = ["peanuts4.jpeg", "peanuts3.jpeg", "peanuts2.jpeg", "peanuts1.jpeg"]
best_file = None

# Find one that exists
for f in files:
    path = os.path.join(folder, f)
    if os.path.exists(path):
        best_file = path
        break

if best_file:
    print(f"Using {best_file}")
    img = cv2.imread(best_file)
    h, w = img.shape[:2]
    print(f"Original: {w}x{h}")
    
    # Resize to max 800px for speed
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized to: {new_w}x{new_h}")
    
    # Save optimized image
    out_path = "OPTIMIZED_TEST_IMAGE.jpg"
    cv2.imwrite(out_path, img)
    print(f"Saved to {out_path}")
else:
    print("No source images found!")
