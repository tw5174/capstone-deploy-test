from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("First 10 labels:", y[:10])
print("Unique labels:", np.unique(y))

print("First feature vector:")
print(X[0])

print("Any NaNs in X?", np.isnan(X).any())
print("Any infs in X?", np.isinf(X).any())

i = 0
print("Label:", y[i])
print("Features:", X[i])

print("Samples per class:")
print(dict(zip(*np.unique(y, return_counts=True))))

print("Feature mins:", X.min(axis=0))
print("Feature maxs:", X.max(axis=0))
