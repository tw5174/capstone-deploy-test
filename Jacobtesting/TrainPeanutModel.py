#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
#import cv2
#import matplotlib.pyplot as plt
import numpy as np
#import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import cross_val_predict
import joblib

X = np.load("XsubLsubMin.npy")
y = np.load("ysubLsubMin.npy")

# rf = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=4,
#     min_samples_leaf=2,
#     max_features="sqrt",
#     random_state=42
# )

# loo = LeaveOneOut()

# scores = cross_val_score(
#     rf,
#     X,
#     y,
#     cv=loo,
#     scoring="accuracy"
# )

# print("LOO accuracy:", scores.mean())
# print("Std:", scores.std())

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42
)

loo = LeaveOneOut()

scores = cross_val_score(
    rf,
    X,
    y,
    cv=loo,
    scoring="neg_mean_absolute_error"
)

mae = -scores.mean()
print("LOO MAE:", mae)

y_pred = cross_val_predict(rf, X, y, cv=loo)

for t, p in zip(y[:42], y_pred[:42]):
    print(f"True: {t:>3}  Pred: {p:5.2f}")

#rf.fit(X, y)

#joblib.dump(rf, "rf_peanut_maturity.joblib")


# rf = joblib.load("rf_peanut_maturity.joblib")

# features = extract_lab_features("new_image.png")
# prediction = rf.predict([features])

# print("Predicted maturity:", prediction[0])