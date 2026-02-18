# PeanutAnalyzer 🥜

A pretty Streamlit frontend for the Peanut Maturity Prediction model.

## 🚀 How to Run

### 1. Prerequisites
Make sure you have the model files in this folder:
-   `sam_vit_h_4b8939.pth` (2.4GB)
-   `rf_peanut_maturity.joblib`

### 2. Install Dependencies
Run this command once to install the necessary libraries:
```powershell
pip install -r requirements.txt
```

### 3. Launch App
Run the following command to start the interface:
```powershell
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

### ⚠️ Note on Image Size
Since this app runs the full AI model on the CPU, large images (like 12MP phone photos) may cause it to crash or run very slowly.
**Recommendation:** Resize images to around **1024x1024 pixels** before uploading for the best experience.
