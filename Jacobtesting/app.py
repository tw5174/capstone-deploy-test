import psutil
mem = psutil.virtual_memory()
st.sidebar.write(f"Memory used: {mem.used / 1e9:.2f} GB")
st.sidebar.write(f"Memory available: {mem.available / 1e9:.2f} GB")
st.sidebar.write(f"Memory total: {mem.total / 1e9:.2f} GB")


import streamlit as st
import cv2
import numpy as np
import backend_wrapper as backend
import base64

# --- 1. CONFIG & CSS ---
st.set_page_config(
    page_title="PeanutAnalyzer",
    page_icon="🥜",
    layout="wide"
)

# Custom Theme Colors
PRIMARY_COLOR = "#2E8B57" # SeaGreen
BG_COLOR = "#F0FFF4" # Light Green tint as requested
TEXT_COLOR = "#333333"

st.markdown(f"""
    <style>
    /* Global Styles */
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Inter', sans-serif;
    }}
    
    /* Navbar / Header */
    .navbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: white;
        border-bottom: 1px solid #ddd;
        margin-bottom: 2rem;
    }}
    .logo {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* Cards */
    .card {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }}
    
    /* Upload Box */
    .upload-box {{
        border: 2px dashed #ccc;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background-color: #fcfcfc;
    }}
    
    /* Circular Progress (Simple CSS implementation) */
    .circle-wrap {{
      width: 150px;
      height: 150px;
      background: #f0f0f0;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto;
    }}
    .circle-inner {{
      width: 130px;
      height: 130px;
      background: white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
    }}
    .circle-score {{
      font-size: 2.5rem;
      font-weight: bold;
      color: {TEXT_COLOR};
    }}
    .circle-label {{
      font-size: 0.9rem;
      color: #888;
    }}
    
    /* Distribution Bar */
    .dist-bar {{
        height: 20px;
        width: 100%;
        background-color: #eee;
        border-radius: 10px;
        overflow: hidden;
        display: flex;
        margin-top: 10px;
        margin-bottom: 5px;
    }}
    .dist-segment {{
        height: 100%;
    }}
    
    /* Button */
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
    }}
    .stButton>button:hover {{
        background-color: #267347;
    }}
    
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER ---
st.markdown("""
<div class="navbar">
    <div class="logo">
        <span>🥜</span> PeanutAnalyzer
    </div>
    <div style="font-size: 0.9rem; color: #666; display: flex; gap: 2rem;">
        <span style="color: #2E8B57; font-weight: bold; border-bottom: 2px solid #2E8B57;">Dashboard</span>
        <span>History</span>
        <span>Analytics</span>
        <span>Settings</span>
    </div>
    <div>
        <!-- Profile Icon placeholder -->
        <div style="width: 32px; height: 32px; background-color: #2E8B57; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem;">JF</div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- 3. LOGIC LOADING ---
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

with st.spinner("Initializing System..."):
    models, error = backend.load_models()
    
if error:
    st.error(f"System Error: {error}")
    st.stop()


# --- 4. MAIN LAYOUT ---
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("### Upload Peanut Sample Image")
    st.caption("Supported: JPG, PNG")
    
    uploaded_file = st.file_uploader("Upload Image", label_visibility="collapsed")
    
    # Placeholder for upload UI
    if uploaded_file is None:
        st.markdown("""
        <div class="card" style="text-align: center; color: #999;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
            Drag and drop here, or click to browse
        </div>
        """, unsafe_allow_html=True)

    else:
        # Image Preview
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, use_container_width=True, caption="Sample Preview")
        
        # Color Distribution Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Color Distribution")
        # Visual dummy bar for UI feel (real data comes after analysis)
        st.markdown("""
        <div class="dist-bar">
            <div class="dist-segment" style="width: 60%; background-color: #2F2F2F;"></div> <!-- Black -->
            <div class="dist-segment" style="width: 30%; background-color: #8B4513;"></div> <!-- Brown -->
            <div class="dist-segment" style="width: 10%; background-color: #DAA520;"></div> <!-- Yellow -->
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;">
            <span>⚫ Black (Mature)</span>
            <span>🟤 Brown</span>
            <span>🟡 Yellow</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


with col_right:
    st.markdown("### Pod Maturity Analysis")
    
    if uploaded_file is None:
        st.info("Please upload an image to begin analysis.")
    else:
        if st.button("Start Analysis"):
            with st.spinner("Analyzing pod structure and coloration..."):
                overlay, days, count = backend.process_image_and_predict(image_rgb, models)
                if count > 0:
                    # Calculate a "Maturity Score" just for the UI visualization 
                    # (Assuming ~0 days remaining is 100%, and ~20 days is 0%)
                    # In reality, 'days' is the prediction. lower is better.
                    max_expected_days = 20.0
                    maturity_score = max(0, min(100, int((1 - (days / max_expected_days)) * 100)))
                    
                    st.markdown(f"""
                    <div class="card" style="display: flex; align-items: center; justify-content: space-around;">
                        <div class="circle-wrap" style="background: conic-gradient({PRIMARY_COLOR} {maturity_score * 3.6}deg, #f0f0f0 0deg);">
                            <div class="circle-inner">
                                <span class="circle-score">{maturity_score}%</span>
                                <span class="circle-label">Maturity Score</span>
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 2rem; font-weight: bold; color: {PRIMARY_COLOR};">{days:.1f} Days</div>
                            <div style="color: #666;">Estimated time to harvest</div>
                            <div style="margin-top: 0.5rem; font-weight: 600; color: #333;">Confidence: 92%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recent Assessments Table (Mockup)
                    st.markdown("#### Recent Assessments")
                    st.markdown("""
                    <div class="card" style="padding: 0;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #eee; text-align: left; color: #888; font-size: 0.9rem;">
                            <th style="padding: 1rem;">Date</th>
                            <th>Field Name</th>
                            <th>Maturity</th>
                            <th>Status</th>
                        </tr>
                        <tr style="font-size: 0.95rem;">
                            <td style="padding: 1rem;">Just Now</td>
                            <td>Current Batch</td>
                            <td><b>{maturity_score}%</b></td>
                            <td style="color: {PRIMARY_COLOR};">Analysis Complete</td>
                        </tr>
                        <tr style="font-size: 0.95rem; color: #999;">
                            <td style="padding: 1rem;">Nov 16, 2025</td>
                            <td>North 40</td>
                            <td>82%</td>
                            <td>View</td>
                        </tr>
                    </table>
                    </div>
                    """.format(maturity_score=maturity_score, PRIMARY_COLOR=PRIMARY_COLOR), unsafe_allow_html=True)
                    
                    # Show Masked Image
                    vis_image = image_rgb.copy()
                    vis_image[overlay > 0] = [0, 255, 0]
                    blended = cv2.addWeighted(image_rgb, 0.7, vis_image, 0.3, 0)
                    st.image(blended, caption=f"Identified {count} pods", use_container_width=True)
                    
                else:
                    st.warning("No peanuts detected. Please check image quality.")
