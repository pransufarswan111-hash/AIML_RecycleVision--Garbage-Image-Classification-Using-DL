import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Garbage Classification System",
    page_icon="♻️",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_resnet_model():
    return load_model("models/resnet50_head_only.keras")

model = load_resnet_model()

# Class names (must match training)
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("♻️ Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Introduction",
        "🧪 Model Prediction",
        "📊 Model Comparison",
        "👤 Creator Info"
    ]
)

# =================================================
# 🏠 INTRODUCTION PAGE
# =================================================
if page == "🏠 Introduction":
    st.title("♻️ Garbage Image Classification Using Deep Learning")

    st.markdown("""
    ## 📌 Project Overview

    Waste management is a major environmental challenge.  
    Manual garbage segregation is inefficient, time-consuming, and error-prone.

    ### 🎯 Objective
    To automatically classify garbage images into different categories using
    **Deep Learning and Transfer Learning techniques**.

    ### 🧠 Models Implemented
    - MobileNetV2
    - EfficientNetB0
    - ResNet50

    ### 🏆 Final Outcome
    After comparing all models, **ResNet50** was selected as the final model
    due to its superior accuracy and generalization ability.
    """)

# =================================================
# 🧪 MODEL PREDICTION PAGE
# =================================================
elif page == "🧪 Model Prediction":
    st.title("🧪 Live Garbage Classification")

    uploaded_file = st.file_uploader(
        "Upload a garbage image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
                image,
                caption="Uploaded Image",
                width=300  # adjust between 250–350 if needed
            )

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        st.success(
            f"### 🗑️ Predicted Category: **{class_names[pred_index].upper()}**"
        )
        st.info(f"Confidence: **{confidence:.2f}%**")
        
        # Top-3 predictions
        top_3_idx = predictions[0].argsort()[-3:][::-1]

        st.subheader("🔝 Top 3 Predictions")
        for i in top_3_idx:
            st.write(
                f"{class_names[i].upper()} : {predictions[0][i]*100:.2f}%"
            )

# =================================================
# 📊 MODEL COMPARISON PAGE
# =================================================
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison & Selection")

    data = {
        "Model": ["MobileNetV2", "EfficientNetB0", "ResNet50"],
        "Validation Accuracy": ["~75%", "~93%", "⭐ ~96%"],
        "Parameters": ["3.5M", "4.4M", "25.6M"],
        "Training Speed": ["Fast", "Medium", "Slow"],
        "Generalization": ["Low", "High", "Very High"]
    }

    df = pd.DataFrame(data)
    st.table(df)

    st.markdown("""
    ## 🏆 Why ResNet50?

    - Achieved the **highest validation accuracy**
    - Excellent feature extraction using **residual connections**
    - Minimal overfitting with frozen pretrained layers
    - Consistent performance across all garbage categories

    ### ✅ Final Decision
    **ResNet50** was selected as the final model for deployment.
    """)

# =================================================
# 👤 CREATOR INFO PAGE
# =================================================
elif page == "👤 Creator Info":
    st.title("👤 Creator Information")

    st.markdown("""
    ### 🎓 Project Developed By
    **Atharva Sanjay Borawake**


    ### 🛠️ Technologies Used
    - Python
    - TensorFlow & Keras
    - Transfer Learning
    - Streamlit

    ### 🌱 Future Scope
    - Real-time camera-based classification
    - Mobile application integration
    - Smart waste management systems
    """)