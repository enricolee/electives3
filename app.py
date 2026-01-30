import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Waste Type Classifier 🌱♻️🚫")

# Class mappings
class_labels = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

waste_category_map = {
    "cardboard": "Recyclable ♻️",
    "paper": "Recyclable ♻️",
    "plastic": "Non-Biodegradable🚫",
    "metal": "Non-Biodegradable🚫",
    "glass": "Non-Biodegradable🚫",
    "trash": "Non-Biodegradable 🚫"
}

# Load model
@st.cache_resource
def load_ml_model():
    return load_model("waste_classifier_6class.h5")

model = load_ml_model()

# -------------------- IMAGE UPLOAD --------------------
st.subheader("Upload Waste Image 📤")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img.resize((224,224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    label = class_labels[predicted_class]
    category = waste_category_map[label]

    st.success(f"Detected: {label}")
    st.write(f"Waste Category: {category}")
    st.write(f"Confidence: {round(confidence*100,2)}%")

# -------------------- LIVE CAMERA --------------------
st.subheader("Live Camera Waste Detection 📷")

camera_image = st.camera_input("Take a picture of the waste")

if camera_image is not None:
    img = Image.open(camera_image).convert("RGB")
    st.image(img, caption="Captured Image", use_column_width=True)

    img_array = image.img_to_array(img.resize((224,224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    label = class_labels[predicted_class]
    category = waste_category_map[label]

    st.success(f"Detected: {label}")
    st.write(f"Waste Category: {category}")
    st.write(f"Confidence: {round(confidence*100,2)}%")
