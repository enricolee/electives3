import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Waste Combustibility Classifier 🔥🚫")
st.write("Upload an image of waste to identify if it is **Combustible** or **Non-Combustible**.")

# Class labels from trained 6-class model
class_labels = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# Combustibility mapping
combustibility_map = {
    "cardboard": "🔥 Combustible",
    "paper": "🔥 Combustible",
    "plastic": "🔥 Combustible",
    "trash": "🔥 Combustible",
    "glass": "🚫 Non-Combustible",
    "metal": "🚫 Non-Combustible"
}

# Load model
@st.cache_resource
def load_ml_model():
    return load_model("waste_classifier_6class.h5")

model = load_ml_model()

# -------------------- IMAGE UPLOAD --------------------
st.subheader("Upload Waste Image 📤")

uploaded_file = st.file_uploader(
    "Choose an image (JPG or PNG)",
    type=["jpg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    label = class_labels[predicted_class]
    result = combustibility_map[label]

    # Display results
    st.success(f"Detected Item: **{label.capitalize()}**")
    st.write(f"Classification: **{result}**")
    st.write(f"Confidence: **{round(confidence * 100, 2)}%**")
