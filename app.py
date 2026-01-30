import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Waste Combustibility Classifier 🔥🚫")
st.write(
    "Upload an image of waste to identify whether it is "
    "**Combustible** or **Non-Combustible**."
)

# Internal class grouping (hidden from user)
COMBUSTIBLE_CLASSES = [0, 3, 4, 5]      # cardboard, paper, plastic, trash
NON_COMBUSTIBLE_CLASSES = [1, 2]        # glass, metal

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
    confidence = float(np.max(predictions))

    # Convert 6-class prediction into 2-class result
    if predicted_class in COMBUSTIBLE_CLASSES:
        result = "🔥 Combustible"
    else:
        result = "🚫 Non-Combustible"

    # Display results (ONLY 2 classes)
    st.success(f"Classification: **{result}**")
    st.write(f"Confidence: **{round(confidence * 100, 2)}%**")
