st.subheader("Live Camera Waste Detection 📷")

camera_image = st.camera_input("Take a picture of the waste")

if camera_image is not None:
    from PIL import Image
    import numpy as np
    from tensorflow.keras.preprocessing import image

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
