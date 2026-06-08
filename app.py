import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Load model
model = load_model("model_cnn_kucing_anjing.h5")

IMG_SIZE = 150

st.title("AI Deteksi Kucing dan Anjing")
st.write("Upload gambar kucing atau anjing untuk diprediksi oleh model CNN.")

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = float(prediction[0][0])

    if prob >= 0.5:
        predicted_class = "Anjing"
        confidence = prob * 100
    else:
        predicted_class = "Kucing"
        confidence = (1 - prob) * 100

    st.subheader("Hasil Prediksi")
    st.write(f"Prediksi: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")