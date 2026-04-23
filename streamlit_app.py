
import streamlit as st
st.set_page_config(page_title="ALS Smart Detection", page_icon="⚕️", layout="centered")
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

# ===== LOAD MODEL & LABEL =====
@st.cache_resource
def load_model_and_classes():
    model = load_model("dropout_s1b/best_model_fold4.keras")
    with open("classes.json", "r") as f:
        classes = json.load(f)
    return model, classes

model, classes = load_model_and_classes()
IMG_SIZE = (128, 128)

def preprocess_image(image):
    image = image.resize(IMG_SIZE).convert("RGB")
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ===== UI =====
st.title("⚕️ ALS Smart Detection")
st.write("Aplikasi deteksi ALS berbasis AI menggunakan citra spektrogram.")

menu = st.sidebar.radio("Menu", ["Home", "Prediksi", "Tim"])

if menu == "Home":
    st.header("Tentang ALS")
    st.write("""
    Amyotrophic Lateral Sclerosis (ALS) adalah penyakit neurodegeneratif progresif yang menyerang neuron motorik di otak dan sumsum tulang belakang. Kerusakan ini menyebabkan otot melemah secara bertahap, kehilangan fungsi gerak, dan pada tahap lanjut dapat mempengaruhi kemampuan berbicara, menelan, dan bernapas.
    
    Deteksi dini sangat penting untuk membantu penanganan dan meningkatkan kualitas hidup pasien. Namun, diagnosis ALS tidak selalu mudah dan membutuhkan analisis yang kompleks.
    
    **Tujuan Sistem**
    - Membantu analisis awal berbasis AI
    - Meningkatkan efisiensi penelitian
    - Mendukung pengembangan sistem diagnosis berbasis data
    
    ⚠️ Aplikasi ini alat diagnosis medis untuk screening awal penyakit ALS.
    """)

elif menu == "Prediksi":
    st.header("Upload Spektrogram")
    st.info("Gunakan gambar .png/.jpg minimal 128x128")
    uploaded_file = st.file_uploader("Upload gambar spektrogram", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Spektrogram yang diupload", use_container_width=True)
        img = preprocess_image(image)
        with st.spinner("Menganalisis spektrogram..."):
            pred = model.predict(img)[0]
        idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
        st.subheader(f"Hasil Prediksi: {classes[idx]}")
        st.write(f"Confidence: {confidence*100:.2f}%")
        if confidence > 0.8:
            st.success("Model sangat yakin terhadap hasil prediksi ini.")
        elif confidence > 0.5:
            st.warning("Model cukup yakin, namun masih perlu verifikasi lebih lanjut.")
        else:
            st.error("Model kurang yakin, hasil perlu ditinjau ulang.")
        st.progress(float(confidence))
        st.markdown("**Distribusi Probabilitas:**")
        for i, prob in enumerate(pred):
            st.write(f"{classes[i]}: {float(prob)*100:.2f}%")
            st.progress(float(prob))

elif menu == "Tim":
    st.header("Tim Peneliti")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("static/nyoman.png", width=120)
        st.markdown("**I Nyoman Switrayana**")
        st.markdown("<span style='font-size:14px;'>Ketua</span>", unsafe_allow_html=True)
    with col2:
        st.image("static/tomi.png", width=120)
        st.markdown("**Tomi Tri Sujaka**")
        st.markdown("<span style='font-size:14px;'>Anggota 1</span>", unsafe_allow_html=True)
    with col3:
        st.image("static/imelda.png", width=120)
        st.markdown("**Imelda Silpiana Putri**")
        st.markdown("<span style='font-size:14px;'>Anggota 2</span>", unsafe_allow_html=True)


st.markdown("---")
colA, colB, colC, colD = st.columns([1,1,1,2])
with colA:
    st.image("static/diktisaintek.png", width=60)
with colB:
    st.image("static/bima.png", width=60)
with colC:
    st.image("static/universitas.png", width=60)
with colD:
    st.caption("© 2026 ALS Smart Detection  ")
    st.caption("Universitas Bumigora | Penelitian Dosen Pemula")
