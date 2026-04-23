# ALS DETECTION

Aplikasi deteksi ALS berbasis Streamlit.

## Cara Menjalankan Lokal

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi:
   ```
   streamlit run streamlit_app.py
   ```

## File yang Diperlukan untuk Deploy
- streamlit_app.py
- requirements.txt
- classes.json
- dropout_s1b/best_model_fold4.keras
- static/ (berisi gambar logo dan tim)

## Deploy ke Streamlit Cloud
1. Push file di atas ke repository GitHub.
2. Di Streamlit Cloud, pilih repo ini dan file utama: `streamlit_app.py`.
3. Pastikan struktur folder sama seperti di repo ini.

---
© 2026 ALS Smart Detection | Universitas Bumigora
