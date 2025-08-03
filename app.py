import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

st.markdown("""
<style>
/* Latar belakang dan font */
.stApp {
    background-color: #fff5f7;  /* pink pastel */
    color: #2d3436;
    font-family: 'Segoe UI', sans-serif;
}

/* Tombol */
.stButton > button {
    background-color: #81c784;  /* hijau pastel */
    color: white;
    border: none;
    padding: 0.5em 1em;
    border-radius: 8px;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background-color: #66bb6a; /* hijau sedikit lebih gelap */
}

/* Sidebar warna pink terang */
[data-testid="stSidebar"] {
    background-color: #ffe4ec !important;
}

/* Judul dan header */
h1, h2, h3 {
    color: #d81b60; /* pink tua */
}
.title-style {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    margin-top: -30px;
    color: #d81b60;
}

/* Kotak input */
.stNumberInput > div > div > input {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)


# Konfigurasi tampilan halaman
st.set_page_config(
    page_title="Diabetes Detection System",
    page_icon="🩺",
    layout="centered"
)

# Kurangi jarak judul ke atas
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model and scaler
model = joblib.load('rf_diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load metrics
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

# Sidebar Menu
st.sidebar.title("☰ Menu ")
mode = st.sidebar.radio("", ['Beranda', 'Input Manual', 'Upload CSV'])

# Halaman Beranda
if mode == 'Beranda':
    st.title('🩺Diabetes Detection System')
    st.markdown("""
    <div style='text-align: justify'>

    ### ⓘ Tentang Aplikasi

    Aplikasi ini menggunakan model algoritma <b>Random Forest Classifier</b> untuk memprediksi risiko penyakit diabetes berdasarkan data medis pasien. Tersedia dua mode input: Input Manual untuk satu pasien dan Upload CSV untuk banyak pasien sekaligus.

    > ⚠️ Hasil prediksi ini bukan merupakan diagnosis medis resmi. Untuk kepastian, tetap dianjurkan melakukan konsultasi langsung dengan tenaga medis profesional.</i>

    <b>Gunakan menu di sebelah kiri untuk memilih mode dan memulai prediksi.</b>

    </div>
    """, unsafe_allow_html=True)

    # Evaluasi Model hanya di halaman Beranda
    st.header('Evaluasi Model')
    st.write(f"**Skor Cross-Validation:** {metrics['cv_score']:.2%}")
    report = metrics['classification_report']
    st.write('**Kelas: Tidak Diabetes (0)**')
    st.write(f"Precision: {report['0']['precision']:.2f}")
    st.write(f"Recall: {report['0']['recall']:.2f}")
    st.write(f"F1-Score: {report['0']['f1-score']:.2f}")

# Halaman Input Manual
if mode == 'Input Manual':
    st.title('📝 Input Manual Data Pasien')
    st.write('Silakan masukkan data pasien untuk prediksi.')

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input('Pregnancies (Jumlah Kehamilan)', min_value=0, max_value=20, value=0,
                                    help="Jumlah kehamilan yang pernah dialami pasien")
        glucose = st.number_input('Glucose (Kadar Glukosa)', min_value=0, max_value=200, value=0,
                                help="Kadar glukosa darah (mg/dL)")
        blood_pressure = st.number_input('Blood Pressure (Tekanan Darah)', min_value=0, max_value=150, value=0,
                                        help="Tekanan darah (mmHg)")
        skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit)', min_value=0, max_value=100, value=0,
                                        help="Ketebalan lipatan kulit (mm) sebagai indikasi lemak tubuh")

    with col2:
        insulin = st.number_input('Insulin (Kadar Insulin)', min_value=0, max_value=900, value=0,
                                help="Kadar insulin dalam darah (mu U/ml)")
        bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=0.0, max_value=70.0, value=0.0,
                            help="Hitung BMI = berat badan / (tinggi badan)^2")
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function (Riwayat Keluarga)', min_value=0.0, max_value=3.0, value=0.0,
                                            help="Nilai fungsi yang mengukur riwayat diabetes dalam keluarga")
        age = st.number_input('Age (Usia)', min_value=0, max_value=120, value=0,
                            help="Usia pasien dalam tahun")

    if st.button('🔍 Prediksi Diabetes'):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f'⚠️ Pasien diprediksi **memiliki Diabetes** dengan probabilitas {prob:.2%}')
        else:
            st.success(f'✅ Pasien diprediksi **tidak memiliki Diabetes** dengan probabilitas {(1 - prob):.2%}')

# Halaman Upload CSV
if  mode == 'Upload CSV':
    st.title('📂 Upload Data CSV Pasien')
    st.markdown("""Unggah file CSV yang berisi informasi medis dari **satu atau lebih pasien** untuk memprediksi risiko diabetes secara otomatis.
                
📌 File harus berformat CSV dan memiliki kolom berikut (tanpa header khusus):
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
                
📌 Hasil prediksi akan ditampilkan dan dapat diunduh kembali setelah file berhasil diproses.""")

    uploaded_file = st.file_uploader('Pilih file csv', type='csv')

    # uploaded_file = st.file_uploader('Unggah file CSV yang berisi informasi medis dari **satu atau lebih pasien** untuk memprediksi risiko diabetes secara otomatis.', type='csv')
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        if all(col in df.columns for col in required_columns):
            X = df[required_columns]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            df['Predicted_Outcome'] = ['Diabetes' if p == 1 else 'No Diabetes' for p in predictions]
            df['Probability'] = probs

            st.success('✅ Prediksi berhasil dilakukan!')
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('📥 Download Hasil Prediksi', csv, 'hasil_prediksi.csv', 'text/csv')
        else:
            st.error('❌ CSV harus mengandung kolom: ' + ', '.join(required_columns)) 
