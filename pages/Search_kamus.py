import streamlit as st
import pandas as pd
import re
import unicodedata

# -------------------------------
# 🔧 Konfigurasi halaman
# -------------------------------
st.set_page_config(page_title="SundaLex", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    .title-style {
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .subtitle-style {
        font-size: 18px;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 📘 Header
# -------------------------------
st.markdown('<div class="title-style">📚 SundaLex - Kamus Digital Basa Sunda</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Cari padanan kata Sunda dari kolom <b>LEMA</b> atau <b>SUBLEMA</b></div>', unsafe_allow_html=True)

# -------------------------------
# 🔄 Normalisasi teks
# -------------------------------
def normalisasi_kata(kata):
    if pd.isna(kata):
        return ""
    # Hilangkan aksen (é, è → e)
    kata = unicodedata.normalize('NFKD', kata).encode('ASCII', 'ignore').decode('utf-8')
    # Ubah ke lowercase
    kata = kata.lower()
    # Hilangkan superscript angka di akhir (urang1, urang2 → urang)
    kata = re.sub(r'\d+$', '', kata)
    # Hilangkan spasi ekstra
    return kata.strip()

# -------------------------------
# 📁 Load dataset kamus
# -------------------------------
@st.cache_data
def load_kamus():
    df = pd.read_excel("dataset/hasil_gabungan1Juli2025.xlsx")
    df["LEMA_NORM"] = df["LEMA"].fillna("").apply(normalisasi_kata)
    df["SUBLEMA_NORM"] = df["SUBLEMA"].fillna("").apply(normalisasi_kata)
    return df

df_kamus = load_kamus()

# -------------------------------
# 🔎 Pencarian
# -------------------------------
st.markdown("### 🔍 Cari Kata Sunda")
kata_input = st.text_input("Masukkan kata Sunda yang ingin dicari:", "", key="input_kata")

if kata_input:
    kata_norm = normalisasi_kata(kata_input)

    cocok = df_kamus[
        (df_kamus["LEMA_NORM"] == kata_norm) |
        (df_kamus["SUBLEMA_NORM"] == kata_norm)
    ]

    if not cocok.empty:
        st.success(f"✅ Ditemukan {len(cocok)} hasil.")
        st.dataframe(
            cocok[
                ['LEMA', 'SUBLEMA', '(HALUS/LOMA/KASAR)', 'KLAS.', 'ARTI EKUIVALEN 1', 'ARTI 1', 'SINONIM']
            ].reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.error(f"🚫 Tidak ditemukan hasil untuk kata: '{kata_input}'")
else:
    st.info("Masukkan kata di atas untuk mulai mencari.")

# -------------------------------
# 📌 Footer
# -------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray'>© 2025 SundaLex by Lestari Bahasa</p>", unsafe_allow_html=True)
