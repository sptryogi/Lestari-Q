import streamlit as st
st.set_page_config(page_title="Lestari Bahasa", page_icon="🌐", layout="centered")  # atau "centered"

import pandas as pd
import re
import pybase64
from AI_chatbot import generate_text_qwen, call_groq_api, kapitalisasi_awal_kalimat, bersihkan_superscript, ekstrak_teks, hitung_token
from constraint1 import highlight_text, constraint_text, ubah_ke_lema, find_the_lema_pair, cari_arti_lema, bersihkan_kamus, koreksi_typo_dari_respon, ganti_sinonim_berdasarkan_tingkat
import streamlit.components.v1 as components
from supabase_helper import *
import uuid, time

# st.set_page_config(page_title="Lestari Bahasa", page_icon="🌐", layout="centered")  # atau "centered"
    
# UI Styling
st.markdown("""
    <style>
    /* Hilangkan seluruh header (logo, GitHub, Share, Fork) */
    header {visibility: hidden !important;}
    [data-testid="stHeader"] {display: none;}

    /* Hilangkan menu titik tiga */
    #MainMenu {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}

    /* Hilangkan footer (Made with Streamlit) */
    footer {visibility: hidden !important;}
    .st-emotion-cache-1v0mbdj {display: none !important;}
    .st-emotion-cache-164pbft {display: none !important;}
    
    /* Hilangkan semua footer */
    footer, .st-emotion-cache-1v0mbdj, .st-emotion-cache-164pbft {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hilangkan elemen Fork/GitHub khusus */
    a[href*="github.com"] {display: none !important;}

    /* Sembunyikan logo SVG (dalam class _link_gzau3_10) */
    ._container_gzau3_1 _viewerBadge_nim44_23 {
        display: none !important;
    }

    /* Sembunyikan avatar profil pembuat aplikasi */
    ._profilePreview_gzau3_63 {
        display: none !important;
    }
    
    /* Hilangkan elemen GitHub (termasuk avatar foto) */
    a[href*="github.com"], img[src*="githubusercontent"] {
        display: none !important;
    }

    /* Target tombol collapse/expand sidebar */
    [data-testid="stSidebarCollapsedControl"] button {
        color: green !important;  /* ganti warna ikon */
        background-color: white !important;  /* opsional: latar belakang tombol */
        border-radius: 8px;
        padding: 6px;
    }

    /* Ganti warna saat hover juga */
    [data-testid="stSidebarCollapsedControl"] button:hover {
        background-color: #ccffcc !important;
        color: darkgreen !important;
    }

    /* Tambahan agar ikon lebih besar */
    button[data-testid="collapsedControl"] svg {
        width: 1.2rem;
        height: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

def set_background_from_file(file_path):
    with open(file_path, "rb") as img_file:
        b64 = pybase64.b64encode(img_file.read()).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-color: white;
            color: black;
            font-family: Arial, sans-serif;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background_from_file("dataset/bg biru.jpg")

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: black;
        font-family: Arial, sans-serif;
    }
    .chat-scroll {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        display: flex;
        flex-direction: column;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
    }

    textarea {
        text-align: justify !important;
        resize: none !important;
    }
    .chat-container-outer {
        height: calc(100vh - 180px); /* beri ruang untuk input tetap tampil */
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        padding: 10px 20px;
        border: 1px solid #444;
        border-radius: 10px;
        margin-bottom: 10px;
        background-color: #121212;
    }

    .chat-bubble-user {
        background-color: white;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-end;
        margin-left: auto;
        font-size: 20px;
        border: 2px solid black;
        text-align: left;
        line-height: 20px;
    }

    .chat-bubble-bot {
        background-color: white;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 90%;
        align-self: flex-start;
        margin-right: auto;
        font-size: 18px;
        font-family: Times New Roman;
        border: 2px solid black;
        text-align: left;
        line-height: 20px;
    }

    .fixed-input {
        position: sticky;
        bottom: 0;
        background-color: #1E1E2F;
        padding-top: 10px;
        border-top: 1px solid #444;
    }

    .stTextInput>div>div>input {
        background-color: white;
        color: black;
        border-radius: 10px;
        border: 2px solid black;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 5px 20px;
        font-weight: bold;
    }

    .yellow-note {
        color: #ffd700;
        font-size: 0.9em;
    }

    .scroll-to-bottom {
        position: fixed;
        bottom: 80px; /* Di atas input tetap */
        right: 30px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 50%;
        padding: 10px 14px;
        font-size: 20px;
        cursor: pointer;
        z-index: 1000;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
        display: none;
    }

    .scroll-to-bottom.show {
        display: block;
    }

    /* Ubah tombol 📎 jadi hijau */
    button[data-testid="stBaseButton-secondary"] {
        background-color: #28a745 !important;  /* hijau */
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.25rem 0.75rem !important;
    }

    /* Tambahan efek hover */
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #218838 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fungsi render topbar yang akan menampilkan login/logout
def render_topbar():     
    if st.session_state.get("user"):
        # Tampilan setelah login (email + tombol logout)
        st.markdown(
            f"""
            <div style='position: fixed; top: 10px; right: 20px; z-index:9999; display: flex; align-items: center; color: white; gap: 8px;'>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px;'>
                    👤 {st.session_state["email"]}
                </span>
                <a href="/login?logout=true&t={int(time.time())}" target="_self">
                    <button style='
                        background: #ff4b4b;
                        color: white;
                        border: none;
                        padding: 6px 12px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-weight: 500;
                    '>Logout</button>
                </a>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Tampilan sebelum login (tombol login)
        st.markdown(
            """
            <div style='position: fixed; top: 10px; right: 20px; z-index:9999;'>
                <a href="/login" target="_self">
                    <button style='
                        background: #4CAF50;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-weight: 500;
                    '>Login</button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

# Panggil topbar
render_topbar()

# Load kamus
# df_kamus = pd.read_excel("dataset/data_kamus_full_14-5-25.xlsx")
# df_kamus = pd.read_excel("dataset/hasil_gabungan16Jun.xlsx")
# df_kamus[['ARTI EKUIVALEN 1', 'ARTI 1']] = df_kamus[['ARTI EKUIVALEN 1', 'ARTI 1']].apply(lambda col: col.str.lower())
# df_idiom = pd.read_excel("dataset/data_idiom (3).xlsx")
# df_kamus = bersihkan_kamus(df_kamus)
# df_kamus["LEMA"] = df_kamus["LEMA"].fillna("").astype(str).apply(bersihkan_superscript)
# df_kamus["SUBLEMA"] = df_kamus["SUBLEMA"].fillna("").astype(str).apply(bersihkan_superscript)

@st.cache_data
def load_kamus_dan_idiom():
    df_kamus = pd.read_excel("dataset/hasil_gabungan1Juli2025.xlsx")
    df_kamus[['ARTI EKUIVALEN 1', 'ARTI 1']] = df_kamus[['ARTI EKUIVALEN 1', 'ARTI 1']].apply(lambda col: col.str.lower())
    df_kamus = bersihkan_kamus(df_kamus)
    df_kamus["LEMA"] = df_kamus["LEMA"].fillna("").astype(str).apply(bersihkan_superscript)
    df_kamus["SUBLEMA"] = df_kamus["SUBLEMA"].fillna("").astype(str).apply(bersihkan_superscript)
    df_idiom = pd.read_excel("dataset/data_idiom (3).xlsx")
    df_pemendekan = pd.read_excel("dataset/PEMENDEKAN-Lema.xlsx")
    return df_kamus, df_idiom, df_pemendekan

df_kamus, df_idiom, df_pemendekan = load_kamus_dan_idiom()

# def auth_guard():
#     if "user" not in st.session_state:
#         session = supabase.auth.get_session()
#         if session and session.user:
#             st.session_state["user"] = session.user
#             st.session_state["email"] = session.user.email
#         else:
#             st.warning("Silakan login terlebih dahulu.")
#             st.session_state.clear()
#             st.switch_page("pages/login.py")
def auth_guard():
    if "user" not in st.session_state:
        st.warning("Silakan login terlebih dahulu.")
        st.switch_page("pages/login.py")
    else:
        # SET session dengan token user
        supabase.auth.set_session(
            st.session_state["access_token"],
            st.session_state["refresh_token"]
        )
        current_session = supabase.auth.get_session()
        if not current_session or current_session.user.id != st.session_state["user"].id:
            st.warning("Sesi Anda tidak valid. Harap login ulang.")
            st.session_state.clear()
            st.switch_page("pages/login.py")

auth_guard()
        
if "show_file_uploader" not in st.session_state:
    st.session_state.show_file_uploader = False

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "last_chat_mode" not in st.session_state:
    st.session_state.last_chat_mode = None

if "sudah_disapa" not in st.session_state:
    st.session_state.sudah_disapa = False

st.markdown("<h1 style='color:white'>Lestari Bahasa</h1>", unsafe_allow_html=True)
st.markdown("""
<style>
/* Container utama radio button */
div[data-testid="stRadio"] {
    margin-top: 2px !important;  /* Jarak dari judul di atas */
    margin-bottom: 2px !important;
}

/* Baris radio button */
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 8px !important;  /* Lebih rapat dari sebelumnya (12px) */
    align-items: center;
}

/* Label bahasa */
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    font-size: 18px !important;  /* Sedikit lebih kecil */
    font-weight: bold !important;
    color: white !important;
    margin: 0 !important;
    padding: 0 4px !important;  /* Padding lebih kecil */
    line-height: 1.2 !important;
}

/* Warna saat dipilih */
div[data-testid="stRadio"] input[type="radio"]:checked + div div[data-testid="stMarkdownContainer"] p {
    color: #FFD700 !important;
    text-shadow: 0 0 2px rgba(255, 215, 0, 0.5);
}
</style>
""", unsafe_allow_html=True)

bahasa_list = ["Sunda", "Indonesia", "English"]

if "mode_bahasa" not in st.session_state:
    st.session_state.mode_bahasa = "Sunda"

mode_bahasa = st.radio(
    "Pilih Mode Bahasa:",
    bahasa_list,
    horizontal=True,
    index=bahasa_list.index(st.session_state.mode_bahasa),
    label_visibility="collapsed",
    key="mode_bahasa_radio"
)

st.session_state.mode_bahasa = mode_bahasa

st.markdown("""
<span style='
    color: white;
    text-shadow: 
        -1px -1px 0 #00008B,
         1px -1px 0 #00008B,
        -1px  1px 0 #00008B,
         1px  1px 0 #00008B;
    font-size: 12px;
'>
    Selamat datang! Silakan ajukan pertanyaan.
</span>
""", unsafe_allow_html=True)

# ========== Sidebar Controls ==========
with st.sidebar:
    st.header("⚙️ Pengaturan")

    option = st.selectbox(
        "Pilih Fitur",
        ["Chatbot", "Terjemah Indo → Sunda", "Terjemah Sunda → Indo"],
        key="fitur_selector"
    )

    fitur = "chatbot"
    if option == "Chatbot":
        fitur = "chatbot"
    elif option == "Terjemah Indo → Sunda":
        fitur = "terjemahindosunda"
    else:
        fitur = "terjemahsundaindo"

    chat_mode = st.selectbox(
        "Pilih Mode Chat",
        ["Ngobrol", "Belajar"],
        key="chat_mode"
    )
    # Deteksi perubahan mode belajar
    if (fitur == "chatbot" and chat_mode == "Belajar" and mode_bahasa == "Sunda" and st.session_state.last_chat_mode != "Belajar"):
        st.session_state.sudah_disapa = False
        st.session_state.last_chat_mode = "Belajar"
        st.rerun()
        
    status = st.toggle("🔍 Lihat Constraint")

    room_options = [f"room-{i}" for i in range(1, 6)]
    current_room = st.session_state.get("room", "room-1")
    room_labels = []

    for r in room_options:
        # preview = get_first_chat_preview(st.session_state.user.id, r)
        try:
            preview = get_first_chat_preview(st.session_state.user.id, r)
        except Exception as e:
            st.warning("Tidak bisa menampilkan chat sebelumnya.")
            preview = ""
        room_labels.append(f"💬 {preview}")

    selected_label = st.selectbox(
        "Pilih Chat-Room",
        room_labels,
        index=room_options.index(current_room)
    )

    # Temukan room_id berdasarkan label
    selected_room = room_options[room_labels.index(selected_label)]

    if selected_room != current_room:
        st.session_state.room = selected_room
        st.rerun()
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inisialisasi session state jika belum ada
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Deteksi perubahan mode belajar
if (
    fitur == "chatbot"
    and chat_mode == "Belajar"
    and mode_bahasa == "Sunda"
    and st.session_state.last_chat_mode != "Belajar"
):
    st.session_state.sudah_disapa = False
    st.session_state.last_chat_mode = "Belajar"
    st.rerun()

# Fungsi untuk mengosongkan input
def clear_input():
    if "user_input" in st.session_state:
        st.session_state["user_input"] = ""

# Modifikasi handle_send()
def handle_send():
    # if "user" not in st.session_state or "email" not in st.session_state:
    #     st.error("Silakan login terlebih dahulu")
    #     st.warning("Silakan login terlebih dahulu.")
    #     st.stop()
    #     return
    if "user" not in st.session_state or "email" not in st.session_state:
        st.error("Sesi Anda tidak ditemukan. Silakan login kembali.")
        st.stop()
        return

    # Dapatkan sesi terbaru dari Supabase untuk memastikan token masih valid
    try:
        supabase.auth.set_session(
            st.session_state["access_token"],
            st.session_state["refresh_token"]
        )
        current_session = supabase.auth.get_session()
        # Jika tidak ada sesi aktif di Supabase, hentikan proses
        if current_session is None or current_session.user is None:
            st.warning("Sesi Anda telah berakhir. Harap segarkan halaman atau login kembali.")
            # Hapus session state yang tidak valid untuk memaksa login ulang
            del st.session_state['user']
            del st.session_state['email']
            st.rerun() # Jalankan ulang aplikasi untuk memicu auth_guard
            return
        
        # (Opsional tapi direkomendasikan) Cocokkan ID user di session state dengan sesi Supabase
        if current_session.user.id != st.session_state.user.id:
            st.error("Terjadi ketidaksesuaian sesi. Silakan login kembali.")
            st.session_state.clear()
            st.rerun()
            return

    except Exception as e:
        st.error(f"Gagal memvalidasi sesi: {e}. Silakan coba lagi.")
        return

    extracted = ""
    if st.session_state.get("uploaded_file"):
        file = st.session_state.uploaded_file
        extracted = ekstrak_teks(file)
        token = hitung_token(extracted)
        if token > 3000:
            st.warning("❗ File anda melebihi batas token")
            return

    manual_input = st.session_state.get("user_input", "").strip()

    if extracted and manual_input:
        user_input = extracted + "\n" + manual_input
    elif extracted:
        user_input = extracted
    elif manual_input:
        user_input = manual_input
    else:
        st.warning("Silakan ketik pesan atau unggah file.")
        return

    pasangan_cag = {}
    # user_input = st.session_state.user_input
    current_room = st.session_state.get("room", "default")
    
    # Ambil history dari database
    history = get_chat_history(st.session_state.user.id, current_room)
    # max_history = 10
    # recent_history = history[-max_history:]
    # history_for_prompt = [{"message": msg["message"], "response": msg["response"]} for msg in recent_history]
    history_for_prompt = st.session_state.chat_history[-10:]
        
    # Proses AI response (sama seperti sebelumnya)
    option = st.session_state.get("fitur_selector", "Chatbot")
    fitur = "chatbot" if option == "Chatbot" else "terjemahindosunda" if option == "Terjemah Indo → Sunda" else "terjemahsundaindo"
    mode_bahasa = st.session_state.get("mode_bahasa", "Sunda") if fitur == "chatbot" else None
    
    if fitur == "chatbot" and mode_bahasa == "Sunda" and chat_mode == "Belajar":
        bot_response = generate_text_qwen(user_input, fitur, pasangan_cag, mode_bahasa, chat_mode, history=history_for_prompt)
        text_constraint = bot_response
        pasangan_kata = {}
        pasangan_ekuivalen = {}
        pasangan_kata = {}
    elif fitur == "chatbot" and mode_bahasa == "Sunda":
        bot_response, klasifikasi_bahasa_umum = generate_text_qwen(user_input, fitur, pasangan_cag, mode_bahasa, chat_mode, history=history_for_prompt)
        pasangan_ganti_ekuivalen = {}
        # bot_response_ekuivalen, pasangan_ganti_ekuivalen = ubah_ke_lema(bot_response, df_kamus, df_idiom)
        # bot_koreksi = koreksi_typo_dari_respon(bot_response, df_kamus)
        text_constraint, kata_terdapat, kata_tidak_terdapat, pasangan_kata, pasangan_ekuivalen = highlight_text(bot_response, df_kamus, df_idiom, fitur, klasifikasi_bahasa_umum)
        text_constraint = kapitalisasi_awal_kalimat(text_constraint)
    elif fitur == "chatbot" and (mode_bahasa == "Indonesia" or mode_bahasa == "English"):
        bot_response = generate_text_qwen(user_input, fitur, pasangan_cag, mode_bahasa, chat_mode, history=history_for_prompt)
        text_constraint = bot_response
        pasangan_kata = {}
        pasangan_ekuivalen = {}
        pasangan_kata = {}
    elif option == "Terjemah Sunda → Indo":
        fitur = "terjemahsundaindo"
        bot_response = generate_text_qwen(user_input, fitur, pasangan_cag, mode_bahasa, chat_mode, history=None)
        # text_constraint, kata_terdapat, kata_tidak_terdapat, pasangan_kata, pasangan_ekuivalen = ubah_ke_lema(bot_response, df_kamus, df_idiom)
        text_constraint = bot_response
        pasangan_kata = {}
        pasangan_ekuivalen = {}
        pasangan_kata = {}
    elif option == "Terjemah Indo → Sunda":
        fitur = "terjemahindosunda"
        bot_response = generate_text_qwen(user_input, fitur, pasangan_cag, mode_bahasa, chat_mode, history=None)
        # bot_response_ekuivalen, pasangan_ganti_ekuivalen = ubah_ke_lema(bot_response, df_kamus, df_idiom)
        # bot_koreksi = koreksi_typo_dari_respon(bot_response, df_kamus)
        text_constraint = ganti_sinonim_berdasarkan_tingkat(bot_response, df_kamus)
        text_constraint, kata_terdapat, kata_tidak_terdapat, pasangan_kata, pasangan_ekuivalen = highlight_text(text_constraint, df_kamus, df_idiom, fitur)
        text_constraint = kapitalisasi_awal_kalimat(text_constraint)
        text_constraint = koreksi_typo_dari_respon(text_constraint, df_kamus, df_pemendekan)
        # text_constraint = kapitalisasi_awal_kalimat(text_constraint)
        pasangan_kata = {}
        pasangan_ekuivalen = {}
        pasangan_kata = {}

    html_block = [
        # "<p style='color: yellow;'>Kata Kata yang diganti dari Indo ke Sunda (Kamus) Setelah AI:</p>",
        # f"<p style='color: yellow;'>{pasangan_ganti_ekuivalen}</p>",
        "<p style='color: yellow;'>Kata Kata yang ada di kamus tapi tidak ada Sinonim LOMA:</p>",
        f"<p style='color: yellow;'>{pasangan_ekuivalen}</p>",
        "<p style='color: yellow;'>Kata Kata yang diganti ke Loma:</p>",
        f"<p style='color: yellow;'>{pasangan_kata}</p>",
        "<p style='color: yellow;'>CAG:</p>",
        f"<p style='color: yellow;'>{pasangan_cag}</p>",
    ]
    
    st.session_state.chat_history.append((user_input, bot_response, text_constraint))
    
    # Simpan ke database
    
    result = save_chat_message(
        user_id=st.session_state.user.id,
        message=user_input,
        response=text_constraint,
        response_raw=bot_response,
        room=st.session_state.get("room", "room-1")
    )
    
    if result.get("error") == "limit_exceeded":
        st.warning("Chat history Anda penuh, silakan hapus terlebih dahulu.")
        st.stop()
    
    clear_input()
    st.session_state.uploaded_file = None
    st.session_state.show_file_uploader = False
    
# Modifikasi bagian tampilan chat history
if "user" in st.session_state:
    current_room = st.session_state.get("room", "room-1")
    chat_history = get_chat_history(
        user_id=st.session_state.user.id,
        room=st.session_state.get("room", "room-1")
    )

    if (
        fitur == "chatbot"
        and chat_mode == "Belajar"
        and mode_bahasa == "Sunda"
        and not st.session_state.sudah_disapa
    ):
        st.markdown(
            f"<div class='chat-container'><div class='chat-bubble-bot'>Wilujeng enjing! Kumaha damang?<br><br>(<i>Selamat pagi! Apa kabar?</i>)</div></div>",
            unsafe_allow_html=True
        )
        st.session_state.sudah_disapa = True
        
    # for chat ASLI
    # for chat in chat_history:
    #     st.markdown(
    #         f"<div class='chat-container'><div class='chat-bubble-user'>{chat['message']}</div></div>",
    #         unsafe_allow_html=True
    #     )
    #     st.markdown(
    #         f"<div class='chat-container'><div class='chat-bubble-bot'>{chat['response']}</div></div>",
    #         unsafe_allow_html=True
    #     )

    # For chat percobaan 1
    for chat in chat_history:
        user_msg = chat['message']
        bot_raw = chat.get('response_raw', chat['response'])  # fallback jika belum ada
        bot_constraint = chat['response']
    
        # Bubble user
        st.markdown(f"<div class='chat-container'><div class='chat-bubble-user'>{user_msg}</div></div>", unsafe_allow_html=True)
    
        # Bubble bot 1 - hasil AI murni
        st.markdown(f"<div class='chat-container'><div class='chat-bubble-bot'>{bot_raw}</div></div>", unsafe_allow_html=True)
    
        # Bubble bot 2 - hasil setelah constraint
        st.markdown(f"<div class='chat-container'><div class='chat-bubble-bot'>{bot_constraint}</div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # ⬅️ END OF chat-container-outer


col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_area(
        label="", height=80, key="user_input", placeholder="Tulis pesan...",
        label_visibility="collapsed"
    )
   
with col2:
    st.button("➡", on_click=handle_send)

col_left, col_right = st.columns([1, 2])
with col_left:
    if st.button("Attach File", help="Lampirkan file"):
        st.session_state.show_file_uploader = not st.session_state.show_file_uploader

    if st.session_state.show_file_uploader:
        uploaded = st.file_uploader(
            label="",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded:
            st.session_state.uploaded_file = uploaded
            st.success(f"📎 File '{uploaded.name}' terunggah")
with col_right:
    if st.button("🔄 Delete Chat History"):
        supabase.table("chat_history") \
            .delete() \
            .eq("user_id", st.session_state.user.id) \
            .eq("room", st.session_state.get("room", "room-1")) \
            .execute()
        st.rerun()

# Tambah anchor di akhir chat
st.markdown('<a name="scroll-bottom"></a>', unsafe_allow_html=True)
st.markdown("""
    <style>
    .scroll-down-btn {
        position: fixed;
        bottom: 80px;
        right: 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 16px;
        font-size: 16px;
        cursor: pointer;
        z-index: 1000;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.3);
    }
    </style>
    <a href="#scroll-bottom"><button class="scroll-down-btn">⬇️</button></a>
""", unsafe_allow_html=True)

components.html("""
    <script>
        setInterval(function() {
            fetch(window.location.href);
        }, 1000 * 30);  // ping setiap 5 detik
    </script>
""", height=0)
st.markdown("</div>", unsafe_allow_html=True)
