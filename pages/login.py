import streamlit as st
# Main
st.set_page_config(page_title="Lestari Bahasa", layout="wide")
from supabase_helper import *
import uuid, time

# # Main
# st.set_page_config(page_title="Lestari Bahasa", layout="wide")

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
    
    /* Hilangkan elemen GitHub (termasuk avatar foto) */
    a[href*="github.com"], img[src*="githubusercontent"] {
        display: none !important;
    }

    /* Ubah warna background tombol collapse sidebar ( > ) */
    button[data-testid="collapsedControl"] {
        background-color: #b0b0b0 !important;  /* abu-abu */
        color: white !important;
        border: none;
        border-radius: 0 8px 8px 0;
        padding: 8px 12px;
        margin-top: 8px;
        margin-left: -4px;
        transition: all 0.3s ease;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.2);
    }

    /* Saat hover */
    button[data-testid="collapsedControl"]:hover {
        background-color: #ffc107 !important;
        color: black !important;
    }

    /* Tambahan agar ikon lebih besar */
    button[data-testid="collapsedControl"] svg {
        width: 1.2rem;
        height: 1.2rem;
    }
    .stTextArea, .stButton {
            margin-top: 0px;
        }
        div[data-testid="column"] {
            display: flex;
            align-items: center;
        }
        button[kind="primary"] {
            background-color: #25D366;  /* WhatsApp green */
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Layout tombol login/logout di kanan atas
def render_topbar():
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.session_state.get("user"):
            st.markdown(
            f"""
            <div style='position: fixed; top: 10px; right: 20px; z-index:9999;'>
                👤 {st.session_state["email"]}<br>
                <a href="/login?logout=true" target="_self">
                    <button style="background:#ff4b4b;color:white;border:none;padding:4px 10px;border-radius:5px;">Logout</button>
                </a>
            </div>
            """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='position: fixed; top: 10px; right: 20px; z-index:9999;'>
                    <form action="?login" method="get">
                        <button style="background:#4bafff;color:white;border:none;padding:6px 10px;border-radius:5px;">Login</button>
                    </form>
                </div>
                """, unsafe_allow_html=True
            )

import requests
    
# Auth flow
def auth_flow():
    if "register_mode" not in st.session_state:
        st.session_state.register_mode = False

    if st.session_state.register_mode:
        st.subheader("Register")
        email = st.text_input("Email")
        age = st.number_input("Umur", min_value=5, max_value=100, step=1)
        password = st.text_input("Password", type="password")
        if st.button("Buat Akun"):
            try:
                result = sign_up_with_email(email, password, age)
                st.success("Akun berhasil dibuat. Silakan login.")
                st.session_state.register_mode = False
            except Exception as e:
                st.error(f"Gagal membuat akun: {e}")
        if st.button("Sudah punya akun? Login"):
            st.session_state.register_mode = False
            st.rerun()
        st.markdown(
            "<small style='color:gray'>*Jika belum muncul, klik tombol sekali lagi.</small>",
            unsafe_allow_html=True
        )   
    else:
        st.subheader("Login")
        email = st.text_input("Email").strip()
        password = st.text_input("Password", type="password").strip()
        if st.button("Login"):
            try:
                result = sign_in_with_email(email, password)
                if result.user:
                    st.session_state.user = result.user
                    st.session_state.email = email
                    st.session_state["access_token"] = result.session.access_token
                    st.session_state["refresh_token"] = result.session.refresh_token
                    st.switch_page("main.py")  # Redirect ke main.py setelah login
                else:
                    st.error("Login gagal.")
            except Exception as e:
                st.error(f"Login gagal: {str(e)}")
        if st.button("Belum punya akun? Daftar"):
            st.session_state.register_mode = True
            st.rerun()
        st.markdown(
            "<small style='color:gray'>*Jika belum muncul, klik tombol sekali lagi.</small>",
            unsafe_allow_html=True
        )
    

# if "logout" in st.query_params:
#     print("Memulai proses logout...")  # Lihat di terminal
#     start_time = time.time()
#     sign_out()
#     st.session_state.clear()
#     print(f"Logout selesai dalam {time.time()-start_time:.2f} detik")
#     st.rerun()
    
# if "user" in st.session_state and "logout" not in st.query_params:
#     with st.spinner("Mengarahkan ke halaman utama..."):
#         st.switch_page("main.py")

if st.session_state.get("user"):
    with st.spinner("Mengarahkan ke halaman utama..."):
        st.switch_page("main.py")  # Redirect ke main jika sudah login

else:
    auth_flow() 
