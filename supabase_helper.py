from supabase import create_client, Client
import streamlit as st
import time
# from supabase.lib.auth_client import SupabaseAuthException
from httpx import RequestError

# SUPABASE_URL = st.secrets["SUPABASE_URL"]
# SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

def load_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
supabase = load_supabase()
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def sign_in_with_email(email, password):
    return supabase.auth.sign_in_with_password({"email": email, "password": password})

def sign_up_with_email(email, password, age):
    result = supabase.auth.sign_up({"email": email, "password": password})
    user_id = result.user.id
    supabase.table("profiles").insert({"id": user_id, "age": age}).execute()
    return result

def sign_out():
    supabase.auth.sign_out()

def get_user_session():
    return supabase.auth.get_session()

def get_user_chat_rooms(user_id):
    result = supabase.table("chat_history") \
        .select("room") \
        .eq("user_id", user_id) \
        .execute()
    if result.data:
        rooms = list(set([r["room"] for r in result.data]))
        return sorted(rooms)
    return []

def delete_chat_room(user_id, room_name):
    supabase.table("chat_history") \
        .delete() \
        .eq("user_id", user_id) \
        .eq("room", room_name) \
        .execute()

def insert_chat_history(user_id, room, message, response):
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "room": room,
        "message": message,
        "response": response
    }).execute()

def fetch_chat_history(user_id, room):
    result = supabase.table("chat_history") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("room", room) \
        .order("created_at", desc=False) \
        .execute()
    return result.data if result else []

# Fungsi untuk menyimpan chat history
def save_chat_message(user_id, message, response, room="default", response_raw=None):
    # Cek jumlah history chat dulu
    result = supabase.table("chat_history") \
        .select("id", count="exact") \
        .eq("user_id", user_id) \
        .eq("room", room) \
        .execute()

    if result.count >= 100:
        return {"error": "limit_exceeded"}

    # Jika belum penuh, simpan
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "message": message,
        "response": response,
        "response_raw": response_raw,
        "room": room
    }).execute()

    return {"status": "success"}

# Fungsi untuk mengambil chat history
def get_chat_history(user_id, room="default", limit=100):
    result = supabase.table("chat_history") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("room", room) \
        .order("created_at", desc=False) \
        .limit(limit) \
        .execute()
    return result.data if result else []

def get_first_chat_preview(user_id, room):
    # result = supabase.table("chat_history") \
    #     .select("message, response") \
    #     .eq("user_id", user_id) \
    #     .eq("room", room) \
    #     .order("created_at", desc=False) \
    #     .limit(1) \
    #     .execute()
    retry = 3
    last_error = None

    for attempt in range(retry):
        try:
            result = supabase.table("chat_history") \
                .select("message, response") \
                .eq("user_id", user_id) \
                .eq("room", room) \
                .order("created_at", desc=False) \
                .limit(1) \
                .execute()
            if result.data:
                msg = result.data[0]["message"] or ""
                res = result.data[0]["response"] or ""
                text = msg.strip() or res.strip()
                # Ambil 5 kata pertama dari message/response
                preview = " ".join(text.split()[:5]) + "..." if text else "Chat kosong..."
                return preview
            return "Chat kosong..."
        except RequestError as e:
            last_error = e
            time.sleep(2)  # jeda sebelum ulangi
        except Exception as e:
            last_error = e
            time.sleep(2)
    st.error(f"Gagal memuat chat pertama dari Supabase: {last_error}")
    return ""

    # if result.data:
    #     msg = result.data[0]["message"] or ""
    #     res = result.data[0]["response"] or ""
    #     text = msg.strip() or res.strip()
    #     # Ambil 5 kata pertama dari message/response
    #     preview = " ".join(text.split()[:5]) + "..." if text else "Chat kosong..."
    #     return preview
    # return "Chat kosong..."

