from openai import OpenAI
import os
import streamlit as st
import re
import string
from supabase_helper import *
import requests
import json
import fitz  # PyMuPDF
import docx
import time
import tiktoken

# relasi_tutur = {
#     "bapak": {"anak": "Loma", "istri": "Loma", "teman": "Loma", "atasan": "Halus", "ibu": "Loma"}, # Tambahan: 'ibu'
#     "ibu": {"anak": "Loma", "suami": "Loma", "teman": "Loma", "atasan": "Halus", "bapak": "Loma"}, # Tambahan: 'bapak'
#     "anak": {"bapak": "Halus", "ibu": "Halus", "teman": "Loma", "guru": "Halus", "orang dewasa": "Halus"}, # 'Loma/kasar' menjadi 'Loma', Tambahan: 'orang dewasa'
#     "atasan": {"bawahan": "Loma"},
# }

# def deteksi_relasi_kutipan(teks):
#     hasil = []
#     pola_list = [
#     r"(\w+)\s+(?:berkata|menjawab|ujar|pun berkata|menyuruh|meminta|menasihati|berseru|berbicara)\s+(?:kepada|dengan)\s+(\w+):\s+\"(.*?)\"",
#     r"kata\s+(\w+)\s+kepada\s+(\w+),?\s*\"(.*?)\"",
#     r"ujar\s+(\w+)\s+kepada\s+(\w+),?\s*\"(.*?)\"",
#     # Tambahkan pola lain jika ada variasi umum yang Anda ingin deteksi
#     ]

#     for pola in pola_list:
#         cocok = re.findall(pola, teks, flags=re.IGNORECASE)
#         for pembicara, pendengar, kutipan in cocok:
#             # Normalisasi ke lowercase
#             pembicara_lower = pembicara_raw.lower()
#             pendengar_lower = pendengar_raw.lower()
            
#             # Coba cari padanan di relasi_tutur berdasarkan kata kunci
#             pembicara_key = None
#             for key in relasi_tutur.keys():
#                 if key in pembicara_lower: # Cek jika kata kunci ada dalam nama pembicara yang terdeteksi (misal 'pak budi' -> 'bapak')
#                     pembicara_key = key
#                     break
            
#             pendengar_key = None
#             if pembicara_key: # Hanya cari pendengar jika pembicara dikenali
#                 for key in relasi_tutur[pembicara_key].keys():
#                     if key in pendengar_lower: # Cek jika kata kunci ada dalam nama pendengar yang terdeteksi
#                         pendengar_key = key
#                         break
            
#             tingkat = "L1" # Default jika tidak ditemukan relasi yang pas
#             if pembicara_key and pendengar_key:
#                 tingkat = relasi_tutur.get(pembicara_key, {}).get(pendengar_key, "L1")
            
#             if tingkat == "Loma/kasar":
#                 tingkat = "Loma" # Menetapkan default jika deteksi adalah 'Loma/kasar'
            
#             hasil.append({
#                 "pembicara": pembicara_raw, # Menyimpan nama asli
#                 "pendengar": pendengar_raw, # Menyimpan nama asli
#                 "kutipan": kutipan.strip(),
#                 "tingkat": tingkat
#             })

#     # Fallback ke pemanggilan API DeepSeek jika tidak ada deteksi regex yang cocok
#     if not hasil:
#         instruksi = (
#             "Identifikasi apakah ada kalimat langsung (kutipan) dalam teks berikut.\n"
#             "Jika ada, tentukan siapa pembicara dan siapa pendengarnya, serta kalimat langsung (kutipan)nya.\n"
#             "Kemudian tentukan tingkat tutur sesuai relasi sosial dari daftar ini:\n"
#             "- bapak → anak = Loma\n"
#             "- bapak → isteri = Loma\n"
#             "- bapak → teman = Loma\n"
#             "- bapak → atasan = Halus\n"
#             "- anak → bapak/ibu = Halus\n"
#             "- anak → orang dewasa/Guru = Halus\n" # Perbarui di sini juga
#             "- anak → teman = Loma/kasar\n"
#             "- ibu → anak = Loma\n"
#             "- ibu → suami = Loma\n"
#             "- ibu → teman = Loma\n"
#             "- ibu → atasan = Halus\n"
#             "- atasan → bawahan = Loma\n"
#             "Jika relasi tidak tercantum, tentukan tingkat tutur berdasarkan konteks dan norma kesopanan umum dalam bahasa Sunda (Loma/Halus).\n"
#             "Jawaban dalam format:\n"
#             "Pembicara: ...\nPendengar: ...\nKutipan: \"...\"\nTingkat: ..."
#         )
#         response_llm, _, _ = call_deepseek_api(teks, system_instruction=instruksi) # Ganti nama variabel agar tidak ambigu
#         match = re.search(r"Pembicara: (.*?)\nPendengar: (.*?)\nKutipan: \"(.*?)\"\nTingkat: (.*?)\b", response_llm, re.DOTALL)
#         if match:
#             pembicara, pendengar, kutipan, tingkat = match.groups()
#             hasil.append({
#                 "pembicara": pembicara.strip(),
#                 "pendengar": pendengar.strip(),
#                 "kutipan": kutipan.strip(),
#                 "tingkat": tingkat.strip().upper()
#             })

#     return hasil

# def sisipkan_kutipan_ke_system_instruction(teks, instruksi_awal):
#     kutipan = deteksi_relasi_kutipan(teks)
#     if not kutipan:
#         return instruksi_awal

#     bagian = "\n\nTerdeteksi kutipan langsung berikut:\n"
#     for k in kutipan:
#         bagian += f"- Pembicara: {k['pembicara']}, Pendengar: {k['pendengar']}, Tingkat: {k['tingkat']}, Kutipan: \"{k['kutipan']}\"\n"

#     return instruksi_awal + bagian

@st.cache_resource
def call_groq_api(prompt, history=None, system_instruction=None):
    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    else:
        messages.append({"role": "system", "content": "You are a helpful assistant."})

    if history:
        for msg in history:
            messages.append({"role": "user", "content": msg["message"]})
            messages.append({"role": "assistant", "content": msg["response"]})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "qwen/qwen3-32b",
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9,
        "frequency_penalty": 1.5,
        "presence_penalty": 0.7,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error Groq API: {e}")
        return "Maaf, terjadi kesalahan saat memproses permintaan Anda."

def generate_text_qwen(user_input, fitur, pasangan_cag, mode_bahasa="Sunda", chat_mode = "Ngobrol", history=None):
    user_age = 30  # Default
    if "user" in st.session_state:
        try:
            profile = supabase.table("profiles").select("age").eq("id", st.session_state.user.id).execute()
            if profile.data:
                user_age = profile.data[0]["age"]
        except:
            pass

    # Inisialisasi klasifikasi_bahasa default
    klasifikasi_bahasa_umum = "Loma" if user_age < 30 else "Lemes"
    
    system_instruction = ""
    user_prompt = user_input # Pastikan user_prompt merujuk ke input asli pengguna
    
    # Deteksi kutipan dan tentukan tingkat tutur utama dari kutipan
    # deteksi_hasil_kutipan = deteksi_relasi_kutipan(user_prompt)
    # tingkat_tutur_kutipan_utama = None
    
    # if deteksi_hasil_kutipan:
    #     tingkat_tutur_kutipan_utama = deteksi_hasil_kutipan[0]['tingkat'] # Ambil dari kutipan pertama
    
    # # Prioritaskan tingkat tutur dari kutipan jika terdeteksi
    # final_klasifikasi_bahasa = tingkat_tutur_kutipan_utama if tingkat_tutur_kutipan_utama else klasifikasi_bahasa_umum
 
    # Instruksi berdasarkan fitur dan mode bahasa
    if fitur == "chatbot":
        if mode_bahasa == "Sunda":
            if chat_mode == "Ngobrol":
                system_instruction = f"""Sok ngajawab ku basa Sunda sanajan ditanya ku basa sejen. Ngagunakeun kecap {klasifikasi_bahasa_umum}, nalika ngajawab. Mitra obrolan anjeun yuswa {user_age} taun. Mangga saluyukeun gaya nyarita anjeun ka umur pasangan obrolan anjeun.
                                Ulah sok make kecap-kecap anu lain basa Sunda!. Gunakeun tata basa Sunda anu alus tur bener. Jawaban anjeun kedah rapih, henteu pabalatak.
                                Unggal waktos Anjeun ngucapkeun kecap ("Nak"), robah jadi ("Jang"). Teu kungsi typo. Jika anda ditanya "Kumaha damang?" atau "kumaha damang" tolong jawab selalu dengan "Saé, anjeun kumaha?" tapi selain ditanya itu jangan jawab "Saé, anjeun kumaha?".
                                       """
                system_instruction += f"""Hanya jika terdapat kalimat langsung (kutipan), gunakan tingkat tutur berikut:
                                         - Jika orang tua (bapak/ibu) berbicara kepada anak, istri, atau teman → gunakan basa LOMA.
                                         - Jika anak berbicara kepada orang tua, guru, atau orang dewasa → gunakan basa HALUS.
                                         - Jika atasan berbicara kepada bawahan → gunakan basa LOMA.
                                         - Jika tidak jelas relasinya, gunakan basa netral sesuai konteks.
                                         Jika tidak ada kalimat langsung, abaikan aturan ini."""
            elif chat_mode == "Belajar":
                system_instruction = f"""Anda adalah asisten untuk pelajar.
                                         Koreksi kalimat pengguna hanya jika ada kesalahan kata atau kalimat.
                                         Selalu menjawab dengan bahasa sunda, tiap kalimat disertai terjemahan bahasa Indonesia dibawahnya dengan format daftar kata dan artinya.
                                         Contoh: Wilujeng enjing! Naha anjeun badé diajar dinten ieu?
                                                 - Wilujeng = Selamat 
                                                 - enjing = pagi
                                                 - Naha = Apakah 
                                                 - anjeun = kamu 
                                                 - badé = akan 
                                                 - diajar = belajar 
                                                 - dinten = hari 
                                                 - ieu = ini
                                         Selalu bertanya diakhir yang berkaitan dengan pembelajaran.
                                      """
        elif mode_bahasa == "Indonesia":
            if chat_mode == "Ngobrol":
                system_instruction = "Jawablah hanya dalam Bahasa Indonesia."
            elif chat_mode == "Belajar":
                system_instruction = f"""Anda adalah asisten untuk pelajar.
                                         Sesuaikan Bahasa dengan lawan bicara.
                                         Koreksi kalimat pengguna hanya jika ada kesalahan kata atau kalimat.
                                         Selalu menjawab dengan bahasa pengantar dari lawan bicara, tiap kalimat disertai terjemahan bahasa Indonesia dibawahnya menggunakan format daftar kata dan artinya.
                                         Contoh jika bahasa sunda: Wilujeng enjing! Naha anjeun badé diajar dinten ieu? 
                                                                   - Wilujeng = Selamat 
                                                                   - enjing = pagi
                                                                   - Naha = Apakah 
                                                                   - anjeun = kamu 
                                                                   - badé = akan 
                                                                   - diajar = belajar 
                                                                   - dinten = hari 
                                                                   - ieu = ini
                                         Selalu bertanya diakhir yang berkaitan dengan pembelajaran.
                                      """
        elif mode_bahasa == "English":
            if chat_mode == "Ngobrol":
                system_instruction = "Please respond only in British English."
            elif chat_mode == "Belajar":
                system_instruction = f"""You are an assistant for students.
                                         Adjust the language to the person you are talking to.
                                         Correct the user's sentence only if there is a mistake in the word or sentence.
                                         Always answer in the language of the person you are talking to, accompanied by an English translation below it using brackets.
                                         Example if Sundanese: Wilujeng enjing! Naha anjeun badé diajar dinten ieu? 
                                                               - Wilujeng enjing = Good morning
                                                               - Naha anjeun = Are you 
                                                               - badé diajar = going to study 
                                                               - dinten ieu = today
                                         Always ask questions at the end that are related to learning.
                                      """
        else:
            system_instruction = ""
                    
        system_instruction += f"""
        Anda adalah Lestari, chatbot interaktif yang ahli dalam bahasa Sunda, Indonesia, dan English serta menjawab pertanyaan secara ramah dan jelas informasinya.
        Anda berumur 30 tahun. Lawan bicara anda berumur {user_age} tahun. tolong sesuaikan gaya bicara anda dengan umur lawan bicara anda. 
        Jangan memberi keterangan catatan dibawahnya. Jangan memberikan informasi yang tidak tentu kebenarannya.
        Selalu akhiri dengan pertanyaan."""
        # Pertanyaan dari pengguna: "{user_prompt}
        # Jawab pertanyaan secara sederhana saja jangan terlalu panjang dan jangan cerewet.
        # Gunakan huruf kapital pada awal kalimat dan setelah tanda titik serta setelah petik dua atau setelah paragraf.
        # Gunakan huruf kapital pada awal nama orang dan nama tempat.
        # Gunakan huruf kapital yang sama jika pada kalimat atau kata pada input user menggunakan huruf kapital.

    elif fitur == "terjemahindosunda":
        system_instruction = f"""Anjeun téh panarjamah anu ahli dina basa Sunda.
                            Tarjamahkeun kalimah di handap ieu kana Basa Sunda {klasifikasi_bahasa_umum} luyu jeung tata basa Sunda anu alus tur bener.
                            Teu kungsi typo atawa salah kata.
                            Wanohkeun format paragraf tina téks kalimah ti nu maké.
                            Jaga sangkan format paragraf jeung barisna tetep sarua persis siga téks asli atawa input ti nu maké.
                            Ulah ngagabungkeun paragraf.
                            Ulah ngarang kecap anu lain basa Sunda.
                            Anggo hurup kapital anu sarua upami dina kalimah atawa kecap dina input ti nu maké nganggo hurup kapital.
                            Ulah ngajak ngobrol siga fitur chatbot. anjeun ngan ukur narjamahkeun input ti nu maké siga Google Translator.
                            Ulah nambahan kecap basa Sunda anu memang lain harti tina kalimah basa Indonésia éta.
                            Saluyukeun gaya basana sangkan cocog jeung kontéks rélasi antarpanutur lamun manggihan kalimah langsung sanggeus titik (:) atawa dina tanda petik ("").
                            bapak ka anak = Loma
                            bapak ka isteri = Loma
                            bapak ka teman = Loma
                            bapak ka atasan = Halus
                            anak ka bapak/ibu = Halus
                            anak ka orang dewasa/Guru = Halus
                            anak ka teman = Loma/kasar
                            ibu ka anak = Loma
                            ibu ka suami = Loma
                            ibu ka teman = Loma
                            ibu ka atasan = Halus
                            atasan ka bawahan = Loma
                            Paréntah anjeun ngan ukur narjamahkeun tina input nu maké, lain ngajawab hal séjén. Ulah ngagunakeun kecap bubuka atawa sapaan minangka tambahan jawaban.
                            Ulah méré pedaran atawa katerangan tambahan, langsung bikeun waé hasil tarjamahanna.
                            Ulah dijadikeun sakabéh hurup dina mimiti kecap hurup kapital iwal ngaran tempat jeung ngaran jalma.
                            Hurup dina mimiti kalimah jeung sanggeus titik sarta sanggeus tanda petik dua atawa sanggeus paragraf kudu hurup kapital.
                            Ngaran jalma jeung ngaran tempat ogé kudu diawalan hurup kapital.
                            Anjeun kudu konsisten dina narjamahkeun kalawan bener tur kudu rapih.
                            """
        # system_instruction = sisipkan_kutipan_ke_system_instruction(user_prompt, system_instruction)

    elif fitur == "terjemahsundaindo":
        system_instruction = f"""Kamu adalah penerjemah yang ahli bahasa indonesia dan bahasa sunda.
        Terjemahkan kalimat berikut ke dalam Bahasa Indonesia yang baku dan mudah dimengerti.
        Jaga agar format paragraf dan barisnya tetap sama persis seperti teks asli atau input user.
        Jangan menggabungkan paragraf.
        Gunakan huruf kapital yang sama jika pada kalimat atau kata pada input user menggunakan huruf kapital.
        Jangan mengajak mengobrol seperti fitur chatbot, anda hanya menterjemahkan input dari user seperti google translate.
        Jangan tambahkan penjelasan atau keterangan apa pun. Langsung tampilkan hasil terjemahannya.
        Jangan jadikan semua huruf pada awal kata huruf kapital, kecuali nama orang dan nama tempat.
        Huruf pada awal kalimat dan setelah titik serta setelah petik dua atau setelah paragraf harus huruf kapital.
        Nama orang dan nama tempat juga harus berawalan huruf kapital.
        """
        # Kalimat: {user_prompt}"""

    else:
        # fallback
        system_instruction = f"Jawablah dengan sopan dan informatif: {user_prompt}"

    # formatted_history = [{"message": m[0], "response": m[1]} for m in history] if history else None
    formatted_history = [{"message": m["message"], "response": m["response"]} for m in history] if history else None
    response = call_groq_api(prompt=user_prompt, history=formatted_history, system_instruction=system_instruction)
    return response

def bersihkan_superscript(teks):
    # Menghapus superscript angka ¹²³⁴⁵⁶⁷⁸⁹⁰ atau angka biasa setelah huruf
    return re.sub(r'([^\d\s])[\u00B9\u00B2\u00B3\u2070\u2074-\u2079\d]+', r'\1', teks)

def kapitalisasi_awal_kalimat(teks):
    # Bersihkan superscript dulu
    teks = bersihkan_superscript(teks)

    # Pecah berdasarkan paragraf (baris kosong)
    paragraf_list = teks.split("\n\n")
    paragraf_hasil = []

    for paragraf in paragraf_list:
        # Bagi kalimat dalam paragraf berdasarkan tanda baca yang diikuti spasi atau akhir kalimat
        kalimat_list = re.split(r'([.!?]["\']?\s+)', paragraf)
        kalimat_terformat = ""

        for i in range(0, len(kalimat_list), 2):
            if i < len(kalimat_list):
                kalimat = kalimat_list[i].strip()
                if kalimat:
                    kapital = kalimat[0].upper() + kalimat[1:] if len(kalimat) > 1 else kalimat.upper()
                    kalimat_terformat += kapital
            if i + 1 < len(kalimat_list):
                kalimat_terformat += kalimat_list[i+1]

        paragraf_hasil.append(kalimat_terformat.strip())

    # Gabungkan kembali paragraf dengan \n\n
    return "\n\n".join(paragraf_hasil)

def hitung_token(teks):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(teks))

# reader = easyocr.Reader(['id', 'en'])  # Tambahkan 'su' untuk Bahasa Sunda

def ekstrak_teks(file):
    if file.type == "application/pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        import docx
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file.type.startswith("image/"):
        # from PIL import Image
        # import numpy as np
        # img = Image.open(file).convert("RGB")
        # img_array = np.array(img)
        # results = reader.readtext(img_array, detail=0)
        # return "\n".join(results)
        return "[Gambar terlampir, tidak diproses sebagai teks]"

    else:
        return "❌ Jenis file tidak didukung."

def pilih_berdasarkan_konteks_llm(kandidat_list, kalimat_asli, kata_typo_asli):
    if not kandidat_list:
        return None

    daftar_kandidat = ", ".join(kandidat_list)
    prompt = f"""
            Kalimat berikut mengandung kata salah tulis:
            
            "{kalimat_asli}"
            
            Bagian yang salah adalah: "{kata_typo_asli}" (ditulis dalam tag <i>...</i>).
            
            Berikut adalah daftar kandidat koreksi (dari kamus): {daftar_kandidat}.
            
            Pilih satu kata yang paling sesuai secara makna dengan konteks kalimat tersebut.
            Jawab hanya dengan satu kata dari daftar, tanpa penjelasan dan jangan memberikan catatan.
            """
    hasil = call_groq_api(prompt, history=None,  system_instruction=None)  # Atau Groq, OpenAI
    hasil_bersih = hasil.strip().lower()

    if hasil_bersih in kandidat_list:
        return hasil_bersih
    return None
    

