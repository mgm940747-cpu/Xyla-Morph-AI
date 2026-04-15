import os

import streamlit as st
import torch
import whisper

# --- Page Config ---
st.set_page_config(page_title="XYLA MORPH AI", page_icon="💎", layout="wide")

# --- Optimized Squared UI & Friendly Upload Box ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;800&family=Inter:wght@500;700&display=swap');

    .stApp {
        background-color: #0B0C10;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }

    /* Strict Squared for ALL Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stFileUploader), 
    .stTextArea textarea, 
    div[data-testid="stSelectbox"] > div,
    .metric-box, .stAudio {
        border-radius: 0px !important;
        background-color: #1F2833 !important;
        border: 2px solid #45A29E !important;
    }

    /* Better Upload Box UI */
    section[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #66FCF1 !important;
        border-radius: 0px !important;
        background-color: #0B0C10 !important;
        padding: 40px !important;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        border-style: solid !important;
        background-color: #1F2833 !important;
    }

    /* Header Branding */
    .header-container {
        border-left: 10px solid #66FCF1;
        padding: 15px 20px;
        margin-bottom: 40px;
        background: #1F2833;
    }
    .header-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 30px;
        font-weight: 800;
        color: #FFFFFF !important;
        text-transform: uppercase;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #66FCF1 !important;
        color: #000000 !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        border: none !important;
        border-radius: 0px !important;
        height: 55px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button:hover {
        background-color: #FFFFFF !important;
        box-shadow: 0 0 20px #66FCF1;
    }

    /* Text & Label Colors */
    label[data-testid="stWidgetLabel"], .stMarkdown p, h5 {
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        font-size: 13px;
        letter-spacing: 1px;
    }

    .stTextArea textarea {
        color: #FFFFFF !important;
        font-size: 15px !important;
        background-color: #0B0C10 !important;
    }

    /* Metric Box Details */
    .metric-box {
        margin-top: -10px;
        background-color: #1F2833 !important;
        border-top: none !important;
    }
    .m-title { color: #66FCF1; font-size: 11px; }
    .m-val { color: #FFFFFF; font-size: 18px; font-weight: bold; }

    /* Fix for Selectbox Text */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        color: #FFFFFF !important;
        background-color: #1F2833 !important;
        border-radius: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def get_stats(text):
    words = text.split()
    count = len(words)
    secs = (count / 150) * 60
    return count, f"{int(secs // 60)}M {int(secs % 60)}S"


st.markdown("<div class='header-container'><div class='header-text'>XYLA_MORPH_V3 / CORE</div></div>",
            unsafe_allow_html=True)

l_col, r_col = st.columns([1, 1], gap="large")

with l_col:
    st.markdown("##### [ 01 / ENGINE_CONFIG ]")
    # Intelligence Selector - Now Strictly Squared
    m_size = st.selectbox("CHOOSE_INTELLIGENCE_LEVEL", ["tiny", "base", "small", "medium", "large"], index=1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### [ 02 / UPLOAD_DATA ]")
    # Friendly Upload Box
    audio = st.file_uploader("DRAG AND DROP AUDIO FILE (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

    if audio:
        st.info(f"FILE LOADED: {audio.name.upper()}")
        st.audio(audio)
        st.session_state['fn'] = os.path.splitext(audio.name)[0]

        if st.button("START TRANSCRIPTION PROCESS"):
            with st.status("💎 ANALYZING AUDIO STREAM...", expanded=False):
                with open("temp.mp3", "wb") as f:
                    f.write(audio.getbuffer())
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                model = whisper.load_model(m_size, device=dev)
                res = model.transcribe("temp.mp3")
                st.session_state['txt'] = res["text"]
                os.remove("temp.mp3")

with r_col:
    st.markdown("##### [ 03 / SCRIPT_OUTPUT ]")
    if 'txt' in st.session_state:
        output_txt = st.text_area("EDITABLE_TRANSCRIPT", value=st.session_state['txt'], height=380)

        w_count, est_time = get_stats(output_txt)
        st.markdown(f"""
            <div class='metric-box'>
                <table style='width:100%'>
                    <tr>
                        <td><div class='m-title'>WORD_COUNT</div><div class='m-val'>{w_count}</div></td>
                        <td style='text-align:right'><div class='m-title'>EST_AUDIO_LENGTH</div><div class='m-val'>{est_time}</div></td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label=f"EXPORT AS {st.session_state['fn'].upper()}.TXT",
            data=output_txt,
            file_name=f"{st.session_state['fn']}.txt",
            use_container_width=True
        )
    else:
        st.markdown("""
            <div style='height:480px; border:2px dashed #1F2833; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#45A29E;'>
                <div style='font-size:24px;'>📡</div>
                <div style='font-weight:bold; margin-top:10px;'>SYSTEM_IDLE: AWAITING_STREAM</div>
            </div>
        """, unsafe_allow_html=True)

# Footer
dev_info = "GPU_ACCELERATED" if torch.cuda.is_available() else "CPU_STANDARD"
st.markdown(
    f"<div style='margin-top:40px; font-size:10px; color:#45A29E; font-family:monospace;'>CORE_STATUS: {dev_info} / READY</div>",
    unsafe_allow_html=True)
