"""
XYLA MORPH AI PRO+

NOTE:
This app requires these packages:
- streamlit
- torch
- openai
- whisper

Install with:
    pip install streamlit torch openai-whisper openai

Run with:
    streamlit run app.py
"""

import os

# --- SAFE IMPORTS (prevents crash if missing deps) ---
try:
    import streamlit as st
except ModuleNotFoundError:
    raise SystemExit("Streamlit is not installed. Run: pip install streamlit")

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch is not installed. Run: pip install torch")

try:
    import whisper
except ModuleNotFoundError:
    raise SystemExit("Whisper is not installed. Run: pip install openai-whisper")

try:
    import openai
except ModuleNotFoundError:
    raise SystemExit("OpenAI SDK missing. Run: pip install openai")

# --- CONFIG ---
st.set_page_config(page_title="XYLA MORPH AI PRO+", page_icon="💎", layout="wide")

# --- API KEY ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- UI STYLE ---
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#050608,#0B0C10); color:#EAEAEA;}
.glass {background: rgba(255,255,255,0.05); backdrop-filter: blur(15px); border-radius:14px; padding:20px;}
.stButton>button {background: linear-gradient(90deg,#66FCF1,#45A29E); border-radius:10px; height:50px;}
.stButton>button:hover {transform: scale(1.05); box-shadow:0 0 20px #66FCF1;}
textarea {background: rgba(0,0,0,0.6)!important; color:white!important;}
</style>
""", unsafe_allow_html=True)

# --- AI FUNCTIONS ---
def gpt_summary(text):
    if not openai.api_key:
        return "Missing OPENAI_API_KEY"
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Summarize this:\n{text}"}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Summary failed: {str(e)}"


def gpt_keywords(text):
    if not openai.api_key:
        return "Missing OPENAI_API_KEY"
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Extract 5 keywords:\n{text}"}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Keyword extraction failed: {str(e)}"


def gpt_translate(text, lang="English"):
    if not openai.api_key:
        return "Missing OPENAI_API_KEY"
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Translate to {lang}:\n{text}"}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Translation failed: {str(e)}"

# --- HEADER ---
st.title("💎 XYLA MORPH V5 / PRO AI")

col1, col2 = st.columns([1,1])

# --- LEFT PANEL ---
with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    model_size = st.selectbox("MODEL", ["tiny","base","small","medium","large"], index=1)
    audio = st.file_uploader("Upload Audio", type=["mp3","wav","m4a"])

    if audio:
        st.audio(audio)
        st.session_state['fn'] = os.path.splitext(audio.name)[0]

        if st.button("TRANSCRIBE"):
            with st.spinner("Transcribing..."):
                with open("temp.mp3","wb") as f:
                    f.write(audio.getbuffer())

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = whisper.load_model(model_size, device=device)
                result = model.transcribe("temp.mp3")

                st.session_state['txt'] = result.get("text", "")
                os.remove("temp.mp3")

    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT PANEL ---
with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    if 'txt' in st.session_state and st.session_state['txt']:
        text = st.text_area("Transcript", value=st.session_state['txt'], height=200)

        if st.button("AI SUMMARY"):
            st.info(gpt_summary(text))

        if st.button("AI KEYWORDS"):
            st.success(gpt_keywords(text))

        lang = st.selectbox("Translate to", ["English","Myanmar","Japanese","Chinese"])
        if st.button("TRANSLATE"):
            st.warning(gpt_translate(text, lang))

        st.download_button("Download TXT", data=text, file_name=f"{st.session_state.get('fn','output')}.txt")

    else:
        st.info("Upload audio to start")

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
device = "GPU" if torch.cuda.is_available() else "CPU"
st.caption(f"SYSTEM: {device} READY | GPT ENABLED")

# --- BASIC TESTS (sanity checks) ---
if __name__ == "__main__":
    assert isinstance(gpt_summary("test"), str)
    assert isinstance(gpt_keywords("test"), str)
    assert isinstance(gpt_translate("hello"), str)
