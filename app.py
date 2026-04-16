88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
import os

</style>
""", unsafe_allow_html=True)

# --- UTIL ---
def get_stats(text):
    words = text.split()
    count = len(words)
    secs = (count / 150) * 60
    return count, f"{int(secs // 60)}M {int(secs % 60)}S"

# --- HEADER ---
st.markdown("<div class='header'>XYLA MORPH V3 / CORE</div>", unsafe_allow_html=True)

# --- LAYOUT ---
col1, col2 = st.columns([1,1])

# --- LEFT PANEL ---
with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("ENGINE CONFIG")
    model_size = st.selectbox("MODEL", ["tiny","base","small","medium","large"], index=1)

    st.subheader("UPLOAD AUDIO")
    audio = st.file_uploader("Drop audio file", type=["mp3","wav","m4a"])

    if audio:
        st.success(f"Loaded: {audio.name}")
        st.audio(audio)
        st.session_state['fn'] = os.path.splitext(audio.name)[0]

        if st.button("START TRANSCRIPTION"):
            with st.spinner("Processing..."):
                with open("temp.mp3","wb") as f:
                    f.write(audio.getbuffer())

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = whisper.load_model(model_size, device=device)
                result = model.transcribe("temp.mp3")

                st.session_state['txt'] = result["text"]
                os.remove("temp.mp3")

    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT PANEL ---
with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("TRANSCRIPT OUTPUT")

    if 'txt' in st.session_state:
        text = st.text_area("", value=st.session_state['txt'], height=350)

        wc, est = get_stats(text)

        st.markdown(f"""
        <div class='metric'>
            <span>WORDS: {wc}</span>
            <span>EST TIME: {est}</span>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            "EXPORT TXT",
            data=text,
            file_name=f"{st.session_state['fn']}.txt"
        )
    else:
        st.info("Waiting for audio input...")

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
device_status = "GPU" if torch.cuda.is_available() else "CPU"
st.caption(f"STATUS: {device_status} READY")
