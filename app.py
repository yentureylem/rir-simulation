import streamlit as st
import pyroomacoustics as pra
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import convolve
import librosa
import librosa.display
import io
import soundfile as sf
import matplotlib.pyplot as plt

# DereverbCNN model (tam)
class DereverbCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

# Model yükle (deploy için CPU)
@st.cache_resource
def load_model():
    model = DereverbCNN()
    model.load_state_dict(torch.load('dereverb_ultra_fast.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

st.title("🧠 Neural Room Dereverberation Demo")
st.markdown("**Upload any audio → AI removes room echo instantly!**")

# Sidebar
st.sidebar.header("🎛️ Controls")
room_size = st.sidebar.slider("Room Size", 3.0, 8.0, 5.5)
absorption = st.sidebar.slider("Reverb Amount", 0.1, 0.8, 0.4)

# File upload
uploaded_file = st.file_uploader("🎤 Upload Audio (WAV/MP3)", type=['wav','mp3','flac'])

if uploaded_file is not None:
    # Audio yükle
    audio_bytes, sr = sf.read(uploaded_file)
    duration = len(audio_bytes) / sr
    st.write(f"📊 Duration: {duration:.1f}s, SR: {sr}Hz")
    
    if sr != 16000:
        audio_bytes = librosa.resample(audio_bytes, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # 2s chunk al
    chunk = min(32000, len(audio_bytes))
    clean_sig = audio_bytes[:chunk].astype(np.float32)
    
    col1, col2, col3 = st.columns(3)
    
    # RIR + Reverb
    with st.spinner("🎧 Simulating room reverb..."):
        room = pra.ShoeBox([room_size]*3, fs=16000, absorption=absorption)
        room.add_microphone([room_size/2]*3)
        room.add_source([room_size*0.3]*3, signal=clean_sig)
        room.compute_rir()
        rir = room.rir[0][0]
        
        reverb_sig = convolve(clean_sig, rir, mode='full')
        reverb_sig = reverb_sig[:len(clean_sig)]
    
    with col1:
        st.subheader("1. 🎵 Clean")
        st.audio(clean_sig)
    
    with col2:
        st.subheader("2. 🏠 Reverb")
        st.audio(reverb_sig)
    
    # Neural dereverb
    with col3:
        st.subheader("3. 🧠 AI Fixed")
        progress = st.progress(0)
        
        spec_transform = torch.nn.Sequential(
            lambda x: torch.stft(x, n_fft=512, hop_length=128, win_length=512, 
                               onesided=True, return_complex=True),
            lambda x: torch.abs(x),
            lambda x: torch.log(x + 1e-8)
        )
        
        test_spec = spec_transform(torch.tensor(reverb_sig).float())
        test_input = test_spec.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            pred_log = model(test_input)
            pred_mag = torch.exp(pred_log.squeeze())
            
            # Inverse STFT
            pred_audio = torch.istft(pred_mag, n_fft=512, hop_length=128, win_length=512,
                                   length=len(clean_sig))
            pred_np = pred_audio.numpy()
            pred_np = pred_np * 0.3 / (np.max(np.abs(pred_np)) + 1e-8)
        
        st.audio(pred_np)
        progress.progress(100)
    
    # Spectrograms
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    for i, (sig, title) in enumerate([(clean_sig, 'Clean'), (reverb_sig, 'Reverb'), (pred_np, 'AI Fixed')]):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', ax=axs[i])
        axs[i].set(title=title, xlabel='', ylabel='')
    
    plt.suptitle('Neural Dereverberation: Echo Removal', fontsize=14)
    st.pyplot(fig)

st.markdown("---")
st.markdown("**Made with ❤️ PyTorch + Pyroomacoustics | [Colab Training](link)**")
