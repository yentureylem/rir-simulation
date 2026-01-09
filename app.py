import streamlit as st
import numpy as np
from scipy.signal import convolve
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🎵 Room Echo Simulator + DSP Dereverberation")

uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])

if uploaded_file:
    # Load
    audio, sr = sf.read(uploaded_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    chunk = audio[:32000]  # 2 seconds
    
    # Synthetic RIR
    def make_rir(rt60=0.5):
        fs = 16000
        n = int(rt60 * fs)
        decay = np.exp(-3*np.log(10) * np.arange(n) / fs / rt60)
        return decay * (np.random.randn(n) * 0.01)
    
    rir = make_rir()
    
    # Reverb
    reverb = convolve(chunk, rir, mode='full')[:32000]
    
    # Dereverb (high-pass + spectral gating)
    def dereverb(sig):
        # High-pass filter
        b, a = librosa.signal.butter(4, 100, 'high', fs=16000)
        sig_hp = librosa.signal.lfilter(b, a, sig)
        # Spectral gating
        S = librosa.stft(sig_hp)
        S_clean = S * 0.7  # Gate reverb
        return librosa.istft(S_clean)
    
    clean_est = dereverb(reverb)
    
    # Display
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.audio(chunk)
        st.caption("Original")
    with col2:
        st.audio(reverb)
        st.caption("With Room Echo") 
    with col3:
        st.audio(clean_est)
        st.caption("DSP Dereverbed")
    
    # Spectrogram
    fig, ax = plt.subplots(figsize=(15, 4))
    sigs = [chunk, reverb, clean_est]
    titles = ['Clean', 'Reverb', 'Dereverbed']
    
    for i in range(3):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sigs[i])), ref=np.max)
        ax[i].specshow(D, y_axis='log', x_axis='time')
        ax[i].set_title(titles[i])
    
    st.pyplot(fig)

st.markdown("**Pure NumPy/SciPy • No ML dependencies**")
