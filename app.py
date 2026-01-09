import streamlit as st
import numpy as np
from scipy.signal import convolve
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

st.set_page_config(page_title="RIR Dereverb", layout="wide")
st.title("🏠 Room Impulse Response + Dereverberation")
st.markdown("**Upload audio → Simulate room echo → Remove it with DSP**")

# Sidebar controls
st.sidebar.header("🎛️ Room Settings")
rt60 = st.sidebar.slider("Reverb Time (RT60)", 0.2, 1.5, 0.5)
room_gain = st.sidebar.slider("Room Gain", 0.1, 0.8, 0.3)

# File upload
uploaded_file = st.file_uploader("🎤 Upload any audio", type=['wav', 'mp3', 'm4a', 'flac'])

if uploaded_file:
    # Load audio
    audio, sr = sf.read(uploaded_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Process 3s chunk
    chunk_size = min(48000, len(audio))
    clean_audio = audio[:chunk_size]
    
    # Generate synthetic RIR (no pyroomacoustics)
    def generate_rir(fs=16000, length=1024, rt60=0.5):
        """Exponential decay RIR"""
        t = np.arange(length) / fs
        decay = np.exp(-3 * np.log(10) * t / rt60)
        rir = decay * np.random.randn(length) * 0.1
        rir = rir / np.max(np.abs(rir))
        return rir
    
    rir = generate_rir(rt60=rt60)
    
    # Add reverb
    reverb_audio = convolve(clean_audio, rir * room_gain, mode='full')[:chunk_size]
    
    # Simple dereverberation (spectral subtraction)
    def spectral_dereverb(reverb_sig, rir_gain=0.3):
        stft = librosa.stft(reverb_sig)
        mag = np.abs(stft)
        
        # Spectral subtraction (reverb gain azalt)
        clean_mag = mag / (1 + rir_gain)
        clean_stft = clean_mag * np.exp(1j * np.angle(stft))
        
        return librosa.istft(clean_stft)
    
    dereverb_audio = spectral_dereverb(reverb_audio, room_gain)
    
    # UI: 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎵 Original")
        st.audio(clean_audio)
        st.caption(f"Duration: {chunk_size/16000:.1f}s")
    
    with col2:
        st.subheader("🏠 + Room Echo")
        st.audio(reverb_audio)
    
    with col3:
        st.subheader("🧠 DSP Dereverb")
        st.audio(dereverb_audio)
    
    # Spectrograms
    st.subheader("📊 Spectrogram Comparison")
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    
    for i, (audio_sig, title) in enumerate([
        (clean_audio, '1. Clean'), 
        (reverb_audio, '2. Reverberant'), 
        (dereverb_audio, '3. Dereverbed')
    ]):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sig)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', ax=axs[i])
        axs[i].set(title=title, xlabel='Time', ylabel='Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # RIR plot
    st.subheader("🔬 Room Impulse Response")
    fig2, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rir)
    ax.set_title(f'Synthetic RIR (RT60={rt60}s)')
    ax.set_xlabel('Samples')
    st.pyplot(fig2)

st.markdown("---")
st.markdown("""
**Tech Stack:** NumPy • SciPy • Librosa • Streamlit  
**Neural version:** [Colab Notebook](COLAB_LINK)  
**DSP Method:** Spectral subtraction on STFT domain
""")
