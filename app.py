import streamlit as st
import numpy as np
from scipy.signal import convolve
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import io

# Basit Wiener filter (model yok, instant çalışır)
def simple_dereverb(reverb_sig, rir, alpha=0.8):
    """Wiener filter based dereverberation"""
    # Inverse filter
    inv_filter = np.fft.rfft(rir)
    inv_filter = 1.0 / (np.abs(inv_filter) + 1e-8) * np.exp(-1j * np.angle(inv_filter))
    
    # FFT domain dereverb
    reverb_fft = np.fft.rfft(reverb_sig)
    dereverb_fft = reverb_fft * inv_filter[:len(reverb_fft)]
    
    # ISTFT
    dereverb = np.fft.irfft(dereverb_fft)
    return np.real(dereverb)[:len(reverb_sig)]

st.title("🏠 Room Impulse Response + Dereverberation")
st.markdown("**Upload audio → RIR simulation → Echo removal**")

uploaded_file = st.file_uploader("🎵 Upload Audio", type=['wav','mp3'])

if uploaded_file:
    # Load & resample
    audio, sr = sf.read(uploaded_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    chunk = audio[:32000]  # 2s
    
    # RIR generate (pyroomacoustics olmadan image source approx)
    def simple_rir(length=1024, rt60=0.5):
        t = np.arange(length)
        rir = np.exp(-3*np.log(10)*(t/16000)/rt60) * np.random.randn(length)*0.1
        return rir / np.max(np.abs(rir))
    
    rir = simple_rir()
    reverb_sig = convolve(chunk, rir, mode='full')[:32000]
    
    # Dereverb
    dereverb_sig = simple_dereverb(reverb_sig, rir)
    
    # Audio columns
    col1, col2, col3 = st.columns(3)
    with col1: st.audio(chunk, sample_rate=16000)
    with col2: st.audio(reverb_sig, sample_rate=16000) 
    with col3: st.audio(dereverb_sig, sample_rate=16000)
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Original', 'Reverberant', 'Dereverbed']
    sigs = [chunk, reverb_sig, dereverb_sig]
    
    for i, (sig, title) in enumerate(zip(sigs, titles)):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', ax=axs[i])
        axs[i].set(title=title)
    
    plt.tight_layout()
    st.pyplot(fig)

st.info("🚀 Neural version için local Streamlit çalıştır")
