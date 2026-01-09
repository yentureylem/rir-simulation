<div align="center">

# 🧠 Room Impulse Response + Neural Dereverberation

**Upload audio → Simulate room acoustics → Remove echo with DSP/CNN!**

</div>

## 🎬 Live Demo
<div align="center">
<img src="cv_results.png" width="100%"/>
</div>

**Before (Reverb tail):** Long low-frequency smear  
**After (Dereverb):** Clean spectral profile

## 🚀 Features
🎵 Real-time audio upload (WAV/MP3)
🏠 Synthetic RIR generation (RT60 control)
🔬 Convolution reverb simulation
🧠 DSP dereverberation (spectral gating)
📊 Live spectrogram visualization

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| RIR Generation | Exponential decay model |
| Reverb | SciPy convolution |
| Dereverberation | Spectral subtraction |
| Frontend | Streamlit |
| Audio | Librosa STFT/iSTFT |
| Backend | NumPy / SciPy |

## 📈 Training Results (Neural Version)
Dataset: 100 randomized rooms
Model: CNN (spectrogram→spectrogram)
Epochs: 60
Loss: 1.72 → 1.22 (29% reduction)

![Training Loss](loss_curve.png)

## 🎯 Usage
1. **Live Demo:** [Click here](https://rir-simulation-jp8bgxwgh4xqp4uyretgcw.streamlit.app/)
2. **Train Neural Model:** 
3. **Local:** `pip install -r requirements.txt && streamlit run app.py`

## 📁 File Structure
rir-simulation/
├── app.py # Streamlit web app
├── requirements.txt # Dependencies
├── cv_results.png # Demo screenshot
└── README.md # You're reading it!


## 🔬 How It Works

```python
# 1. Synthetic RIR (exponential decay)
rir = exp(-3*log(10)*t/RT60) * noise

# 2. Convolution reverb  
reverb = conv(audio, rir)

# 3. Spectral dereverberation
S_clean = |STFT(reverb)| / (1 + reverb_gain) * phase
audio_clean = iSTFT(S_clean)
🎓 Learning Outcomes
Audio signal processing fundamentals

Room impulse response theory

Spectral domain operations

Real-time web deployment

PyTorch CNN training (bonus)
