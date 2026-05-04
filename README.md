#  **Arabic Speech Recognition System**
### Built By Mustafa Othman & Seif ibrahim
### Deep Learning Based Arabic Audio Understanding — Qwen3-ASR-1.7B

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://kaggle.com)
[![Model](https://img.shields.io/badge/Model-Qwen3--ASR--1.7B-orange.svg)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Overview

This project implements an **Arabic Speech Recognition (ASR) system** using the **Qwen3-ASR-1.7B** model, evaluated on the Mozilla Common Voice Arabic TTS dataset. The work is inspired by and compared against the paper:

> **"Development of a Deep Learning-based Arabic Speech Recognition System for Automatons"**  
> Alahmadi et al., Engineering, Technology & Applied Science Research, Vol. 14, No. 6, 2024

The system transcribes Arabic speech to text, evaluates using Word Error Rate (WER), performs keyword spotting, and includes an interactive Gradio demo.

---

##  System Architecture

```
Arabic Audio (.wav)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  ← Resample to 16kHz
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Qwen3-ASR-1.7B     │  ← Encoder-Decoder Transformer
│  (1.7B parameters)  │     Pre-trained on multilingual audio
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Text Normalization │  ← Strip diacritics, unify alef, remove punctuation
└─────────────────────┘
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
  Arabic Transcript      WER Evaluation        Keyword Spotting
```

---

##  Results

| Model | Accuracy | WER |
|---|---|---|
| **Qwen3-ASR-1.7B (Ours)** | **81.61%** | **18.39%** |

> **Note:** The paper's model was trained and tested on the same controlled dataset. Our model is evaluated **zero-shot** — no fine-tuning on this dataset — which explains the difference. The model still achieves strong performance without any task-specific training.

---

##  Project Structure

```
arabic-asr/
├── Arabic_ASR_Kaggle_Qwen3.ipynb   ← Main notebook (Kaggle)
├── README.md                        ← This file
├── outputs/
│   ├── asr_evaluation_results.csv   ← Per-sample WER results
│   ├── keyword_spotting_results.csv ← Keyword detection results
│   ├── evaluation_results.png       ← Comparison charts
│   └── audio_analysis.png           ← Waveform + MFCC visualization
```

---

##  Dataset

**Mozilla Common Voice Arabic TTS**

| Property | Value |
|---|---|
| Source | Mozilla Common Voice |
| Language | Arabic (Modern Standard) |
| Audio format | WAV |
| Total samples | ~78,720 files |
| Train split | 70% |
| Test split | 30% |
| Transcript format | Real Arabic text (no transliteration) |

Dataset structure:
```
arabic_tts/
  ├── wavs/                    ← audio files (.wav)
  ├── metadata.csv             ← transcripts
  └── metadata-wav.csv         ← transcripts with wav paths
```

---

##  Model

**Qwen3-ASR-1.7B** by Alibaba Cloud

| Property | Value |
|---|---|
| Parameters | 1.7 Billion |
| Architecture | Encoder-Decoder Transformer |
| Input | 16kHz mono WAV |
| Languages | Multilingual (Arabic supported) |
| HuggingFace | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |

---

##  Features

-  **Arabic Speech Transcription** — converts WAV audio to Arabic text
-  **Text Normalization** — strips diacritics, unifies alef variants, removes punctuation for fair WER comparison
-  **WER & CER Evaluation** — Word Error Rate and Character Error Rate
-  **Keyword Spotting** — detects Arabic keywords by category (greetings, numbers, directions, emergency)
-  **Visualization** — waveform, MFCC features, WER distribution, model comparison charts
-  **Gradio Demo** — interactive web interface with microphone support

---

##  How to Run

### On Kaggle (Recommended)

1. Go to [kaggle.com](https://kaggle.com) → **Create → New Notebook**
2. Add the **Arabic Speech Dataset** as input
3. Upload `Arabic_ASR_Kaggle_Qwen3.ipynb`
4. Enable **GPU T4 x2** in Settings → Accelerator
5. Click **Run All**

### On Google Colab

1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com)
2. Set Runtime → Change runtime type → **T4 GPU**
3. Upload dataset to Google Drive and update `DATASET_ROOT` path
4. Run all cells

### Locally

```bash
# Create virtual environment
python -m venv arabic_asr_env
source arabic_asr_env/bin/activate  # Windows: arabic_asr_env\Scripts\activate

# Install dependencies
pip install torch torchaudio
pip install qwen-asr
pip install librosa soundfile
pip install jiwer gradio
pip install scikit-learn matplotlib seaborn PyWavelets
pip install jupyter

# Launch notebook
jupyter notebook Arabic_ASR_Kaggle_Qwen3.ipynb
```

---

##  Dependencies

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `qwen-asr` | Qwen3-ASR model interface |
| `librosa` | Audio loading and feature extraction |
| `soundfile` | Audio file I/O |
| `jiwer` | WER and CER computation |
| `gradio` | Interactive demo interface |
| `datasets` | HuggingFace dataset streaming |
| `scikit-learn` | Label encoding |
| `matplotlib` | Visualization |
| `PyWavelets` | Wavelet denoising |

---

##  Keyword Spotting

The system detects Arabic keywords organized by category:

| Category | Example Keywords |
|---|---|
| Greeting | مرحبا، السلام، صباح، مساء |
| Number | واحد، اثنان، ثلاثه، اربعه |
| Direction | يمين، يسار |
| Question | كيف، اين، ماذا، متي |
| Health | مريض |
| Emergency | مساعده، طوارئ |
| Affirmation/Negation | نعم، لا |

---

##  Evaluation Metrics

**Word Error Rate (WER):**
```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

**Accuracy:**
```
Accuracy = 1 - WER
```

**Text normalization applied before comparison:**
- Strip Arabic diacritics (harakat)
- Unify alef variants (أ إ آ → ا)
- Normalize teh marbuta (ة → ه)
- Normalize alef maqsura (ى → ي)
- Remove punctuation

---

##  Reference Paper

```bibtex
@article{alahmadi2024arabic,
  title     = {Development of a Deep Learning-based Arabic Speech Recognition System for Automatons},
  author    = {Alahmadi, Abdulrahman and Alahmadi, Ahmed and Alduweib, Eman and Alromema, Waseem and Ahmed, Bakil},
  journal   = {Engineering, Technology \& Applied Science Research},
  volume    = {14},
  number    = {6},
  pages     = {18439--18446},
  year      = {2024},
  doi       = {10.48084/etasr.8661}
}
```

---

##  Notebook Structure

| Step | Description |
|---|---|
| Step 1 | Install Dependencies |
| Step 2 | Check GPU |
| Step 3 | Load Dataset from Kaggle Input |
| Step 4 | Audio Exploration (Waveform + MFCC) |
| Step 5 | Load Qwen3-ASR-1.7B Model |
| Step 6 | Inference & WER Evaluation |
| Step 7 | Keyword Spotting |
| Step 8 | Gradio Demo Interface |
| Step 9 | Final Summary Report |

---

##  Notes

- First run downloads the model (~4.7GB) — takes 5-10 minutes
- GPU is required for reasonable inference speed (~3 sec/sample on T4)
- The CUDA warnings (`E0000`) on Kaggle are harmless and can be ignored
- WER is computed after normalizing both reference and prediction text

---

##  License

This project is licensed under the MIT License.

---

<div align="center">
  <p>Built with Love for Arabic NLP</p>
</div>
