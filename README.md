# EchoNet 🎙️

EchoNet is a lightweight, full-stack Speech-to-Text (STT) web application built to record multilingual audio (English and Hindi) directly from a browser and transcribe it using a custom-built deep learning model. 

Instead of relying on black-box commercial APIs, EchoNet orchestrates a custom Recurrent Neural Network (RNN) architecture built from scratch in PyTorch, utilizing Mel-Frequency Cepstral Coefficients (MFCC) for audio feature extraction and Connectionist Temporal Classification (CTC) Loss for sequence alignment.

---

## 🚀 Key Features
* **Browser-Native Audio Capture:** Simple, responsive interface to start, stop, and reset microphone recordings using the browser's `MediaRecorder` API.
* **Custom Acoustic Core:** A deep learning architecture engineered in PyTorch—no pre-trained model wrappers or external APIs.
* **Multilingual Transcription:** Built to process both English and Hindi vocal sequences.
* **Noise Resilient:** Specifically tested against common ambient conditions like fan noise, street traffic, keyboard typing, and background conversations.
* **Utility Tools:** One-click copy transcript to clipboard and local `.txt` file export

---

## 📐 System Architecture & Data Flow
1. **Frontend Layer:** Built with React, it requests secure microphone permissions, records audio chunks, compiles them into a binary `Blob`, and transmits a `FormData` object to the backend.
2. **Backend Gateway:** Built with FastAPI, the server accepts the file, converts the raw audio stream into a normalized 16kHz matrix, and extracts frequency dimensions.
3. **Acoustic Model:** A custom PyTorch `nn.Module` consisting of a linear projection layer, multi-layer Bidirectional Gate Recurrent Units (GRUs), and a character softmax classifier evaluated via CTC Loss.

---

## 🛠️ Tech Stack
* **Frontend:** React.js (Vite), Axios, HTML5 Media Stream APIs 
* **Backend:** FastAPI, Uvicorn, Python-Multipart 
* **Machine Learning Core:** PyTorch, Torchaudio 

---

## 📦 Local Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Node.js v18 or higher

### 1. Backend Setup
Navigate into your backend folder, create a virtual environment, and install dependencies:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install torch torchaudio fastapi uvicorn python-multipart
uvicorn main:app --reload --port 8000
```

### 1. Frontent Setup
```bash
cd ../frontend
npm install
```
Open src/config.js (or your preferred environment manager) and verify the backend pointer:
```JavaScript
export const API_BASE_URL = "http://localhost:8000";
```
```bash
npm run dev
```
