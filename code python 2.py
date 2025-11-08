import numpy as np
import librosa
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer

# --- Initialization ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Audio Utils (Feature Extraction) ---

def preprocess_audio(audio_path, sr=22050):
    """Loads, normalizes, and trims silence from audio."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        y = librosa.util.normalize(y)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed, sr
    except Exception:
        return None, None

def extract_acoustic_features(y, sr, n_mfcc=40):
    """Extracts and aggregates MFCCs, Chroma, and Spectral Contrast."""
    if y is None or sr is None: return np.zeros(n_mfcc + 12 + 7) 
    
    # Time-averaged features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    return np.hstack([mfccs, chroma, spec_contrast])

# --- Text Utils (Preprocessing and Tokenization) ---

def preprocess_text(text):
    """Cleans text: lowercasing, stopword removal, and lemmatization."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split()
    processed_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(processed_words)

def get_bert_encodings(texts, max_len=128, model_name='bert-base-uncased'):
    """Tokenizes clean text for BERT model input."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    encodings = tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    return encodings

def fuse_predictions(P_audio, P_text, w_audio=0.6):
    """Fusion P_fused = w_audio * P_audio + (1-w_audio) * P_text."""
    w_text = 1.0 - w_audio
    return (w_audio * P_audio) + (w_text * P_text)
