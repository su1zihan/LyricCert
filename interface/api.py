"""
LyricCert API — FastAPI backend
Run with: uvicorn api:app --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import os
import re
import warnings
import nltk

warnings.filterwarnings("ignore")

# ── NLTK setup (same as original) ──────────────────────────────────────────
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)

from transformers import LongformerModel, LongformerConfig, LongformerTokenizer, PreTrainedModel

# ── Model definition (exact copy from original) ────────────────────────────
class LongformerMultiLabel(PreTrainedModel):
    config_class = LongformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = LongformerModel(config)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def mean_pool(self, last_hidden_state, attention_mask):
        m = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return {'logits': logits}


# ── Preprocessor (exact copy from original) ────────────────────────────────
class LongformerPreprocessor:
    def __init__(self):
        try:
            import nltk.stem
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            self.lemmatizer.lemmatize("testing", "v")
        except Exception:
            self.lemmatizer = None

        self.contractions = {
            r"\bI'm\b": "I am", r"\byou're\b": "you are", r"\bhe's\b": "he is",
            r"\bshe's\b": "she is", r"\bit's\b": "it is", r"\bwe're\b": "we are",
            r"\bthey're\b": "they are", r"\bI've\b": "I have", r"\byou've\b": "you have",
            r"\bwe've\b": "we have", r"\bthey've\b": "they have", r"\bI'll\b": "I will",
            r"\byou'll\b": "you will", r"\bhe'll\b": "he will", r"\bshe'll\b": "she will",
            r"\bwe'll\b": "we will", r"\bthey'll\b": "they will", r"\bI'd\b": "I would",
            r"\byou'd\b": "you would", r"\bhe'd\b": "he would", r"\bshe'd\b": "she would",
            r"\bwe'd\b": "we would", r"\bthey'd\b": "they would",
            r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
            r"\bwon't\b": "will not", r"\bwouldn't\b": "would not", r"\bcan't\b": "can not",
            r"\bcouldn't\b": "could not", r"\bshouldn't\b": "should not",
            r"\bmustn't\b": "must not", r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
            r"\baren't\b": "are not", r"\bisn't\b": "is not", r"\bain't\b": "am not",
            r"\blet's\b": "let us"
        }

    def wn_pos(self, t):
        return {"J": "a", "V": "v", "N": "n", "R": "r"}.get(t[0], "n")

    def expand_contractions_str(self, text):
        for pattern, replacement in self.contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def clean_and_lemmatize(self, text):
        import pandas as pd
        if pd.isna(text):
            return ""
        cleaned = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = self.expand_contractions_str(line)
            try:
                tokens = [w for w in nltk.word_tokenize(line.lower()) if w.isalpha()]
                if not tokens:
                    continue
                if self.lemmatizer:
                    tagged = nltk.pos_tag(tokens)
                    lemmas = [self.lemmatizer.lemmatize(w, self.wn_pos(t)) for w, t in tagged]
                    if lemmas:
                        cleaned.append(" ".join(lemmas))
                else:
                    cleaned.append(" ".join(tokens))
            except Exception:
                basic_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', line)]
                if basic_tokens:
                    cleaned.append(" ".join(basic_tokens))

        merged = " ".join(cleaned)
        merged = re.sub(r"[^a-zA-Z\s]", "", merged)
        return re.sub(r"\s+", " ", merged).strip()


# ── Load model once at startup ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None
preprocessor = None

def load_model():
    global tokenizer, model, preprocessor
    if not os.path.exists(MODEL_DIR):
        print(f"[WARNING] Model directory not found: {MODEL_DIR}")
        return

    tokenizer = LongformerTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    config = LongformerConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    config.num_labels = 4
    model = LongformerMultiLabel.from_pretrained(MODEL_DIR, config=config)
    model = model.to(device)
    model.eval()
    preprocessor = LongformerPreprocessor()
    print(f"[OK] Model loaded on {device}")

load_model()


# ── MusicContentRatingSystem (import from your existing file) ───────────────
try:
    from music_content_rating import MusicContentRatingSystem
except ImportError:
    # Inline fallback if the file isn't in the same directory
    class MusicContentRatingSystem:
        def calculate_rating(self, violence_score, sexual_score, language_score, substance_score):
            descriptors = []
            if sexual_score > 0.6:        descriptors.append("Strong Sexual Content")
            elif sexual_score > 0.35:     descriptors.append("Suggestive Content")
            elif sexual_score > 0.15:     descriptors.append("Mild Suggestive Themes")
            if violence_score > 0.6:      descriptors.append("Graphic Violence")
            elif violence_score > 0.35:   descriptors.append("Moderate Violence")
            elif violence_score > 0.15:   descriptors.append("Mild Violence")
            if substance_score > 0.5:     descriptors.append("Substance Use")
            elif substance_score > 0.25:  descriptors.append("Substance References")
            if language_score > 0.6:      descriptors.append("Strong Language")
            elif language_score > 0.35:   descriptors.append("Moderate Language")
            elif language_score > 0.15:   descriptors.append("Mild Language")

            if sexual_score > 0.75 or (sexual_score > 0.5 and violence_score > 0.5):
                rating = "M-AO"
            elif sexual_score > 0.6 or violence_score > 0.6 or language_score > 0.7:
                rating = "M-R"
            elif sexual_score > 0.35 or violence_score > 0.35 or language_score > 0.45 or substance_score > 0.5:
                rating = "M-T"
            elif sexual_score > 0.15 or violence_score > 0.15 or language_score > 0.2 or substance_score > 0.25:
                rating = "M-P"
            else:
                rating = "M-E"

            explanations = {
                "M-E":  "Suitable for all audiences. No significant content concerns detected.",
                "M-P":  "Generally clean. Parental guidance suggested for young children.",
                "M-T":  "Contains moderate content. Recommended for teen audiences and above.",
                "M-R":  "Contains strong content. Parental discretion strongly advised.",
                "M-AO": "Explicit content. Intended for adults 18 and older only.",
            }
            recommendations = {
                "M-E":  "Safe for all ages.",
                "M-P":  "Suitable for most audiences.",
                "M-T":  "Suitable for teens and above.",
                "M-R":  "Not suitable for minors.",
                "M-AO": "Adults only.",
            }
            return {
                "rating": rating,
                "descriptors": descriptors,
                "details": {
                    "rating_explanation": explanations[rating],
                    "recommendation": recommendations[rating],
                }
            }


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="LyricCert API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow the HTML file opened locally
    allow_methods=["POST"],
    allow_headers=["*"],
)


class LyricsRequest(BaseModel):
    lyrics: str


@app.post("/analyze")
def analyze(req: LyricsRequest):
    if model is None or tokenizer is None or preprocessor is None:
        return {"error": "Model not loaded. Check that /model directory exists."}

    lyrics = req.lyrics.strip()
    if not lyrics:
        return {"error": "No lyrics provided."}

    # ── Preprocess (exact same as original) ──
    cleaned = preprocessor.clean_and_lemmatize(lyrics)
    if not cleaned:
        probs = [0.0, 0.0, 0.0, 0.0]
    else:
        enc = tokenizer(
            cleaned,
            max_length=2048,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        global_attention_mask = torch.zeros_like(enc["input_ids"])
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                global_attention_mask=global_attention_mask.to(device)
            )
            logits = outputs["logits"].cpu().numpy()[0]
            probs = (1 / (1 + np.exp(-logits))).tolist()

    sexual, violence, substance, language = probs[0], probs[1], probs[2], probs[3]

    # ── Rating (exact same as original) ──
    mcr = MusicContentRatingSystem()
    mcr_result = mcr.calculate_rating(
        violence_score=violence,
        sexual_score=sexual,
        language_score=language,
        substance_score=substance,
    )

    return {
        "sexual":           round(sexual, 4),
        "violence":         round(violence, 4),
        "substance":        round(substance, 4),
        "language":         round(language, 4),
        "total_words":      len(lyrics.split()),
        "mcr_rating":       mcr_result["rating"],
        "mcr_descriptors":  mcr_result["descriptors"],
        "mcr_explanation":  mcr_result["details"]["rating_explanation"],
        "mcr_recommendation": mcr_result["details"]["recommendation"],
        "kid_safe":         mcr_result["rating"] in ["M-E", "M-P"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
