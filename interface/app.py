import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import torch
import torch.nn as nn
import os
import warnings
import random
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go

st.set_page_config(
    page_title="LyricCert",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from music_content_rating import MusicContentRatingSystem
from transformers import LongformerModel, LongformerConfig, PreTrainedModel

TRANSFORMERS_AVAILABLE = True
_transformers_warning = None
try:
    from transformers import LongformerTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    _transformers_warning = "Transformers library not available. Running in demo mode."

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# NLTK setup
try:
    nltk.data.path.append(os.path.expanduser('~/nltk_data'))
except:
    pass

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.warning(f"NLTK download warning: {e}. Some features may be limited.")

if _transformers_warning:
    st.warning(_transformers_warning)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #f8f9fa;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --bg-hover: #f1f3f5;
        --accent-green: #00DC82;
        --accent-blue: #0ea5e9;
        --accent-purple: #8b5cf6;
        --accent-pink: #ec4899;
        --warning: #f59e0b;
        --danger: #ef4444;
        --success: #10b981;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        --border: #e5e7eb;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    .main, .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .header-container {
        margin-bottom: 2rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(0, 220, 130, 0.08), rgba(139, 92, 246, 0.08));
        border-radius: 24px;
    }

    .app-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-green), var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    .rating-card {
        background: var(--bg-card);
        border: 2px solid var(--border);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .rating-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    .rating-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
    }

    .rating-badge.m-e {
        background: linear-gradient(135deg, var(--accent-green), #00f5a0);
    }

    .rating-badge.m-p {
        background: linear-gradient(135deg, var(--accent-blue), #38bdf8);
    }

    .rating-badge.m-t {
        background: linear-gradient(135deg, var(--warning), #fbbf24);
    }

    .rating-badge.m-r {
        background: linear-gradient(135deg, var(--danger), #f87171);
    }

    .rating-badge.m-ao {
        background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
    }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .metric-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transform: translateY(-2px);
        border-color: var(--accent-green);
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .metric-bar {
        width: 100%;
        height: 8px;
        background: #e5e7eb;
        border-radius: 100px;
        margin-top: 1rem;
        overflow: hidden;
    }

    .metric-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 1s ease;
    }

    .metric-fill.low {
        background: linear-gradient(90deg, var(--success), var(--accent-green));
    }

    .metric-fill.medium {
        background: linear-gradient(90deg, var(--warning), #fbbf24);
    }

    .metric-fill.high {
        background: linear-gradient(90deg, var(--danger), #f87171);
    }

    .safety-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-weight: 700;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .safety-badge.safe {
        background: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 2px solid #10b981;
    }

    .safety-badge.unsafe {
        background: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 2px solid #ef4444;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .descriptor-pill {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: var(--bg-hover);
        border: 1px solid var(--border);
        border-radius: 100px;
        font-size: 0.875rem;
        margin: 0.25rem;
        color: var(--text-primary);
        font-weight: 500;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent-green), var(--accent-blue)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(0, 220, 130, 0.3) !important;
    }

    textarea {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 12px !important;
    }

    textarea:focus {
        border-color: var(--accent-green) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Custom model class
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


class LongformerPreprocessor:
    def __init__(self):
        try:
            import nltk.stem
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            test_result = self.lemmatizer.lemmatize("testing", "v")
            if not isinstance(test_result, str):
                raise ValueError("Lemmatizer returned unexpected type")
        except Exception as e:
            st.warning(f"WordNet lemmatizer initialization failed: {e}. Using basic text processing.")
            self.lemmatizer = None

        self.contractions = {
            r"\bI'm\b": "I am", r"\byou're\b": "you are", r"\bhe's\b": "he is", r"\bshe's\b": "she is",
            r"\bit's\b": "it is", r"\bwe're\b": "we are", r"\bthey're\b": "they are",
            r"\bI've\b": "I have", r"\byou've\b": "you have", r"\bwe've\b": "we have", r"\bthey've\b": "they have",
            r"\bI'll\b": "I will", r"\byou'll\b": "you will", r"\bhe'll\b": "he will", r"\bshe'll\b": "she will",
            r"\bwe'll\b": "we will", r"\bthey'll\b": "they will",
            r"\bI'd\b": "I would", r"\byou'd\b": "you would", r"\bhe'd\b": "he would", r"\bshe'd\b": "she would",
            r"\bwe'd\b": "we would", r"\bthey'd\b": "they would",
            r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
            r"\bwon't\b": "will not", r"\bwouldn't\b": "would not", r"\bcan't\b": "can not",
            r"\bcouldn't\b": "could not", r"\bshouldn't\b": "should not", r"\bmustn't\b": "must not",
            r"\bwasn't\b": "was not", r"\bweren't\b": "were not", r"\baren't\b": "are not", r"\bisn't\b": "is not",
            r"\bain't\b": "am not", r"\blet's\b": "let us"
        }

    def wn_pos(self, t):
        return {"J": "a", "V": "v", "N": "n", "R": "r"}.get(t[0], "n")

    def expand_contractions_str(self, text):
        for pattern, replacement in self.contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def clean_and_lemmatize(self, text):
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
                    if tokens:
                        cleaned.append(" ".join(tokens))
            except Exception:
                basic_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', line)]
                if basic_tokens:
                    cleaned.append(" ".join(basic_tokens))

        merged = " ".join(cleaned)
        merged = re.sub(r"[^a-zA-Z\s]", "", merged)
        return re.sub(r"\s+", " ", merged).strip()


def parse_arguments():
    parser = argparse.ArgumentParser(description='LyricCert - AI-Powered Lyric Content Assessment')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--host', type=str, default='localhost')

    import sys
    filtered_argv = []
    skip_next = False

    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('--server.') or arg.startswith('--global.') or arg.startswith('--logger.'):
            if '=' not in arg and i + 1 < len(sys.argv):
                skip_next = True
            continue
        if i == 0 and ('streamlit' in arg or 'run' in arg):
            continue
        filtered_argv.append(arg)

    if not any('--model-path' in arg for arg in filtered_argv):
        return argparse.Namespace(model_path=None, port=8501, host='localhost')

    args = parser.parse_args(filtered_argv[1:] if filtered_argv else [])
    return args


@st.cache_resource(show_spinner=False)
def load_longformer_model():
    try:
        if not TRANSFORMERS_AVAILABLE:
            return None, None, None

        from transformers import LongformerTokenizer, LongformerConfig

        args = parse_arguments()
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if args.model_path:
            model_dir = os.path.abspath(args.model_path)
        else:
            model_dir = os.path.join(script_dir, "model")

        if not os.path.exists(model_dir):
            return None, None, None

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = LongformerTokenizer.from_pretrained(model_dir, local_files_only=True)
        config = LongformerConfig.from_pretrained(model_dir, local_files_only=True)
        config.num_labels = 4

        model = LongformerMultiLabel.from_pretrained(model_dir, config=config)
        model = model.to(device)
        model.eval()

        preprocessor = LongformerPreprocessor()

        return tokenizer, model, preprocessor

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def analyze_lyrics_longformer(lyrics, tokenizer, model, preprocessor):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cleaned = preprocessor.clean_and_lemmatize(lyrics)
    if not cleaned:
        return {
            'sexual': 0.0,
            'violence': 0.0,
            'substance': 0.0,
            'language': 0.0,
            'total_words': len(lyrics.split()),
            'confidence_scores': [0, 0, 0, 0],
            'raw_probabilities': [0, 0, 0, 0]
        }

    enc = tokenizer(cleaned, max_length=2048, truncation=True,
                    padding="max_length", return_tensors="pt")

    global_attention_mask = torch.zeros_like(enc["input_ids"])
    global_attention_mask[:, 0] = 1

    with torch.no_grad():
        inputs = {
            'input_ids': enc['input_ids'].to(device),
            'attention_mask': enc['attention_mask'].to(device),
            'global_attention_mask': global_attention_mask.to(device)
        }
        outputs = model(**inputs)
        logits = outputs['logits'].cpu().numpy()[0]
        probs = 1 / (1 + np.exp(-logits))

    results = {
        'sexual': float(probs[0]),
        'violence': float(probs[1]),
        'substance': float(probs[2]),
        'language': float(probs[3]),
        'total_words': len(lyrics.split()),
        'confidence_scores': probs.tolist(),
        'raw_probabilities': probs.tolist()
    }

    return results


def classify_lyrics(lyrics, tokenizer, model, preprocessor):
    results = analyze_lyrics_longformer(lyrics, tokenizer, model, preprocessor)

    mcr_system = MusicContentRatingSystem()
    mcr_result = mcr_system.calculate_rating(
        violence_score=results['violence'],
        sexual_score=results['sexual'],
        language_score=results['language'],
        substance_score=results['substance']
    )

    results['mcr_rating'] = mcr_result['rating']
    results['mcr_descriptors'] = mcr_result['descriptors']
    results['mcr_recommendation'] = mcr_result['details']['recommendation']
    results['mcr_explanation'] = mcr_result['details']['rating_explanation']
    results['kid_safe'] = mcr_result['rating'] in ['M-E', 'M-P']

    safety_level_map = {
        'M-E': 'Clean',
        'M-P': 'Clean',
        'M-T': 'Moderate',
        'M-R': 'Explicit',
        'M-AO': 'Explicit'
    }
    results['safety_level'] = safety_level_map.get(mcr_result['rating'], 'Clean')

    color_map = {
        'M-E': '#00DC82',
        'M-P': '#00A3FF',
        'M-T': '#F59E0B',
        'M-R': '#EF4444',
        'M-AO': '#9333EA'
    }
    results['rating_color'] = color_map.get(mcr_result['rating'], '#71717A')

    return results


def display_results(results):
    kid_safe = results.get('kid_safe', True)
    mcr_rating = results.get('mcr_rating', 'M-E')
    mcr_descriptors = results.get('mcr_descriptors', [])
    mcr_explanation = results.get('mcr_explanation', '')

    rating_descriptions = {
        'M-E': 'Everyone',
        'M-P': 'Parental Guidance',
        'M-T': 'Teen Audiences',
        'M-R': 'Restricted',
        'M-AO': 'Adults Only'
    }

    rating_detail = rating_descriptions.get(mcr_rating, '')
    mcr_class = mcr_rating.lower().replace('-', '-')

    # columns
    col1, col2 = st.columns([40, 60])

    with col1:
        # Rating Card
        st.markdown(f"""
        <div class="rating-card">
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
                <div class="rating-badge {mcr_class}" style="width: 130px; height: 130px; font-size: 2.5rem;">
                    {mcr_rating}
                </div>
                <h2 style="font-size: 1.75rem; margin-top: 1.5rem; margin-bottom: 0.5rem; color: var(--text-primary);">{rating_detail}</h2>
                <div class="safety-badge {'safe' if kid_safe else 'unsafe'}" style="margin-top: 0.75rem;">
                    {'✓ Kid Safe' if kid_safe else '⚠ Not Kid Safe'}
                </div>
                <p style="color: var(--text-secondary); margin-top: 1rem; font-size: 1rem; max-width: 320px; line-height: 1.6;">
                    {mcr_explanation}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Content Descriptors Card
        st.markdown(f"""
        <div class="rating-card" style="margin-top: 1rem;">
            <h3 style="font-size: 1rem; color: var(--text-secondary); margin-bottom: 1rem; font-weight: 600; text-align: center; text-transform: uppercase; letter-spacing: 0.05em;">Content Descriptors</h3>
            <div style="text-align: center;">
                {''.join([f'<span class="descriptor-pill" style="padding: 0.6rem 1.2rem; font-size: 0.95rem;">{desc}</span>' for desc in mcr_descriptors]) if mcr_descriptors else '<p style="color: var(--text-muted); font-size: 1rem;">No content warnings</p>'}
            </div>
        </div>
        """, unsafe_allow_html=True)


        with st.expander("Analysis Details", expanded=False):
            total_words = results.get('total_words', 0)
            st.markdown(f"""
            <div style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6;">
                <p><strong>Model:</strong> Longformer</p>
                <p><strong>Categories:</strong> Sexual, Violence, Substance, Language</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        language = results.get('language', 0)
        violence = results.get('violence', 0)
        sexual = results.get('sexual', 0)
        substance = results.get('substance', 0)

        categories = ['Sexual', 'Violence', 'Substance', 'Language']
        values = [sexual, violence, substance, language]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 220, 130, 0.2)',
            line=dict(color='rgb(0, 220, 130)', width=3),
            marker=dict(size=12, color='rgb(0, 220, 130)'),
            name='Content Scores'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=13, color='#6b7280'),
                    gridcolor='#e5e7eb',
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                ),
                angularaxis=dict(
                    tickfont=dict(size=15, color='#1f2937', family='Inter'),
                    linecolor='#e5e7eb'
                ),
                bgcolor='rgba(255, 255, 255, 0.5)'
            ),
            showlegend=False,
            height=380,
            margin=dict(l=80, r=80, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        metrics = [
            ('Sexual', sexual, '#ec4899'),
            ('Violence', violence, '#ef4444'),
            ('Substance', substance, '#f59e0b'),
            ('Language', language, '#8b5cf6')
        ]
        st.markdown("""
                <style>
                .flip-card {
                    background-color: transparent;
                    perspective: 1000px;
                    cursor: pointer;
                }
                .flip-card-inner {
                    position: relative;
                    width: 100%;
                    height: 100%;
                    text-align: center;
                    transition: transform 0.6s;
                    transform-style: preserve-3d;
                }
                .flip-card:hover .flip-card-inner {
                    transform: rotateY(180deg);
                }
                .flip-card-front, .flip-card-back {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    backface-visibility: hidden;
                    border-radius: 8px;
                }
                .flip-card-front {
                    background: var(--bg-card);
                    border: 1px solid var(--border);
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
                    padding: 0.5rem;
                }
                .flip-card-back {
                    background: linear-gradient(135deg, #1f2937, #374151);
                    border: 1px solid #4b5563;
                    transform: rotateY(180deg);
                    padding: 0.5rem;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }
                </style>
                """, unsafe_allow_html=True)


        grid_html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 1rem;">'

        for label, value, color in metrics:
            percentage = value * 100
            raw_value = value


            front = (f'<div style="font-size: 0.65rem; color: var(--text-muted); font-weight: 600; '
                     f'text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">{label}</div>'
                     f'<div style="font-size: 1rem; font-weight: 700; color: {color}; margin-bottom: 0.3rem;">{percentage:.2f}%</div>'
                     f'<div style="width: 100%; height: 3px; background: #e5e7eb; border-radius: 100px; overflow: hidden;">'
                     f'<div style="height: 100%; width: {percentage}%; background: {color}; border-radius: 100px;"></div></div>')

            back = (f'<div style="font-size: 0.65rem; color: #9ca3af; font-weight: 600; '
                    f'text-transform: uppercase; margin-bottom: 0.5rem;">RAW OUTPUT</div>'
                    f'<div style="font-size: 1.1rem; font-weight: 700; color: {color}; margin-bottom: 0.25rem;">{raw_value:.4f}</div>'
                    f'<div style="font-size: 0.75rem; color: #d1d5db;">{label}</div>')

            grid_html += (f'<div class="flip-card" style="height: 80px;">'
                          f'<div class="flip-card-inner">'
                          f'<div class="flip-card-front">{front}</div>'
                          f'<div class="flip-card-back">{back}</div>'
                          f'</div></div>')

        grid_html += '</div>'
        st.markdown(grid_html, unsafe_allow_html=True)

def main():
    st.markdown("""
    <div class="header-container">
        <h1 class="app-title">LyricCert</h1>
        <p class="app-subtitle">AI-Powered Music Content Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    longformer_tokenizer, longformer_model, longformer_preprocessor = load_longformer_model()

    if not all([longformer_tokenizer, longformer_model, longformer_preprocessor]):
        st.info("Running in demo mode")

    col_input, col_results = st.columns([30, 70])

    with col_input:
        st.markdown("""
        <div style="background: var(--bg-card); border: 2px solid var(--border); border-radius: 16px; padding: 1.5rem;">
            <h3 style="margin-bottom: 1rem; color: var(--text-primary); font-size: 1.1rem;"> Enter Lyrics</h3>
        </div>
        """, unsafe_allow_html=True)

        lyrics = st.text_area(
            "Lyrics",
            height=400,
            placeholder="Paste your song lyrics here for content analysis...",
            label_visibility="collapsed"
        )

        if lyrics:
            word_count = len(lyrics.split())
            char_count = len(lyrics)
            st.caption(f"{word_count} words • {char_count} characters")

        analyze_clicked = st.button("Analyze Content", type="primary", use_container_width=True)

    with col_results:
        if not analyze_clicked:
            st.markdown("""
            <div class="rating-card" style="height: 500px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🎵</div>
                    <h3 style="color: var(--text-primary); margin-bottom: 1rem; font-size: 1.5rem;">Ready to Analyze</h3>
                    <p style="color: var(--text-secondary); max-width: 400px; font-size: 1rem; line-height: 1.6;">
                        Paste your lyrics and click analyze to get instant content ratings.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if analyze_clicked:
            if not lyrics.strip():
                st.warning("Please enter lyrics to analyze")
            else:
                with st.spinner("Analyzing content..."):
                    try:
                        if all([longformer_tokenizer, longformer_model, longformer_preprocessor]):
                            results = classify_lyrics(
                                lyrics,
                                longformer_tokenizer,
                                longformer_model,
                                longformer_preprocessor
                            )
                            display_results(results)
                        else:
                            st.error("Model not loaded")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem; color: var(--text-muted); font-size: 0.875rem; border-top: 1px solid var(--border); margin-top: 3rem;">
        <p>© 2025 LyricCert</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()