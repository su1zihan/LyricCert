# LyricCert

AI-powered music content analysis system that provides content ratings for song lyrics using a fine-tuned Longformer model.

## Features

- Multi-label content classification (Sexual, Violence, Substance, Language)
- Music Content Rating (MCR) system (M-E, M-P, M-T, M-R, M-AO)
- Interactive radar chart visualization
- Real-time analysis via FastAPI backend

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

## Installation

**1. Clone the repository**

```bash
cd LyricCert/interface
```

**2. Create virtual environment (optional but recommended)**

```bash
python -m venv env
```

macOS/Linux:
```bash
source env/bin/activate
```

Windows:
```bash
env\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Prepare the model**

Download the pre-trained checkpoint from the link below:

🔗 [Google Drive — Pre-trained Checkpoint](https://drive.google.com/drive/folders/1H-nVC4Q8OO_n_Bw5WyWgUKnXmjKHoDj8?usp=drive_link)

After downloading:
1. Unzip the archive
2. Create a folder named `model/` one level above `interface/`
3. Copy all files into `LyricCert/model/`

Project structure:

```
LyricCert/
├── model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── interface/
    ├── api.py
    ├── app.py
    ├── music_content_rating.py
    ├── lyriccert.html
    └── requirements.txt
```

## Usage

**Step 1 — Start the FastAPI backend:**

```bash
cd interface
uvicorn api:app --port 8000
```

**Step 2 — Open the webpage:**

Open `lyriccert.html` directly in a browser. The page calls `http://localhost:8000/analyze` for model inference.

## How It Works

1. **Input** — Paste song lyrics into the text area
2. **Preprocessing** — Expands contractions, tokenizes and lemmatizes text, removes non-alphabetic characters
3. **Analysis** — Longformer model predicts content scores for 4 categories
4. **Rating** — MCR system calculates final rating based on content scores
5. **Visualization** — Interactive display with radar chart and score bars

## Content Rating System

| Rating | Description | Criteria |
|--------|-------------|----------|
| M-E | Everyone | Minimal mature content across all categories |
| M-P | Parental Guidance | Mild content, suitable for most audiences with parental awareness |
| M-T | Teen Audiences | Moderate content themes appropriate for teenagers |
| M-R | Restricted | Strong mature content, adult supervision recommended |
| M-AO | Adults Only | Intense explicit content, 18+ only |

## Model Architecture

- **Base Model:** Longformer
- **Task:** Multi-label classification
- **Max Sequence Length:** 2048 tokens
- **Output:** 4 sigmoid probabilities (0–1 range)
- **Labels:** Sexual, Violence, Substance, Language

## Troubleshooting

**Model not loading**
- Ensure all model files are in the `model/` directory (one level above `interface/`)
- Check that file names match the Hugging Face format
- Verify PyTorch and Transformers versions are compatible

**NLTK errors**
```python
import nltk
nltk.download('all')
```

**FastAPI backend not reachable**
- Make sure `uvicorn api:app --port 8000` is running before opening `lyriccert.html`
- Check that nothing else is using port 8000

**GPU out of memory**
```python
device = "cpu"  # Force CPU usage in analyze_lyrics_longformer()
```

## Performance

- Average analysis time: 5–10 seconds (CPU)
- Supported lyric length: Up to ~8000 words (2048 tokens after preprocessing)
