# LyricCert

A content rating system for song lyrics. Paste a song, get an audience rating across four content dimensions.

## Setup

**1. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**2. Set up the model**

The trained model is provided as `model.zip` in the supplementary materials. Unzip it and place the resulting `model/` folder next to `api.py`.

```
LyricCert/
├── api.py
├── lyriccert.html
├── ...
└── model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    └── special_tokens_map.json
```

## Run

**1. Start the backend**

```bash
uvicorn api:app --port 8000
```

Wait for `[OK] Model loaded` to appear in the terminal. Keep this window open.

**2. Open the page**

Double-click `lyriccert.html` to open it in your browser.

## Use

1. Paste any song lyrics into the left panel
2. Click **Analyze**
3. The right panel shows the audience rating, a breakdown across four content categories, and any content warnings

Use the icon in the top right of the page to switch between light and dark mode.

## Rating system

| Rating | Audience | Criteria |
|--------|----------|----------|
| M-E    | Everyone | Minimal mature content across all categories |
| M-P    | Parental Guidance | Mild content, suitable for most audiences with parental awareness |
| M-T    | Teen Audiences | Moderate content themes appropriate for teenagers |
| M-R    | Restricted | Strong mature content, adult supervision recommended |
| M-AO   | Adults Only | Intense explicit content, 18+ only |

## Troubleshooting

**"Cannot reach the inference backend"**
The backend is not running. Open a terminal in the project folder and run `uvicorn api:app --port 8000`.

**"Model not loaded"**
The `model/` folder is missing or incomplete. Make sure `model.zip` was fully unzipped and the resulting folder sits next to `api.py`. The folder should contain `config.json`, `model.safetensors`, and the tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json`).

**Slow first analysis**
The first run on a fresh machine downloads NLTK data. Wait a minute, then try again.

**Resource `punkt_tab` not found**
This means the NLTK data download did not finish. Stop the backend, then run this once in a Python shell, then start the backend again:

```python
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```
