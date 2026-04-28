# LyricCert

A content rating system for song lyrics. Paste a song, get an audience rating across four content dimensions.

## Setup

**1. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the model**

[Download the model checkpoint](https://drive.google.com/drive/folders/1H-nVC4Q8OO_n_Bw5WyWgUKnXmjKHoDj8?usp=drive_link) from Google Drive, unzip it, and place the contents in a folder named `model/` next to `api.py`.

```
LyricCert/
├── api.py
├── lyriccert.html
├── ...
└── model/
    ├── config.json
    ├── model.safetensors
    └── ...
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

## Troubleshooting

**"Cannot reach the inference backend"**
The backend is not running. Open a terminal in the project folder and run `uvicorn api:app --port 8000`.

**"Model not loaded"**
The `model/` folder is missing or incomplete. Download the checkpoint again and make sure the folder sits next to `api.py`.

**Slow first analysis**
The first run on a fresh machine downloads NLTK data. Wait a minute, then try again.
