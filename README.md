# AI Video Classifier

Classifies videos from Instagram, YouTube, TikTok using AI with 250+ specific categories.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. For Instagram Videos
**You need a local API server running on localhost:3000**

```bash
python3 video_classifier.py
```


That's it!

## What You Get

The system analyzes videos and tells you exactly what type of content it is, like:
- "step by step cooking recipe" (confidence: 0.89)
- "makeup tutorial and transformation" (confidence: 0.92) 
- "hip hop dance routine" (confidence: 0.85)

## Requirements
- Python 3.9+
- For Instagram: Local API server on localhost:3000