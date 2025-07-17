# AI Video Classifier with Audio Transcription

Classifies videos from Instagram, YouTube, TikTok using AI with 250+ specific categories and audio transcription.

## 🚀 Features
- **Visual Classification**: 250+ ultra-specific categories using OpenAI CLIP
- **Audio Transcription**: Google Cloud Speech-to-Text API
- **Keyword Extraction**: OpenAI-powered keyword extraction from audio
- **Combined Analysis**: Merges visual and audio insights for comprehensive results
- **Modular Architecture**: Separated video and audio processing modules

## 📁 Project Structure
```
├── main_classifier.py      # Main classifier combining video + audio
├── video_processor.py      # CLIP-based video analysis module  
├── audio_processor.py      # Audio transcription & keyword extraction
├── classify.py            # Command-line interface
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (for audio processing)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 3. For Instagram Videos
**You need a local API server running on localhost:3000**

### 4. Run Classification
```bash
# Quick test with full audio processing
python3 main_classifier.py

# Command line interface with audio
python3 classify.py "https://www.instagram.com/reel/your-video-url/"

# Disable audio processing for faster results
python3 classify.py "https://www.youtube.com/shorts/video-id" --no-audio

# With detailed frame analysis
python3 classify.py "https://www.tiktok.com/@user/video/id" --frames

# JSON output for API integration
python3 classify.py "https://video-url" --output json
```

## 📊 What You Get

The system analyzes videos and provides:
- **Visual Categories**: "step by step cooking recipe" (confidence: 0.89)
- **Audio Keywords**: ["cooking", "recipe", "ingredients", "tutorial"]  
- **Full Transcript**: Complete speech-to-text transcription
- **Combined Tags**: Merged visual and audio insights
- **Frame Analysis**: Per-frame classification results

## 🔧 Module Usage

### Video Processing Only
```python
from video_processor import VideoProcessor

processor = VideoProcessor()
video_path = processor.download_video_temp("https://video-url")
frames = processor.extract_frames(video_path)
results = processor.classify_frames(frames)
```

### Audio Processing Only  
```python
from audio_processor import AudioProcessor

processor = AudioProcessor()
audio_result = processor.process_audio_complete("video.mp4")
print(f"Transcript: {audio_result['transcript']}")
print(f"Keywords: {audio_result['keywords']}")
```

### Combined Processing
```python
from main_classifier import ReelClassifier

classifier = ReelClassifier()
result = classifier.classify_reel("https://video-url")
```

## 🔑 API Credentials
The system uses pre-configured API keys for:
- Google Cloud Speech-to-Text API  
- OpenAI Keyword Extraction API

## 📋 Requirements
- Python 3.9+
- FFmpeg (for audio processing)
- For Instagram: Local API server on localhost:3000