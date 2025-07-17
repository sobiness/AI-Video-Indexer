#!/bin/bash

echo "🚀 Setting up AI Video Classifier with Audio Transcription"

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Install FFmpeg
echo "🎵 Installing FFmpeg..."
if command -v brew &> /dev/null; then
    brew install ffmpeg
else
    echo "❌ Homebrew not found. Please install FFmpeg manually."
    echo "Visit: https://ffmpeg.org/download.html"
fi

# Environment variables setup
echo ""
echo "🔑 REQUIRED: Set up your API credentials"
echo ""
echo "1. Google Cloud Speech-to-Text:"
echo "   - Create project at: https://console.cloud.google.com/"
echo "   - Enable Speech-to-Text API"
echo "   - Create Service Account & download JSON key"
echo "   - Set: export GOOGLE_APPLICATION_CREDENTIALS='path/to/your/key.json'"
echo ""
echo "2. OpenAI API:"
echo "   - Get API key from: https://platform.openai.com/api-keys"
echo "   - Set: export OPENAI_API_KEY='your-openai-api-key'"
echo ""
echo "3. Add to your ~/.zshrc or ~/.bash_profile:"
echo "   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/google-key.json'"
echo "   export OPENAI_API_KEY='your-openai-api-key'"
echo ""
echo "✅ Setup complete! Don't forget to set your API keys."
