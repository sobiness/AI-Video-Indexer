#!/usr/bin/env python3
"""
Audio processing module for video classification
Handles audio extraction, transcription, and keyword extraction
"""

import os
import tempfile
import logging
import requests
import base64
import ffmpeg
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles all audio processing tasks including transcription and keyword extraction"""
    
    def __init__(self, google_api_key: str = None, openai_endpoint: str = None, openai_api_key: str = None):
        """
        Initialize audio processor with API credentials
        
        Args:
            google_api_key: Google Cloud Speech-to-Text API key
            openai_endpoint: OpenAI API endpoint URL
            openai_api_key: OpenAI API key
        """
        # Default API credentials (can be overridden)
        self.google_api_key = google_api_key or "AIzaSyBqvoxXL84ARw-I7GkS8Y8yaI1szvtVu3c"
        self.openai_endpoint = openai_endpoint or "https://mlw313lea6.execute-api.us-east-1.amazonaws.com/development/genai"
        self.openai_api_key = openai_api_key or "rxQIb0rorO2HRkwcsahNQ4hxny1hUAVg2ZOKRBtE"
        
        logger.info("🎵 AudioProcessor initialized")
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to extracted audio file or None if failed
        """
        try:
            temp_dir = os.path.dirname(video_path)
            audio_path = os.path.join(temp_dir, "audio.wav")
            
            # Extract audio using ffmpeg with optimized settings for speech recognition
            (
                ffmpeg
                .input(video_path)
                .output(
                    audio_path, 
                    acodec='pcm_s16le',  # 16-bit PCM
                    ac=1,                # Mono channel
                    ar='16000'           # 16kHz sample rate (optimal for speech)
                )
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                logger.info("🎵 Audio extracted successfully")
                return audio_path
            else:
                logger.warning("⚠️ Audio extraction failed - file too small or doesn't exist")
                return None
                
        except ffmpeg.Error as e:
            logger.error(f"❌ FFmpeg error during audio extraction: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Audio extraction error: {e}")
            return None
    
    def transcribe_audio_google(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using Google Cloud Speech-to-Text API
        Handles both short and long audio files automatically
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                content = audio_file.read()
            
            # Check file size and implement proper truncation for long audio
            audio_size_mb = len(content) / (1024 * 1024)
            logger.info(f"📊 Audio file size: {audio_size_mb:.2f} MB")
            
            # Google Speech API sync limit is ~10MB but for reliability, we use smaller limit
            max_size = 800 * 1024  # 800KB limit for better reliability
            is_truncated = False
            if len(content) > max_size:
                logger.info("🎙️ Audio file is large, truncating for sync API compatibility...")
                content = content[:max_size]
                is_truncated = True
                logger.info("📏 Truncated audio to fit sync API limits")
            
            # Prepare the API request using REST API with API key
            url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.google_api_key}"
            
            # Prepare request data with optimized settings
            request_data = {
                "config": {
                    "encoding": "LINEAR16",
                    "sampleRateHertz": 16000,
                    "languageCode": "en-US",
                    "enableAutomaticPunctuation": True,
                    "model": "latest_short",
                    "useEnhanced": True,
                    "enableWordTimeOffsets": False,
                    "enableWordConfidence": False
                },
                "audio": {
                    "content": base64.b64encode(content).decode('utf-8')
                }
            }
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            logger.info("🎙️ Transcribing audio with Google Cloud Speech-to-Text...")
            response = requests.post(url, json=request_data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'results' in result and result['results']:
                    transcript = ""
                    for speech_result in result['results']:
                        if 'alternatives' in speech_result and speech_result['alternatives']:
                            transcript += speech_result['alternatives'][0].get('transcript', '') + " "
                    
                    transcript = transcript.strip()
                    if transcript:
                        logger.info(f"✅ Transcription successful: {len(transcript)} characters")
                        if is_truncated:
                            logger.info("ℹ️ Note: Transcription may be partial for long audio")
                        return transcript
                    else:
                        logger.warning("⚠️ Empty transcription result")
                        return None
                else:
                    logger.warning("⚠️ No speech detected in audio")
                    return None
            else:
                logger.error(f"❌ Google Speech API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return None
    
    def extract_keywords_openai(self, transcript: str) -> List[str]:
        """
        Extract keywords from transcript using OpenAI API
        
        Args:
            transcript: Transcribed text
            
        Returns:
            List of extracted keywords
        """
        try:
            if not transcript or len(transcript.strip()) < 10:
                logger.warning("⚠️ Transcript too short for keyword extraction")
                return []
            
            # Improved prompt for better keyword extraction
            prompt = "You are a keyword extraction specialist. Extract only the most relevant keywords from video content."
            question = f"""Extract the most important keywords and phrases from this video transcript. Focus on:
- Main topics and subjects
- Important actions or activities  
- Key objects or items mentioned
- Relevant descriptive terms
- Avoid common words, filler phrases, and stop words
Think like a common user would search for this content.

Return ONLY the keywords as a comma-separated list with no extra text or explanations.

Transcript: "{transcript}"

Keywords:"""
            
            # Use the correct format for this AWS Lambda OpenAI endpoint
            request_data = {
                "llm": "gpt-4",
                "temperature": 0.1,
                "top_p": 0.1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.1,
                "prompt": prompt,
                "question": question
            }
            
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.openai_api_key
            }
            
            logger.info("🧠 Extracting keywords with OpenAI...")
            logger.debug(f"Request endpoint: {self.openai_endpoint}")
            logger.debug(f"Request data keys: {list(request_data.keys())}")
            
            response = requests.post(self.openai_endpoint, json=request_data, headers=headers, timeout=30)
            
            logger.info(f"OpenAI API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"OpenAI Response: {result}")
                
                # Parse the response body (it should be in result.body)
                keywords_response = ""
                if 'body' in result:
                    try:
                        import json
                        # The body might be a string that needs parsing
                        if isinstance(result['body'], str):
                            try:
                                keywords_text = json.loads(result['body'])
                                keywords_response = str(keywords_text).strip()
                            except json.JSONDecodeError:
                                keywords_response = result['body'].strip()
                        else:
                            keywords_response = str(result['body']).strip()
                    except Exception:
                        keywords_response = str(result['body']).strip()
                else:
                    # Fallback: check for other possible response formats
                    keywords_response = str(result).strip()
                
                logger.info(f"Raw keywords response: {keywords_response}")
                
                # Parse keywords from response
                keywords = []
                if keywords_response:
                    # Split by comma and clean up
                    raw_keywords = keywords_response.split(',')
                    for kw in raw_keywords:
                        cleaned = kw.strip().lower()
                        # Remove quotes, numbers, brackets, and filter short words
                        cleaned = cleaned.strip('"\'()[]{}').strip()
                        if len(cleaned) > 2 and not cleaned.isdigit() and cleaned not in ['keywords', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for']:
                            keywords.append(cleaned)
                
                logger.info(f"✅ Extracted {len(keywords)} keywords from audio")
                return keywords[:15]  # Limit to 15 keywords
            else:
                logger.error(f"❌ OpenAI API error: {response.status_code}")
                logger.error(f"Response text: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Keyword extraction error: {e}")
            return []
    
    def process_audio_complete(self, video_path: str) -> Dict[str, any]:
        """
        Complete audio processing pipeline: extract audio, transcribe, extract keywords
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing transcript, keywords, and audio status
        """
        audio_result = {
            "transcript": None,
            "keywords": [],
            "has_audio": False,
            "audio_duration": 0
        }
        
        try:
            # Extract audio from video
            audio_path = self.extract_audio_from_video(video_path)
            if not audio_path:
                logger.warning("⚠️ No audio could be extracted from video")
                return audio_result
            
            # Check if audio file has meaningful content
            audio_size = os.path.getsize(audio_path)
            if audio_size < 1000:  # Less than 1KB
                logger.warning("⚠️ Audio file too small, likely no audio content")
                self._cleanup_audio_file(audio_path)
                return audio_result
            
            audio_result["has_audio"] = True
            logger.info(f"📊 Audio file size: {audio_size} bytes")
            
            # Transcribe audio
            transcript = self.transcribe_audio_google(audio_path)
            if transcript:
                audio_result["transcript"] = transcript
                
                # Extract keywords from transcript
                keywords = self.extract_keywords_openai(transcript)
                audio_result["keywords"] = keywords
                
                logger.info(f"🎯 Audio processing complete: {len(transcript)} chars transcript, {len(keywords)} keywords")
            else:
                logger.warning("⚠️ No transcript generated from audio")
            
            # Cleanup audio file
            self._cleanup_audio_file(audio_path)
            
            return audio_result
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            return audio_result
    
    def _cleanup_audio_file(self, audio_path: str):
        """Clean up temporary audio file"""
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug("🧹 Audio file cleaned up")
        except Exception as e:
            logger.warning(f"Could not cleanup audio file: {e}")
