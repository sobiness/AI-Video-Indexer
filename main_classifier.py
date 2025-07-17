#!/usr/bin/env python3
"""
Main video classifier that combines video and audio processing
"""

import os
import logging
import tempfile
from typing import Dict, Optional
from video_processor import VideoProcessor
from audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReelClassifier:
    """AI-powered video classification using CLIP model with audio transcription"""
    
    def __init__(self, google_api_key: str = None, openai_endpoint: str = None, openai_api_key: str = None):
        """
        Initialize the complete video classification system
        
        Args:
            google_api_key: Google Cloud Speech-to-Text API key
            openai_endpoint: OpenAI API endpoint URL  
            openai_api_key: OpenAI API key
        """
        logger.info("🚀 Initializing ReelClassifier...")
        
        # Initialize video processor
        self.video_processor = VideoProcessor()
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            google_api_key=google_api_key,
            openai_endpoint=openai_endpoint, 
            openai_api_key=openai_api_key
        )
        
        logger.info("✅ ReelClassifier initialized successfully!")
    
    def classify_reel(self, url: str, top_k: int = 5, enable_audio: bool = True) -> Dict:
        """
        Main method to classify a video from URL using CLIP model with optional audio transcription
        
        Args:
            url: The video URL (Instagram, TikTok, YouTube, etc.)
            top_k: Number of top categories to return
            enable_audio: Whether to process audio for transcription and keywords
        
        Returns:
            Dictionary with detailed classification results, frame analysis, and audio processing
        """
        logger.info(f"🎬 Processing video: {url}")
        
        # Extract video info
        video_info = self.video_processor.get_video_info(url)
        if not video_info:
            return {"error": "Could not extract video information"}
        
        # Download video temporarily
        video_path = self.video_processor.download_video_temp(url)
        if not video_path:
            return {"error": "Could not download video"}
        
        try:
            # Extract frames for visual analysis
            frames = self.video_processor.extract_frames(video_path)
            if not frames:
                return {"error": "Could not extract frames from video"}
            
            # Classify using CLIP model
            aggregate_classifications, frame_classifications, broader_categories = self.video_processor.classify_frames(frames)
            if not aggregate_classifications:
                return {"error": "Could not classify video"}
            
            # Process audio if enabled
            audio_result = {
                "transcript": None,
                "keywords": [],
                "has_audio": False
            }
            
            if enable_audio:
                logger.info("🎵 Processing audio...")
                audio_result = self.audio_processor.process_audio_complete(video_path)
            else:
                logger.info("⏭️ Skipping audio processing (disabled)")
            
            # Get top categories
            top_categories = dict(list(aggregate_classifications.items())[:top_k])
            
            # Generate semantic tags (combine visual tags with audio keywords)
            visual_tags = self.video_processor.generate_semantic_tags(top_categories)
            all_tags = visual_tags.copy()
            if audio_result["keywords"]:
                all_tags.extend(audio_result["keywords"])
                # Remove duplicates while preserving order
                seen = set()
                all_tags = [tag for tag in all_tags if not (tag.lower() in seen or seen.add(tag.lower()))]
            
            # Prepare detailed frame-by-frame results
            frame_results = []
            for i, frame_classification in enumerate(frame_classifications):
                frame_top_3 = dict(list(frame_classification.items())[:3])
                frame_results.append({
                    "frame_number": i + 1,
                    "top_categories": frame_top_3,
                    "dominant_category": list(frame_classification.keys())[0] if frame_classification else "unknown",
                    "confidence": list(frame_classification.values())[0] if frame_classification else 0
                })
            
            return {
                "url": url,
                "title": video_info.get('title', 'Unknown'),
                "duration": video_info.get('duration', 0),
                "uploader": video_info.get('uploader', 'Unknown'),
                "top_categories": top_categories,
                "broader_categories": broader_categories,
                "semantic_tags": all_tags,
                "visual_tags": visual_tags,
                "audio_keywords": audio_result["keywords"],
                "transcript": audio_result["transcript"],
                "has_audio": audio_result["has_audio"],
                "all_categories": aggregate_classifications,
                "confidence_score": max(top_categories.values()) if top_categories else 0,
                "frame_by_frame": frame_results,
                "total_frames_analyzed": len(frames)
            }
            
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(video_path)
    
    def _cleanup_temp_files(self, video_path: str):
        """Clean up temporary video files"""
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                temp_dir = os.path.dirname(video_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                logger.debug("🧹 Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not cleanup temp file: {e}")
    
    def test_audio_keyword_extraction(self, sample_text: str = None) -> Dict:
        """Test audio keyword extraction functionality"""
        if not sample_text:
            sample_text = "Welcome to my cooking tutorial where I'll show you how to make delicious chocolate chip cookies from scratch using simple ingredients like flour, butter, sugar, eggs, and vanilla extract. This is a step by step baking recipe perfect for beginners."
        
        logger.info("🧪 Testing keyword extraction...")
        keywords = self.audio_processor.extract_keywords_openai(sample_text)
        
        return {
            "sample_text": sample_text,
            "extracted_keywords": keywords,
            "success": len(keywords) > 0
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    print("🚀 Initializing AI Video Classification System...")
    classifier = ReelClassifier()
    
    # Test URLs for different platforms
    instagram_url = "https://www.instagram.com/reel/DL92h8OIAOv/"
    youtube_url = "https://www.youtube.com/shorts/abc123"
    
    # Test with Instagram (API-based)
    test_url = instagram_url
    
    print(f"🎬 Processing: {test_url}")
    print("💡 Make sure your API server is running on localhost:3000")
    print("-" * 60)
    
    # Classify with full audio processing
    result = classifier.classify_reel(test_url, top_k=5, enable_audio=True)
    
    if "error" not in result:
        print(f"✅ CLASSIFICATION COMPLETE")
        print("=" * 60)
        print(f"📺 Title: {result['title']}")
        print(f"👤 Uploader: {result['uploader']}")
        print(f"⏱️  Duration: {result['duration']}s")
        print(f"🎯 Confidence: {result['confidence_score']:.3f}")
        print(f"🖼️  Frames Analyzed: {result['total_frames_analyzed']}")
        print(f"🎵 Audio Processed: {'Yes' if result['has_audio'] else 'No'}")
        
        print(f"\n🏆 TOP CATEGORIES:")
        for i, (category, score) in enumerate(result['top_categories'].items(), 1):
            print(f"  {i}. {category} ({score:.3f})")
        
        print(f"\n🌐 BROADER CATEGORIES:")
        for category, score in list(result['broader_categories'].items())[:5]:
            print(f"  • {category.replace('_', ' ').title()}: {score:.3f}")
        
        # Show visual tags if available
        if result['visual_tags']:
            print(f"\n🔖 Visual Tags: {', '.join(result['visual_tags'])}")
        else:
            print(f"\n🔖 Visual Tags: [Generated from top categories]")
        
        # Show audio results if available
        if result['audio_keywords']:
            print(f"\n🎙️ Audio Keywords: {', '.join(result['audio_keywords'])}")
        else:
            print(f"\n🎙️ Audio Keywords: [No keywords extracted]")
        
        if result['transcript']:
            transcript_preview = result['transcript'][:200] + "..." if len(result['transcript']) > 200 else result['transcript']
            print(f"\n📝 Transcript: {transcript_preview}")
        else:
            print(f"\n📝 Transcript: [No transcript available]")
        
        # Show combined tags
        if result['semantic_tags']:
            print(f"\n🏷️ Combined Tags: {', '.join(result['semantic_tags'][:10])}")
        else:
            print(f"\n🏷️ Combined Tags: [No tags generated]")
            
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("🎯 Classification Complete!")
    print("💡 Use classify.py for command-line interface with more options")
