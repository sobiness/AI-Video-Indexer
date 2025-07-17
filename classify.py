#!/usr/bin/env python3
"""
Simple command-line interface for AI-powered video classification
"""

import argparse
import json
import sys
from main_classifier import ReelClassifier

def main():
    parser = argparse.ArgumentParser(description="Classify videos from URLs using OpenAI CLIP")
    parser.add_argument("url", help="URL of the video to classify")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top categories to show")
    parser.add_argument("--output", choices=["json", "pretty"], default="pretty", help="Output format")
    parser.add_argument("--frames", action="store_true", help="Show detailed frame-by-frame analysis")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio processing (faster)")
    
    args = parser.parse_args()
    
    # Initialize classifier
    print("🚀 Initializing AI Video Classifier with Audio Processing...")
    classifier = ReelClassifier()
    
    # Classify the video
    print(f"🎬 Processing: {args.url}")
    enable_audio = not args.no_audio
    if args.no_audio:
        print("⏭️ Audio processing disabled")
    
    result = classifier.classify_reel(args.url, args.top_k, enable_audio=enable_audio)
    
    # Handle errors
    if "error" in result:
        print(f"❌ Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    
    # Output results
    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print("="*60)
        print("🎯 AI VIDEO CLASSIFICATION RESULTS")
        print("="*60)
        print(f"📺 Title: {result['title']}")
        print(f"⏱️  Duration: {result['duration']} seconds")
        print(f"🎯 Overall Confidence: {result['confidence_score']:.3f}")
        print(f"🖼️  Frames Analyzed: {result['total_frames_analyzed']}")
        print(f"🎵 Has Audio: {'Yes' if result['has_audio'] else 'No'}")
        
        print(f"\n🏆 TOP {args.top_k} SPECIFIC CATEGORIES:")
        print("-" * 40)
        for i, (category, score) in enumerate(result['top_categories'].items(), 1):
            print(f"{i:2d}. {category:<35} ({score:.3f})")
        
        print(f"\n🌐 BROADER CATEGORY GROUPS:")
        print("-" * 40)
        for i, (category, score) in enumerate(result['broader_categories'].items(), 1):
            print(f"{i:2d}. {category.replace('_', ' ').title():<35} ({score:.3f})")
        
        print(f"\n🔖 VISUAL TAGS:")
        print("-" * 20)
        print(", ".join(result['visual_tags']))
        
        # Show audio analysis if available
        if result['audio_keywords']:
            print(f"\n🎙️ AUDIO KEYWORDS:")
            print("-" * 20)
            print(", ".join(result['audio_keywords']))
        
        if result['transcript']:
            print(f"\n📝 TRANSCRIPT:")
            print("-" * 20)
            transcript_preview = result['transcript'][:300] + "..." if len(result['transcript']) > 300 else result['transcript']
            print(f"{transcript_preview}")
        
        print(f"\n🏷️ ALL COMBINED TAGS:")
        print("-" * 20)
        print(", ".join(result['semantic_tags']))
        
        # Show frame-by-frame analysis if requested
        if args.frames and 'frame_by_frame' in result:
            print(f"\n🎞️  DETAILED FRAME ANALYSIS:")
            print("-" * 60)
            for frame_data in result['frame_by_frame']:
                print(f"Frame {frame_data['frame_number']:2d}: {frame_data['dominant_category']:<30} ({frame_data['confidence']:.3f})")
                top_3 = frame_data['top_categories']
                categories_str = " | ".join([f"{cat}: {score:.3f}" for cat, score in list(top_3.items())[:3]])
                print(f"         Top 3: {categories_str}")
                print()

if __name__ == "__main__":
    main()
