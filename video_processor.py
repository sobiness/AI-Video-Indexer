#!/usr/bin/env python3
"""
Video processing module for content classification
Handles video download, frame extraction, and CLIP-based classification
"""

import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
import yt_dlp
import requests
import tempfile
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video processing and visual classification using CLIP model"""
    
    def __init__(self):
        """Initialize video processor with CLIP model and categories"""
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load categories and hierarchy
        self.categories = self._load_categories()
        self.category_hierarchy = self._load_category_hierarchy()
        
        # Precompute text embeddings for efficient classification
        self.text_embeddings = self._compute_text_embeddings()
        
        # Configuration for video download
        self.ydl_opts = {
            'format': 'best[height<=720]/mp4',
            'quiet': True,
            'no_warnings': True,
        }
        
        logger.info(f"✅ VideoProcessor initialized with {len(self.categories)} categories")
    
    def _load_categories(self) -> List[str]:
        """Load the comprehensive list of video categories"""
        return [
            # Food & Cooking (18 categories)
            "step by step cooking recipe", "quick meal preparation", "dessert baking tutorial", 
            "bread and pastry making", "cake decorating techniques", "food taste test reaction",
            "restaurant meal review", "street food exploration", "cooking tips and hacks",
            "kitchen appliance demonstration", "knife skills tutorial", "meal prep planning",
            "international cuisine cooking", "vegan recipe preparation", "grilling and barbecue",
            "cocktail and drink mixing", "food science explanation", "professional chef techniques",
            
            # Beauty & Cosmetics (15 categories)
            "full face makeup transformation", "eyeshadow blending tutorial", "contouring and highlighting",
            "skincare routine demonstration", "acne treatment tips", "anti aging skincare",
            "nail art design tutorial", "gel nail application", "hair curling techniques",
            "hair braiding styles", "hair coloring process", "beauty product testing",
            "makeup brand comparison", "skincare ingredient analysis", "special effects makeup",
            
            # Fashion & Style (12 categories)
            "clothing haul and try on", "outfit styling ideas", "thrift shopping finds",
            "designer fashion review", "seasonal wardrobe planning", "accessory styling tips",
            "shoe collection showcase", "fashion trend analysis", "vintage clothing styling",
            "sustainable fashion tips", "fashion week highlights", "personal style evolution",
            
            # Fitness & Exercise (18 categories)
            "high intensity interval training", "strength training workout", "cardio exercise routine",
            "yoga flow sequence", "pilates core workout", "stretching and flexibility",
            "dance fitness choreography", "martial arts training", "swimming technique tutorial",
            "running form correction", "weightlifting technique", "bodyweight exercises",
            "outdoor fitness activity", "sports specific training", "rehabilitation exercises",
            "fitness transformation journey", "workout equipment review", "nutrition for athletes",
            
            # Health & Wellness (12 categories)
            "mental health awareness", "meditation and mindfulness", "stress relief techniques",
            "sleep hygiene tips", "healthy eating habits", "weight loss journey",
            "muscle building nutrition", "supplement review", "disease prevention tips",
            "physical therapy exercises", "wellness routine planning", "health myth debunking",
            
            # Entertainment & Comedy (15 categories)
            "comedy sketch performance", "stand up comedy routine", "funny prank video",
            "viral challenge attempt", "reaction video content", "meme recreation",
            "impressions and mimicry", "comedic storytelling", "blooper and fail compilation",
            "satirical commentary", "comedy duo performance", "improvisation acting",
            "funny animal moments", "kids saying funny things", "everyday humor situations",
            
            # Music & Audio (12 categories)
            "original song performance", "cover song rendition", "music instrument tutorial",
            "singing technique lesson", "beatboxing demonstration", "music production process",
            "album reaction and review", "music theory explanation", "band rehearsal footage",
            "acoustic performance", "electronic music creation", "music collaboration",
            
            # Dance & Movement (12 categories)
            "hip hop dance routine", "ballet technique demonstration", "contemporary dance performance",
            "cultural traditional dance", "couple dance tutorial", "dance battle competition",
            "choreography breakdown", "dance fitness workout", "flexibility training for dancers",
            "dance improvisation", "group dance synchronization", "dance costume showcase",
            
            # Arts & Crafts (15 categories)
            "painting technique tutorial", "digital art creation process", "sculpture making",
            "pottery and ceramics", "jewelry making tutorial", "paper craft project",
            "embroidery and sewing", "woodworking project", "calligraphy and lettering",
            "photography tips and tricks", "photo editing tutorial", "art supply review",
            "creative drawing techniques", "mixed media art", "art history explanation",
            
            # Technology & Gadgets (12 categories)
            "smartphone review and comparison", "laptop performance testing", "gaming setup showcase",
            "app tutorial and tips", "software demonstration", "tech news commentary",
            "gadget unboxing experience", "troubleshooting technical issues", "coding tutorial",
            "artificial intelligence explanation", "cryptocurrency discussion", "future technology prediction",
            
            # Education & Learning (15 categories)
            "academic subject explanation", "language learning lesson", "study tips and techniques",
            "exam preparation strategy", "book review and summary", "historical fact presentation",
            "science experiment demonstration", "mathematics problem solving", "geography exploration",
            "cultural education content", "skill development tutorial", "career advice guidance",
            "online learning tips", "research methodology", "critical thinking exercises",
            
            # Travel & Adventure (12 categories)
            "destination travel guide", "budget travel tips", "solo travel experience",
            "cultural immersion journey", "adventure sports activity", "camping and hiking",
            "city exploration walk", "food tourism experience", "transportation review",
            "travel packing tutorial", "photography location scouting", "travel safety advice",
            
            # Lifestyle & Daily Life (18 categories)
            "morning routine optimization", "productive daily habits", "time management tips",
            "home organization system", "minimalist lifestyle", "room decoration ideas",
            "cleaning and maintenance tips", "personal development journey", "goal setting strategy",
            "financial planning advice", "shopping haul experience", "product unboxing",
            "life hack demonstration", "seasonal preparation tips", "self care routine",
            "work from home setup", "relationship communication tips", "parenting advice",
            
            # Sports & Athletics (15 categories)
            "football training drill", "basketball skill development", "tennis technique improvement",
            "swimming stroke tutorial", "cycling performance tips", "marathon training plan",
            "team sport strategy", "individual sport technique", "sports equipment review",
            "athletic nutrition guidance", "injury prevention exercise", "sports psychology tips",
            "competition preparation", "referee rule explanation", "sports history documentary",
            
            # Gaming & Esports (10 categories)
            "video game walkthrough", "gaming strategy guide", "game review and rating",
            "esports tournament highlights", "gaming setup optimization", "speedrun attempt",
            "gaming challenge completion", "multiplayer team coordination", "game development insight", "retro gaming nostalgia",
            
            # Automotive & Vehicles (8 categories)
            "car review and test drive", "vehicle maintenance tutorial", "automotive modification",
            "driving technique instruction", "motorcycle riding tips", "classic car restoration",
            "racing technique analysis", "vehicle comparison test",
            
            # Business & Entrepreneurship (10 categories)
            "startup journey documentation", "business strategy explanation", "marketing campaign analysis",
            "investment advice guidance", "networking tips", "leadership skill development",
            "productivity tool review", "workplace culture discussion", "entrepreneur interview", "side hustle ideas",
            
            # Social Issues & Awareness (8 categories)
            "environmental conservation", "social justice advocacy", "community service project",
            "charity fundraising campaign", "political commentary", "cultural awareness education",
            "human rights discussion", "sustainable living practices",
            
            # Family & Relationships (10 categories)
            "parenting tips and advice", "child development milestone", "family activity ideas",
            "relationship communication", "dating advice guidance", "marriage celebration",
            "pregnancy journey documentation", "grandparent wisdom sharing", "sibling bonding activities", "pet and family interaction",
            
            # Special Events & Celebrations (8 categories)
            "wedding preparation process", "birthday party planning", "holiday tradition celebration",
            "cultural festival participation", "graduation ceremony", "anniversary celebration",
            "religious ceremony", "community event organization"
        ]
    
    def _load_category_hierarchy(self) -> Dict[str, List[str]]:
        """Load hierarchical category structure"""
        return {
            "food_cooking": [
                "step by step cooking recipe", "quick meal preparation", "dessert baking tutorial", 
                "bread and pastry making", "cake decorating techniques", "food taste test reaction",
                "restaurant meal review", "street food exploration", "cooking tips and hacks",
                "kitchen appliance demonstration", "knife skills tutorial", "meal prep planning",
                "international cuisine cooking", "vegan recipe preparation", "grilling and barbecue",
                "cocktail and drink mixing", "food science explanation", "professional chef techniques"
            ],
            "beauty_fashion": [
                "full face makeup transformation", "eyeshadow blending tutorial", "contouring and highlighting",
                "skincare routine demonstration", "acne treatment tips", "anti aging skincare",
                "nail art design tutorial", "gel nail application", "hair curling techniques",
                "hair braiding styles", "hair coloring process", "beauty product testing",
                "makeup brand comparison", "skincare ingredient analysis", "special effects makeup",
                "clothing haul and try on", "outfit styling ideas", "thrift shopping finds",
                "designer fashion review", "seasonal wardrobe planning", "accessory styling tips",
                "shoe collection showcase", "fashion trend analysis", "vintage clothing styling",
                "sustainable fashion tips", "fashion week highlights", "personal style evolution"
            ],
            "fitness_health": [
                "high intensity interval training", "strength training workout", "cardio exercise routine",
                "yoga flow sequence", "pilates core workout", "stretching and flexibility",
                "dance fitness choreography", "martial arts training", "swimming technique tutorial",
                "running form correction", "weightlifting technique", "bodyweight exercises",
                "outdoor fitness activity", "sports specific training", "rehabilitation exercises",
                "fitness transformation journey", "workout equipment review", "nutrition for athletes",
                "mental health awareness", "meditation and mindfulness", "stress relief techniques",
                "sleep hygiene tips", "healthy eating habits", "weight loss journey",
                "muscle building nutrition", "supplement review", "disease prevention tips",
                "physical therapy exercises", "wellness routine planning", "health myth debunking"
            ],
            "entertainment_performance": [
                "comedy sketch performance", "stand up comedy routine", "funny prank video",
                "viral challenge attempt", "reaction video content", "meme recreation",
                "impressions and mimicry", "comedic storytelling", "blooper and fail compilation",
                "satirical commentary", "comedy duo performance", "improvisation acting",
                "funny animal moments", "kids saying funny things", "everyday humor situations",
                "original song performance", "cover song rendition", "music instrument tutorial",
                "singing technique lesson", "beatboxing demonstration", "music production process",
                "album reaction and review", "music theory explanation", "band rehearsal footage",
                "acoustic performance", "electronic music creation", "music collaboration",
                "hip hop dance routine", "ballet technique demonstration", "contemporary dance performance",
                "cultural traditional dance", "couple dance tutorial", "dance battle competition",
                "choreography breakdown", "dance fitness workout", "flexibility training for dancers",
                "dance improvisation", "group dance synchronization", "dance costume showcase"
            ],
            "education_tutorial": [
                "academic subject explanation", "language learning lesson", "study tips and techniques",
                "exam preparation strategy", "book review and summary", "historical fact presentation",
                "science experiment demonstration", "mathematics problem solving", "geography exploration",
                "cultural education content", "skill development tutorial", "career advice guidance",
                "online learning tips", "research methodology", "critical thinking exercises",
                "painting technique tutorial", "digital art creation process", "sculpture making",
                "pottery and ceramics", "jewelry making tutorial", "paper craft project",
                "embroidery and sewing", "woodworking project", "calligraphy and lettering",
                "photography tips and tricks", "photo editing tutorial", "art supply review",
                "creative drawing techniques", "mixed media art", "art history explanation",
                "smartphone review and comparison", "laptop performance testing", "gaming setup showcase",
                "app tutorial and tips", "software demonstration", "tech news commentary",
                "gadget unboxing experience", "troubleshooting technical issues", "coding tutorial",
                "artificial intelligence explanation", "cryptocurrency discussion", "future technology prediction"
            ],
            "lifestyle_daily": [
                "morning routine optimization", "productive daily habits", "time management tips",
                "home organization system", "minimalist lifestyle", "room decoration ideas",
                "cleaning and maintenance tips", "personal development journey", "goal setting strategy",
                "financial planning advice", "shopping haul experience", "product unboxing",
                "life hack demonstration", "seasonal preparation tips", "self care routine",
                "work from home setup", "relationship communication tips", "parenting advice",
                "destination travel guide", "budget travel tips", "solo travel experience",
                "cultural immersion journey", "adventure sports activity", "camping and hiking",
                "city exploration walk", "food tourism experience", "transportation review",
                "travel packing tutorial", "photography location scouting", "travel safety advice"
            ],
            "sports_activities": [
                "football training drill", "basketball skill development", "tennis technique improvement",
                "swimming stroke tutorial", "cycling performance tips", "marathon training plan",
                "team sport strategy", "individual sport technique", "sports equipment review",
                "athletic nutrition guidance", "injury prevention exercise", "sports psychology tips",
                "competition preparation", "referee rule explanation", "sports history documentary",
                "video game walkthrough", "gaming strategy guide", "game review and rating",
                "esports tournament highlights", "gaming setup optimization", "speedrun attempt",
                "gaming challenge completion", "multiplayer team coordination", "game development insight", "retro gaming nostalgia",
                "car review and test drive", "vehicle maintenance tutorial", "automotive modification",
                "driving technique instruction", "motorcycle riding tips", "classic car restoration",
                "racing technique analysis", "vehicle comparison test"
            ],
            "social_culture": [
                "startup journey documentation", "business strategy explanation", "marketing campaign analysis",
                "investment advice guidance", "networking tips", "leadership skill development",
                "productivity tool review", "workplace culture discussion", "entrepreneur interview", "side hustle ideas",
                "environmental conservation", "social justice advocacy", "community service project",
                "charity fundraising campaign", "political commentary", "cultural awareness education",
                "human rights discussion", "sustainable living practices",
                "parenting tips and advice", "child development milestone", "family activity ideas",
                "relationship communication", "dating advice guidance", "marriage celebration",
                "pregnancy journey documentation", "grandparent wisdom sharing", "sibling bonding activities", "pet and family interaction",
                "wedding preparation process", "birthday party planning", "holiday tradition celebration",
                "cultural festival participation", "graduation ceremony", "anniversary celebration",
                "religious ceremony", "community event organization"
            ]
        }
    
    def _compute_text_embeddings(self) -> torch.Tensor:
        """Precompute text embeddings for categories"""
        text_inputs = clip.tokenize(self.categories).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings
    
    def get_instagram_video_url(self, instagram_url: str) -> Optional[str]:
        """Get Instagram video download URL using local API endpoint"""
        try:
            api_url = f"http://localhost:3000/api/video"
            params = {'postUrl': instagram_url}
            
            logger.info(f"🔗 Getting Instagram video URL via API...")
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success' and 'data' in data:
                video_url = data['data'].get('videoUrl')
                if video_url:
                    logger.info(f"✅ Got Instagram video URL successfully")
                    return video_url
                    
            logger.error(f"❌ API response missing video URL: {data}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            logger.error("💡 Make sure your local API server is running on localhost:3000")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting Instagram URL: {e}")
            return None
    
    def get_video_info(self, url: str) -> Optional[Dict]:
        """Extract basic video information"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown')
                }
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            return None
    
    def download_video_temp(self, url: str) -> Optional[str]:
        """Download video to temporary file with Instagram API optimization"""
        try:
            temp_dir = tempfile.mkdtemp()
            
            # Handle Instagram URLs using local API
            if 'instagram.com' in url:
                logger.info(f"📱 Processing Instagram URL via API...")
                video_url = self.get_instagram_video_url(url)
                if not video_url:
                    return None
                
                # Download the direct video URL
                temp_path = os.path.join(temp_dir, "instagram_video.mp4")
                logger.info(f"📥 Downloading Instagram video...")
                
                response = requests.get(video_url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"✅ Instagram video downloaded successfully")
                return temp_path
            
            else:
                # Use yt-dlp for other platforms (YouTube, TikTok, etc.)
                temp_path = os.path.join(temp_dir, "video.%(ext)s")
                
                ydl_opts = {
                    **self.ydl_opts,
                    'outtmpl': temp_path,
                    'format': 'mp4/best[ext=mp4]/best',
                }
                
                logger.info(f"📥 Downloading from: {url}")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Find downloaded file
                for file in os.listdir(temp_dir):
                    if file.startswith("video."):
                        return os.path.join(temp_dir, file)
                
                return None
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            if 'instagram.com' in url:
                logger.error("💡 Make sure your local API server is running on localhost:3000")
            return None
    
    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            # Sample frames uniformly across the video
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def classify_single_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Classify a single frame using CLIP model"""
        try:
            pil_image = Image.fromarray(frame)
            processed_frame = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_embedding = self.model.encode_image(processed_frame)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                
                # Compute similarities with categories
                similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)
                probabilities = torch.softmax(similarities, dim=0)
            
            # Create results dictionary
            results = {}
            for i, category in enumerate(self.categories):
                results[category] = probabilities[i].item()
            
            return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error classifying frame: {e}")
            return {}

    def classify_frames(self, frames: List[np.ndarray]) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float]]:
        """Classify frames using CLIP model"""
        if not frames:
            return {}, [], {}
        
        try:
            frame_classifications = []
            all_embeddings = []
            
            for i, frame in enumerate(frames):
                logger.info(f"🔍 Analyzing frame {i+1}/{len(frames)}")
                
                # Classify individual frame
                frame_result = self.classify_single_frame(frame)
                frame_classifications.append(frame_result)
                
                # Collect embeddings for aggregate analysis
                pil_image = Image.fromarray(frame)
                processed_frame = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.model.encode_image(processed_frame)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    all_embeddings.append(embedding)
            
            # Aggregate classification (average of all frames)
            with torch.no_grad():
                avg_embedding = torch.cat(all_embeddings, dim=0).mean(dim=0, keepdim=True)
                similarities = (avg_embedding @ self.text_embeddings.T).squeeze(0)
                probabilities = torch.softmax(similarities, dim=0)
            
            # Create aggregate results
            aggregate_results = {}
            for i, category in enumerate(self.categories):
                aggregate_results[category] = probabilities[i].item()
            
            aggregate_results = dict(sorted(aggregate_results.items(), key=lambda x: x[1], reverse=True))
            
            # Generate broader category scores
            broader_categories = self.get_broader_categories(aggregate_results)
            
            return aggregate_results, frame_classifications, broader_categories
            
        except Exception as e:
            logger.error(f"Error classifying frames: {e}")
            return {}, [], {}
    
    def generate_semantic_tags(self, top_categories: Dict[str, float]) -> List[str]:
        """Generate semantic tags from top categories"""
        tags = []
        
        # For very low confidence scores, be more generous with tag extraction
        # Take top 10 categories regardless of score
        top_items = list(top_categories.items())[:10]
        
        for category, score in top_items:
            # Split category name and extract meaningful words
            words = category.replace(" and ", " ").replace("_", " ").split()
            # Filter and clean words
            filtered_words = []
            for word in words:
                # Skip common words and short words
                if len(word) > 2 and word.lower() not in {"and", "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by"}:
                    filtered_words.append(word.lower())
            tags.extend(filtered_words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        # Add contextual tags based on top categories
        if top_categories:
            top_category = list(top_categories.keys())[0]
            
            # Financial/Business content
            if any(word in top_category for word in ["investment", "advice", "guidance", "entrepreneur", "business"]):
                unique_tags.extend(["finance", "business", "investment", "money"])
            # Unboxing/Product content  
            elif "unboxing" in top_category:
                unique_tags.extend(["unboxing", "product", "review"])
            # Challenge content
            elif "challenge" in top_category:
                unique_tags.extend(["challenge", "viral", "trending"])
            # Tutorial content
            elif "tutorial" in top_category:
                unique_tags.extend(["tutorial", "education", "howto"])
            # Interview content
            elif "interview" in top_category:
                unique_tags.extend(["interview", "conversation", "discussion"])
        
        # Remove duplicates and limit
        final_tags = list(dict.fromkeys(unique_tags))  # Preserves order while removing duplicates
        return final_tags[:15]
    
    def get_broader_categories(self, specific_categories: Dict[str, float]) -> Dict[str, float]:
        """Map specific categories to broader parent categories"""
        broader_scores = {}
        
        for broad_cat, specific_list in self.category_hierarchy.items():
            total_score = 0
            count = 0
            
            for specific_cat, score in specific_categories.items():
                if specific_cat in specific_list:
                    total_score += score
                    count += 1
            
            if count > 0:
                broader_scores[broad_cat] = total_score / count
        
        return dict(sorted(broader_scores.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    # Test video processor initialization
    processor = VideoProcessor()
    print(f"✅ Video processor initialized with {len(processor.categories)} categories")
