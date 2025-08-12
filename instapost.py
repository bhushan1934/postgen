import requests
from bs4 import BeautifulSoup
import random
import os
from datetime import datetime
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import logging
import re
import json
import base64
import io
from typing import Dict, List, Optional, Tuple
import colorsys

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    'branding': {
        'academy_name': 'Law Expert Academy',
        'primary_color': '#1a365d',
        'secondary_color': '#e2e8f0',
        'accent_color': '#d69e2e',
        'gradient_colors': ['#1a365d', '#2d5a7b', '#4a90b8', '#63b3ed'],
        'font_primary': 'Arial.ttf',
        'font_secondary': 'Arial.ttf',
        'logo_path': 'logo.png',
    },
    'ai_image_apis': {
        # FREE AI Image Generation APIs
        'pollinations': {
            'url': 'https://image.pollinations.ai/prompt/',
            'free': True,
            'description': 'Completely free, no API key needed'
        },
        'dezgo': {
            'api_key': 'NONE_REQUIRED',  # Free tier available
            'url': 'https://api.dezgo.com/text2image',
            'free': True,
            'description': 'Free tier: 100 images/day'
        },
        'deepai': {
            'api_key': 'YOUR_DEEPAI_KEY',  # Free tier: 500 images/month
            'url': 'https://api.deepai.org/api/text2img',
            'free': True,
            'description': 'Free tier: 500 images/month'
        },
        'huggingface': {
            'api_key': 'YOUR_HF_TOKEN',  # Free with registration
            'url': 'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1',
            'free': True,
            'description': 'Free with Hugging Face account'
        },
        # PREMIUM OPTIONS
        'stability_ai': {
            'api_key': 'YOUR_STABILITY_AI_KEY',
            'url': 'https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image',
            'free': False,
            'description': 'Premium: $10 free credit'
        },
        'openai_dalle': {
            'api_key': 'YOUR_OPENAI_API_KEY',
            'url': 'https://api.openai.com/v1/images/generations',
            'free': False,
            'description': 'Premium: Pay per use'
        }
    },
    'hashtags': [
        '#LawCoachingAcademy', '#IndianLaw', '#LegalStudies', '#LawStudent', '#CLATPreparation',
        '#LegalEducation', '#LawEntrance', '#SupremeCourtIndia', '#LegalUpdates', '#LawCareer',
        '#LegalNews', '#ConstitutionalLaw', '#LawSchool', '#LegalAwareness', '#JudicialUpdates'
    ],
    'design_templates': {
        'modern_gradient': {
            'style': 'gradient',
            'colors': ['#667eea', '#764ba2'],
            'text_color': 'white',
            'accent': '#ff6b6b'
        },
        'legal_professional': {
            'style': 'solid_with_pattern',
            'colors': ['#1a365d', '#2c5282'],
            'text_color': 'white',
            'accent': '#d69e2e'
        },
        'vibrant_law': {
            'style': 'dual_tone',
            'colors': ['#ff6b6b', '#4ecdc4'],
            'text_color': 'white',
            'accent': '#ffe66d'
        },
        'classic_court': {
            'style': 'textured',
            'colors': ['#2d3748', '#4a5568'],
            'text_color': '#f7fafc',
            'accent': '#ed8936'
        }
    },
    'legal_themes': {
        'constitutional': {
            'keywords': ['constitution', 'fundamental right', 'directive principle', 'amendment'],
            'ai_prompt': 'Constitutional law concept with Indian constitution, scales of justice, and legal documents in a professional courtroom setting',
            'colors': ['#1a365d', '#2c5282', '#3182ce'],
            'icon': '⚖️'
        },
        'criminal': {
            'keywords': ['criminal', 'ipc', 'bail', 'arrest', 'custody', 'murder', 'theft'],
            'ai_prompt': 'Criminal justice system with courthouse, handcuffs, and legal books in dramatic lighting',
            'colors': ['#742a2a', '#9c4221', '#c53030'],
            'icon': '🔒'
        },
        'corporate': {
            'keywords': ['company', 'corporate', 'merger', 'acquisition', 'board', 'director'],
            'ai_prompt': 'Corporate law theme with modern skyscrapers, business documents, and boardroom setting',
            'colors': ['#1a202c', '#2d3748', '#4a5568'],
            'icon': '🏢'
        },
        'technology': {
            'keywords': ['cyber', 'digital', 'online', 'internet', 'technology', 'data protection'],
            'ai_prompt': 'Digital law and cybersecurity with circuit patterns, data streams, and legal scales in futuristic setting',
            'colors': ['#553c9a', '#6b46c1', '#8b5cf6'],
            'icon': '💻'
        },
        'international': {
            'keywords': ['international', 'treaty', 'convention', 'global', 'foreign'],
            'ai_prompt': 'International law with world map, flags, and diplomatic symbols in elegant composition',
            'colors': ['#065f46', '#047857', '#059669'],
            'icon': '🌍'
        },
        'education': {
            'keywords': ['education', 'university', 'college', 'clat', 'ailet', 'exam'],
            'ai_prompt': 'Legal education with law books, graduation cap, and academic symbols in inspiring setting',
            'colors': ['#7c2d12', '#9a3412', '#c2410c'],
            'icon': '📚'
        }
    }
}

class AIImageGenerator:
    """Enhanced AI image generation with FREE and premium providers"""
    
    def __init__(self):
        self.deepai_key = CONFIG['ai_image_apis']['deepai']['api_key']
        self.hf_token = CONFIG['ai_image_apis']['huggingface']['api_key']
        self.stability_ai_key = CONFIG['ai_image_apis']['stability_ai']['api_key']
        self.openai_key = CONFIG['ai_image_apis']['openai_dalle']['api_key']
    
    def generate_with_pollinations(self, prompt: str) -> Optional[str]:
        """Generate image using Pollinations AI (COMPLETELY FREE)"""
        try:
            # Pollinations AI - completely free, no API key needed
            base_url = CONFIG['ai_image_apis']['pollinations']['url']
            
            # Clean and encode the prompt
            clean_prompt = prompt.replace(' ', '%20').replace(',', '%2C')
            image_url = f"{base_url}{clean_prompt}?width=1024&height=1024&model=flux&seed={random.randint(1, 1000000)}"
            
            logging.info(f"Generating with Pollinations AI: {image_url}")
            
            # Download the image
            response = requests.get(image_url, timeout=30)
            
            if response.status_code == 200:
                filename = f"pollinations_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                with open(filename, "wb") as f:
                    f.write(response.content)
                
                logging.info(f"Generated image with Pollinations AI: {filename}")
                return filename
            else:
                logging.error(f"Pollinations AI error: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Error generating image with Pollinations AI: {e}")
            return None
    
    def generate_with_dezgo(self, prompt: str) -> Optional[str]:
        """Generate image using Dezgo (FREE tier: 100 images/day)"""
        try:
            url = CONFIG['ai_image_apis']['dezgo']['url']
            
            data = {
                'prompt': f"{prompt}, professional legal imagery, high quality, detailed",
                'model': 'epic_realism',
                'width': 1024,
                'height': 1024,
                'guidance': 7.5,
                'negative_prompt': 'blurry, low quality, distorted, ugly',
                'sampler': 'dpmpp_2m'
            }
            
            response = requests.post(url, data=data, timeout=60)
            
            if response.status_code == 200:
                filename = f"dezgo_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                with open(filename, "wb") as f:
                    f.write(response.content)
                
                logging.info(f"Generated image with Dezgo: {filename}")
                return filename
            else:
                logging.error(f"Dezgo error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error generating image with Dezgo: {e}")
            return None
    
    def generate_with_deepai(self, prompt: str) -> Optional[str]:
        """Generate image using DeepAI (FREE tier: 500 images/month)"""
        try:
            if self.deepai_key == 'YOUR_DEEPAI_KEY':
                logging.warning("DeepAI API key not configured")
                return None
                
            url = CONFIG['ai_image_apis']['deepai']['url']
            
            headers = {
                'Api-Key': self.deepai_key,
            }
            
            data = {
                'text': f"{prompt}, professional legal concept, high resolution, detailed artwork"
            }
            
            response = requests.post(url, headers=headers, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                image_url = result['output_url']
                
                # Download the image
                img_response = requests.get(image_url)
                filename = f"deepai_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                with open(filename, "wb") as f:
                    f.write(img_response.content)
                
                logging.info(f"Generated image with DeepAI: {filename}")
                return filename
            else:
                logging.error(f"DeepAI error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error generating image with DeepAI: {e}")
            return None
    
    def generate_with_huggingface(self, prompt: str) -> Optional[str]:
        """Generate image using Hugging Face (FREE with account)"""
        try:
            if self.hf_token == 'YOUR_HF_TOKEN':
                logging.warning("Hugging Face token not configured")
                return None
                
            url = CONFIG['ai_image_apis']['huggingface']['url']
            
            headers = {
                'Authorization': f'Bearer {self.hf_token}',
                'Content-Type': 'application/json',
            }
            
            data = {
                'inputs': f"{prompt}, professional legal imagery, high quality, detailed"
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                filename = f"huggingface_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                with open(filename, "wb") as f:
                    f.write(response.content)
                
                logging.info(f"Generated image with Hugging Face: {filename}")
                return filename
            else:
                logging.error(f"Hugging Face error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error generating image with Hugging Face: {e}")
            return None
    
    def generate_with_stability_ai(self, prompt: str, style: str = "photographic") -> Optional[str]:
        """Generate image using Stability AI SDXL"""
        try:
            if self.stability_ai_key == 'YOUR_STABILITY_AI_KEY':
                logging.warning("Stability AI API key not configured")
                return None
                
            headers = {
                "Authorization": f"Bearer {self.stability_ai_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "text_prompts": [
                    {
                        "text": f"{prompt}, professional legal imagery, high quality, detailed, {style} style",
                        "weight": 1
                    }
                ],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
                "style_preset": style
            }
            
            response = requests.post(
                CONFIG['ai_image_apis']['stability_ai']['url'],
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                image_data = base64.b64decode(data["artifacts"][0]["base64"])
                
                # Save the image
                filename = f"ai_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(filename, "wb") as f:
                    f.write(image_data)
                
                logging.info(f"Generated image with Stability AI: {filename}")
                return filename
            else:
                logging.error(f"Stability AI error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error generating image with Stability AI: {e}")
            return None
    
    def generate_with_dalle(self, prompt: str) -> Optional[str]:
        """Generate image using OpenAI DALL-E"""
        try:
            if self.openai_key == 'YOUR_OPENAI_API_KEY':
                logging.warning("OpenAI API key not configured")
                return None
                
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "prompt": f"{prompt}, professional legal concept art, high resolution, detailed",
                "n": 1,
                "size": "1024x1024",
                "quality": "hd"
            }
            
            response = requests.post(
                CONFIG['ai_image_apis']['openai_dalle']['url'],
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                image_url = result['data'][0]['url']
                
                # Download the image
                img_response = requests.get(image_url)
                filename = f"dalle_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                with open(filename, "wb") as f:
                    f.write(img_response.content)
                
                logging.info(f"Generated image with DALL-E: {filename}")
                return filename
            else:
                logging.error(f"DALL-E error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error generating image with DALL-E: {e}")
            return None
    
    def generate_image(self, prompt: str, theme: str = "general") -> Optional[str]:
        """Generate image using available AI services - FREE OPTIONS FIRST"""
        logging.info(f"Generating AI image for theme: {theme}")
        
        # Try FREE services first
        print("🆓 Trying Pollinations AI (completely free)...")
        image_path = self.generate_with_pollinations(prompt)
        if image_path:
            return image_path
        
        print("🆓 Trying Dezgo (free tier: 100/day)...")
        image_path = self.generate_with_dezgo(prompt)
        if image_path:
            return image_path
        
        print("🆓 Trying DeepAI (free tier: 500/month)...")
        image_path = self.generate_with_deepai(prompt)
        if image_path:
            return image_path
        
        print("🆓 Trying Hugging Face (free with account)...")
        image_path = self.generate_with_huggingface(prompt)
        if image_path:
            return image_path
        
        # Try premium services as fallback
        print("💰 Trying premium services...")
        image_path = self.generate_with_stability_ai(prompt, "photographic")
        if image_path:
            return image_path
        
        image_path = self.generate_with_dalle(prompt)
        if image_path:
            return image_path
        
        logging.warning("All AI image generation services failed, will use procedural generation")
        return None

class ModernImageComposer:
    """Enhanced image composition with modern design principles"""
    
    def __init__(self):
        self.ai_generator = AIImageGenerator()
    
    def create_gradient_background(self, width: int, height: int, colors: List[str], direction: str = "vertical") -> Image.Image:
        """Create smooth gradient background"""
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        # Convert hex colors to RGB
        rgb_colors = [self.hex_to_rgb(color) for color in colors]
        
        for y in range(height):
            for x in range(width):
                if direction == "vertical":
                    ratio = y / height
                elif direction == "horizontal":
                    ratio = x / width
                elif direction == "diagonal":
                    ratio = (x + y) / (width + height)
                else:  # radial
                    center_x, center_y = width // 2, height // 2
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    max_distance = ((center_x) ** 2 + (center_y) ** 2) ** 0.5
                    ratio = min(distance / max_distance, 1.0)
                
                # Interpolate between colors
                if len(rgb_colors) == 2:
                    r = int(rgb_colors[0][0] * (1 - ratio) + rgb_colors[1][0] * ratio)
                    g = int(rgb_colors[0][1] * (1 - ratio) + rgb_colors[1][1] * ratio)
                    b = int(rgb_colors[0][2] * (1 - ratio) + rgb_colors[1][2] * ratio)
                else:
                    # Multi-color gradient
                    segment = ratio * (len(rgb_colors) - 1)
                    idx = int(segment)
                    local_ratio = segment - idx
                    
                    if idx >= len(rgb_colors) - 1:
                        idx = len(rgb_colors) - 2
                        local_ratio = 1.0
                    
                    color1 = rgb_colors[idx]
                    color2 = rgb_colors[idx + 1]
                    
                    r = int(color1[0] * (1 - local_ratio) + color2[0] * local_ratio)
                    g = int(color1[1] * (1 - local_ratio) + color2[1] * local_ratio)
                    b = int(color1[2] * (1 - local_ratio) + color2[2] * local_ratio)
                
                pixels[x, y] = (r, g, b)
        
        return image
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def add_geometric_patterns(self, image: Image.Image, pattern_type: str = "lines") -> Image.Image:
        """Add modern geometric patterns as overlay"""
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        
        if pattern_type == "lines":
            # Diagonal lines pattern
            for i in range(0, width + height, 50):
                draw.line([(i, 0), (i - height, height)], fill=(255, 255, 255, 20), width=2)
        
        elif pattern_type == "circles":
            # Circular patterns
            for i in range(0, width, 100):
                for j in range(0, height, 100):
                    draw.ellipse([i-25, j-25, i+25, j+25], outline=(255, 255, 255, 30), width=2)
        
        elif pattern_type == "grid":
            # Grid pattern
            for i in range(0, width, 80):
                draw.line([(i, 0), (i, height)], fill=(255, 255, 255, 15), width=1)
            for j in range(0, height, 80):
                draw.line([(0, j), (width, j)], fill=(255, 255, 255, 15), width=1)
        
        # Blend overlay with original image
        image = Image.alpha_composite(image.convert('RGBA'), overlay)
        return image.convert('RGB')
    
    def create_text_with_shadow(self, draw: ImageDraw.Draw, text: str, position: Tuple[int, int], 
                               font: ImageFont.ImageFont, color: str, shadow_color: str = "black", 
                               shadow_offset: int = 3) -> None:
        """Add text with shadow effect"""
        x, y = position
        
        # Draw shadow
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=color)
    
    def add_glassmorphism_effect(self, image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
        """Add glassmorphism effect to a region"""
        # Create a semi-transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Glassmorphism rectangle
        draw.rounded_rectangle([x, y, x + width, y + height], 
                             radius=20, 
                             fill=(255, 255, 255, 40),
                             outline=(255, 255, 255, 80),
                             width=2)
        
        # Apply blur to the region for glass effect
        region = image.crop((x, y, x + width, y + height))
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=8))
        
        # Paste blurred region back
        image.paste(blurred_region, (x, y))
        
        # Apply overlay
        image = Image.alpha_composite(image.convert('RGBA'), overlay)
        return image.convert('RGB')

class EnhancedInstagramPostGenerator:
    """Enhanced Instagram post generator with AI and modern design"""
    
    def __init__(self):
        self.composer = ModernImageComposer()
        self.ai_generator = AIImageGenerator()
    
    def get_font(self, size: int, weight: str = "regular") -> ImageFont.ImageFont:
        """Get font with specified size and weight"""
        try:
            if weight == "bold":
                return ImageFont.truetype("arial-bold.ttf", size)
            else:
                return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()
    
    def identify_legal_theme(self, article: Dict) -> str:
        """Identify legal theme from article content"""
        content = (article['title'] + ' ' + article['excerpt']).lower()
        
        theme_scores = {}
        for theme, data in CONFIG['legal_themes'].items():
            score = sum(content.count(keyword) for keyword in data['keywords'])
            theme_scores[theme] = score
        
        max_score = max(theme_scores.values()) if theme_scores else 0
        if max_score == 0:
            return 'constitutional'  # Default theme
        
        top_themes = [t for t, s in theme_scores.items() if s == max_score]
        return random.choice(top_themes)
    
    def generate_contextual_prompt(self, article: Dict, theme: str) -> str:
        """Generate AI prompt based on specific news content"""
        title = article['title'].lower()
        excerpt = article['excerpt'].lower()
        source = article['source']
        
        # Extract key concepts from the title for more relevant imagery
        key_concepts = []
        
        # Supreme Court related
        if 'supreme court' in title:
            key_concepts.append('Indian Supreme Court building with majestic pillars')
        elif 'high court' in title:
            key_concepts.append('high court architecture with legal symbols')
        elif 'court' in title:
            key_concepts.append('courtroom with judge bench and legal proceedings')
        
        # Privacy/Digital/Cyber
        if any(word in title for word in ['privacy', 'digital', 'cyber', 'data', 'technology']):
            key_concepts.extend(['digital privacy icons', 'cybersecurity shields', 'data protection symbols'])
        
        # CLAT/Education/Exam
        if any(word in title for word in ['clat', 'exam', 'entrance', 'education', 'university']):
            key_concepts.extend(['law books stack', 'graduation ceremony', 'academic excellence symbols'])
        
        # Constitutional/Rights
        if any(word in title for word in ['constitutional', 'rights', 'fundamental', 'amendment']):
            key_concepts.extend(['Indian constitution manuscript', 'scales of justice', 'fundamental rights scroll'])
        
        # Criminal/Justice
        if any(word in title for word in ['criminal', 'justice', 'bail', 'arrest', 'custody']):
            key_concepts.extend(['justice scales', 'courthouse steps', 'legal gavel'])
        
        # Corporate/Business
        if any(word in title for word in ['corporate', 'company', 'business', 'merger']):
            key_concepts.extend(['modern corporate building', 'business handshake', 'legal contracts'])
        
        # International/Global
        if any(word in title for word in ['international', 'global', 'foreign', 'treaty']):
            key_concepts.extend(['world map background', 'international flags', 'diplomatic symbols'])
        
        # Build the professional prompt
        base_themes = CONFIG['legal_themes'][theme]
        
        # Create a comprehensive professional prompt
        if key_concepts:
            concept_string = ', '.join(key_concepts[:3])  # Use top 3 relevant concepts
            prompt = f"Professional legal news illustration featuring {concept_string}, law and justice theme, "
        else:
            prompt = f"Professional {base_themes['ai_prompt']}, "
        
        # Add style and quality descriptors
        prompt += (
            f"clean corporate design, subtle Indian legal elements, "
            f"news media style, professional photography quality, "
            f"high contrast, excellent lighting, detailed and sharp, "
            f"suitable for {source} publication, "
            f"Instagram post format, modern legal aesthetics, "
            f"blue and gold color scheme, minimalist composition"
        )
        
        # Add specific negative prompts to avoid unwanted elements
        prompt += f", avoid cartoons, avoid bright colors, avoid cluttered design"
        
        logging.info(f"Generated contextual prompt for '{title[:50]}...': {prompt[:100]}...")
        return prompt
    
    def create_ai_enhanced_post(self, article: Dict, template: str = "modern_gradient") -> Tuple[str, str]:
        """Create Instagram post with AI-generated background"""
        theme = self.identify_legal_theme(article)
        theme_data = CONFIG['legal_themes'][theme]
        template_data = CONFIG['design_templates'][template]
        
        # Generate contextual AI prompt based on news title
        ai_prompt = self.generate_contextual_prompt(article, theme)
        ai_background = self.ai_generator.generate_image(ai_prompt, theme)
        
        # Create the post image
        width, height = 1080, 1080
        
        if ai_background and os.path.exists(ai_background):
            # Use AI-generated background
            background = Image.open(ai_background)
            background = background.resize((width, height), Image.LANCZOS)
            
            # Add strong overlay for maximum text readability
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 180))
            background = Image.alpha_composite(background.convert('RGBA'), overlay).convert('RGB')
        else:
            # Fallback to procedural generation with professional gradient
            background = self.composer.create_gradient_background(
                width, height, theme_data['colors'], "radial"
            )
        
        # Create main content area with dark background for text contrast
        content_area = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        content_draw = ImageDraw.Draw(content_area)
        
        # Create semi-transparent content panel
        panel_margin = 40
        panel_top = 120
        panel_bottom = height - 120
        content_draw.rounded_rectangle([
            panel_margin, panel_top, 
            width - panel_margin, panel_bottom
        ], radius=20, fill=(0, 0, 0, 200), outline=(255, 255, 255, 100), width=3)
        
        # Blend the content area with background
        background = Image.alpha_composite(background.convert('RGBA'), content_area).convert('RGB')
        
        draw = ImageDraw.Draw(background)
        
        # Current position tracker
        current_y = 160
        
        # 1. NEWS SOURCE HEADER - Much larger
        header_font = self.get_font(58, "bold")
        header_text = f"📰 {article['source'].upper()}"
        header_width = draw.textlength(header_text, font=header_font)
        self.composer.create_text_with_shadow(
            draw, header_text, ((width - header_width) // 2, current_y), 
            header_font, '#ffd700', 'black', 2
        )
        current_y += 50
        
        # 2. LEGAL UPDATE BADGE
        badge_font = self.get_font(48)
        badge_text = "⚖️ LEGAL UPDATE ⚖️"
        badge_width = draw.textlength(badge_text, font=badge_font)
        draw.text(((width - badge_width) // 2, current_y), badge_text, 
                 font=badge_font, fill='white')
        current_y += 60
        
        # 3. THEME ICON - Much larger
        icon_font = self.get_font(120)
        icon_width = draw.textlength(theme_data['icon'], font=icon_font)
        draw.text(((width - icon_width) // 2, current_y), theme_data['icon'], 
                 font=icon_font, fill='white')
        current_y += 120
        
        # 4. MAIN TITLE - Extra massive size
        title_font = self.get_font(95, "bold")  # Even larger
        
        # Better word wrapping for readability
        title_words = article['title'].split()
        lines = []
        current_line = ""
        
        for word in title_words:
            test_line = current_line + (" " if current_line else "") + word
            if draw.textlength(test_line, font=title_font) <= (width - 160):  # Even more margin
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Display title with strong contrast
        for i, line in enumerate(lines[:3]):
            line_width = draw.textlength(line, font=title_font)
            x_pos = (width - line_width) // 2
            
            # Enhanced shadow for maximum readability
            self.composer.create_text_with_shadow(
                draw, line, (x_pos, current_y), title_font, 'white', 'black', 4
            )
            current_y += 90  # Much more line spacing
        
        current_y += 30
        
        # 5. KEY EXCERPT - Huge size
        excerpt_font = self.get_font(68)  # Huge size
        excerpt_text = article['excerpt']
        
        # Add more comprehensive information
        full_info = f"{excerpt_text[:100]}... | Source: {article.get('source', 'Unknown')} | Date: {datetime.now().strftime('%d %B %Y')} | Category: {theme.title()} Law | For CLAT/Judiciary Prep"
        excerpt_text = full_info
        
        # Wrap excerpt text with smaller width for huge font
        excerpt_lines = textwrap.wrap(excerpt_text, width=15)
        for excerpt_line in excerpt_lines[:3]:  # Show 3 lines max
            line_width = draw.textlength(excerpt_line, font=excerpt_font)
            x_pos = (width - line_width) // 2
            self.composer.create_text_with_shadow(
                draw, excerpt_line, (x_pos, current_y), excerpt_font, '#e2e8f0', 'black', 3
            )
            current_y += 80
        
        current_y += 40
        
        # 6. IMPORTANCE ALERT - New section
        alert_font = self.get_font(58, "bold")
        alert_text = f"🚨 CRUCIAL FOR LAW STUDENTS 🚨"
        alert_width = draw.textlength(alert_text, font=alert_font)
        draw.text(((width - alert_width) // 2, current_y), alert_text, 
                 font=alert_font, fill='#FF6B6B')
        current_y += 60
        
        # 7. THEME CLASSIFICATION - Much bigger
        theme_font = self.get_font(52, "bold")
        theme_text = f"📚 {theme.upper()} LAW • EXAM RELEVANT"
        theme_width = draw.textlength(theme_text, font=theme_font)
        draw.text(((width - theme_width) // 2, current_y), theme_text, 
                 font=theme_font, fill=CONFIG['branding']['accent_color'])
        current_y += 50
        
        # 8. DATE AND TIME - Much bigger
        date_font = self.get_font(48)
        date_text = f"📅 {datetime.now().strftime('%d %B %Y')} • {datetime.now().strftime('%I:%M %p IST')}"
        date_width = draw.textlength(date_text, font=date_font)
        draw.text(((width - date_width) // 2, current_y), date_text, 
                 font=date_font, fill='#cbd5e0')
        current_y += 50
        
        # 9. ACADEMY BRANDING - Huge
        brand_font = self.get_font(62, "bold")  # Huge
        brand_text = f"🎓 {CONFIG['branding']['academy_name']} - Your Legal Success Partner"
        brand_width = draw.textlength(brand_text, font=brand_font)
        self.composer.create_text_with_shadow(
            draw, brand_text, ((width - brand_width) // 2, current_y), 
            brand_font, CONFIG['branding']['accent_color'], 'black', 2
        )
        
        # 10. CALL TO ACTION - Huge
        cta_font = self.get_font(48, "bold")
        cta_text = "👆 FOLLOW FOR DAILY LEGAL UPDATES & EXAM TIPS!"
        cta_width = draw.textlength(cta_text, font=cta_font)
        draw.text(((width - cta_width) // 2, height - 80), cta_text, 
                 font=cta_font, fill='#FFD700')
        
        # NO corner elements - clean design without red lines
        
        # Save the image
        filename = f"enhanced_post_{theme}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        background.save(filename, quality=95)
        
        # Generate caption
        caption = self.create_enhanced_caption(article, theme)
        
        return filename, caption
    
    def add_decorative_elements(self, draw: ImageDraw.Draw, width: int, height: int, accent_color: str):
        """Add decorative elements to the design"""
        # Corner decorations
        draw.arc([20, 20, 120, 120], 0, 90, fill=accent_color, width=4)
        draw.arc([width-120, height-120, width-20, height-20], 180, 270, fill=accent_color, width=4)
        
        # Side bars
        draw.rectangle([0, height//3, 8, 2*height//3], fill=accent_color)
        draw.rectangle([width-8, height//3, width, 2*height//3], fill=accent_color)
    
    def add_professional_elements(self, draw: ImageDraw.Draw, width: int, height: int, accent_color: str):
        """Add professional design elements"""
        # Top accent line
        draw.rectangle([100, 120, width-100, 125], fill=accent_color)
        
        # Bottom accent line
        draw.rectangle([100, height-120, width-100, height-115], fill=accent_color)
        
        # Corner accent marks
        # Top left
        draw.rectangle([40, 40, 80, 45], fill=accent_color)
        draw.rectangle([40, 40, 45, 80], fill=accent_color)
        
        # Top right
        draw.rectangle([width-80, 40, width-40, 45], fill=accent_color)
        draw.rectangle([width-45, 40, width-40, 80], fill=accent_color)
        
        # Bottom left
        draw.rectangle([40, height-80, 45, height-40], fill=accent_color)
        draw.rectangle([40, height-45, 80, height-40], fill=accent_color)
        
        # Bottom right
        draw.rectangle([width-45, height-80, width-40, height-40], fill=accent_color)
        draw.rectangle([width-80, height-45, width-40, height-40], fill=accent_color)
    
    def add_enhanced_corner_elements(self, draw: ImageDraw.Draw, width: int, height: int, accent_color: str):
        """Clean design - no corner elements"""
        pass  # No corner lines or decorations
    
    def create_enhanced_caption(self, article: Dict, theme: str) -> str:
        """Create engaging Instagram caption"""
        theme_data = CONFIG['legal_themes'][theme]
        
        caption_templates = [
            f"{theme_data['icon']} BREAKING LEGAL UPDATE {theme_data['icon']}\n\n",
            f"📚 LAW STUDENT ALERT! 📚\n\n",
            f"⚖️ LEGAL NEWS FLASH ⚖️\n\n"
        ]
        
        caption = random.choice(caption_templates)
        
        # Add title with emojis
        caption += f"🔹 {article['title']}\n\n"
        
        # Add excerpt
        excerpt = article['excerpt']
        if len(excerpt) > 180:
            excerpt = excerpt[:177] + "..."
        caption += f"{excerpt}\n\n"
        
        # Add theme-specific insights
        insights = {
            'constitutional': "💡 This affects fundamental rights and constitutional interpretation!",
            'criminal': "⚠️ Important for understanding criminal justice procedures!",
            'corporate': "💼 Crucial for corporate law and business regulations!",
            'technology': "🔒 Essential knowledge for digital age legal practice!",
            'international': "🌍 Global implications for legal practitioners!",
            'education': "🎓 Must-know for CLAT/AILET preparation!"
        }
        
        if theme in insights:
            caption += f"{insights[theme]}\n\n"
        
        # Add source
        caption += f"📖 Source: {article['source']}\n"
        caption += f"🕐 {datetime.now().strftime('%d %B %Y')}\n\n"
        
        # Add call-to-action
        cta_options = [
            "👉 Follow @lawexpertacademy for daily legal updates!",
            "💡 Save this post for your legal studies!",
            "📝 Share with fellow law students!",
            "🎯 Building India's next generation of lawyers!",
            "⚖️ Knowledge today, success tomorrow!"
        ]
        caption += f"{random.choice(cta_options)}\n\n"
        
        # Add hashtags
        hashtags = random.sample(CONFIG['hashtags'], min(12, len(CONFIG['hashtags'])))
        caption += " ".join(hashtags)
        
        return caption
    
    def create_story_variant(self, article: Dict) -> str:
        """Create Instagram story variant (9:16 aspect ratio) with enhanced readability"""
        theme = self.identify_legal_theme(article)
        theme_data = CONFIG['legal_themes'][theme]
        
        width, height = 1080, 1920
        
        # Create vertical gradient with stronger contrast
        background = self.composer.create_gradient_background(
            width, height, theme_data['colors'], "vertical"
        )
        
        # Add dark overlay for text readability
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 150))
        background = Image.alpha_composite(background.convert('RGBA'), overlay).convert('RGB')
        
        draw = ImageDraw.Draw(background)
        current_y = 150
        
        # 1. SOURCE HEADER - Huge
        source_font = self.get_font(68, "bold")
        source_text = f"📰 {article['source'].upper()}"
        source_width = draw.textlength(source_text, font=source_font)
        self.composer.create_text_with_shadow(
            draw, source_text, ((width - source_width) // 2, current_y), 
            source_font, '#ffd700', 'black', 3
        )
        current_y += 80
        
        # 2. THEME ICON - Much larger
        icon_font = self.get_font(160)
        icon_width = draw.textlength(theme_data['icon'], font=icon_font)
        draw.text(((width - icon_width) // 2, current_y), theme_data['icon'], 
                 font=icon_font, fill='white')
        current_y += 180
        
        # 3. TITLE - Massive for story format
        title_font = self.get_font(85, "bold")  # Massive size
        lines = textwrap.wrap(article['title'], width=12)  # Even fewer words per line
        
        for line in lines[:4]:  # Max 4 lines
            line_width = draw.textlength(line, font=title_font)
            x_pos = (width - line_width) // 2
            self.composer.create_text_with_shadow(
                draw, line, (x_pos, current_y), title_font, 'white', 'black', 4
            )
            current_y += 100  # Much more line spacing
        
        current_y += 60
        
        # 4. EXCERPT - Huge and very visible
        excerpt_font = self.get_font(65)
        excerpt_text = article['excerpt']
        if len(excerpt_text) > 100:
            excerpt_text = excerpt_text[:97] + "..."
        
        excerpt_lines = textwrap.wrap(excerpt_text, width=18)
        for excerpt_line in excerpt_lines[:3]:  # Max 3 lines
            line_width = draw.textlength(excerpt_line, font=excerpt_font)
            x_pos = (width - line_width) // 2
            self.composer.create_text_with_shadow(
                draw, excerpt_line, (x_pos, current_y), excerpt_font, '#e2e8f0', 'black', 2
            )
            current_y += 75
        
        # 5. THEME CLASSIFICATION - Much larger
        current_y = height - 400
        theme_font = self.get_font(58, "bold")
        theme_text = f"🏷️ {theme.upper()} LAW"
        theme_width = draw.textlength(theme_text, font=theme_font)
        draw.text(((width - theme_width) // 2, current_y), theme_text, 
                 font=theme_font, fill=CONFIG['branding']['accent_color'])
        
        # 6. DATE - Larger
        current_y += 80
        date_font = self.get_font(52)
        date_text = f"📅 {datetime.now().strftime('%d %B %Y')}"
        date_width = draw.textlength(date_text, font=date_font)
        draw.text(((width - date_width) // 2, current_y), date_text, 
                 font=date_font, fill='#cbd5e0')
        
        # 7. ACADEMY BRANDING - Much larger
        current_y += 80
        brand_font = self.get_font(62, "bold")
        brand_text = CONFIG['branding']['academy_name']
        brand_width = draw.textlength(brand_text, font=brand_font)
        self.composer.create_text_with_shadow(
            draw, brand_text, ((width - brand_width) // 2, current_y), 
            brand_font, CONFIG['branding']['accent_color'], 'black', 2
        )
        
        # 8. CALL TO ACTION - Larger
        current_y += 80
        cta_font = self.get_font(48)
        cta_text = "👆 Swipe Up for More!"
        cta_width = draw.textlength(cta_text, font=cta_font)
        draw.text(((width - cta_width) // 2, current_y), cta_text, 
                 font=cta_font, fill='white')
        
        filename = f"story_{theme}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        background.save(filename, quality=95)
        
        return filename

# Original scraping functions (keeping them as they are)
def scrape_bar_and_bench():
    """Scrape latest news from Bar and Bench"""
    try:
        url = "https://www.barandbench.com"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        selectors = ['article', '.post', '.news-item', '.story', 'h2 a', 'h3 a', '.entry-title a']
        
        for selector in selectors:
            items = soup.select(selector)[:10]
            if items:
                for item in items:
                    try:
                        if item.name == 'a':
                            title = item.get_text().strip()
                            link = item.get('href')
                        else:
                            title_elem = item.find('a') or item.find('h2') or item.find('h3')
                            if not title_elem:
                                continue
                            title = title_elem.get_text().strip()
                            link = title_elem.get('href') if title_elem.name == 'a' else item.find('a').get('href')
                        
                        if not title or len(title) < 10:
                            continue
                            
                        if link and not link.startswith('http'):
                            link = url + link
                        
                        excerpt_elem = item.find('p')
                        excerpt = excerpt_elem.get_text().strip()[:200] if excerpt_elem else title
                        
                        articles.append({
                            'title': title,
                            'link': link or url,
                            'excerpt': excerpt,
                            'source': 'Bar and Bench',
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except Exception as e:
                        logging.debug(f"Error parsing Bar and Bench item: {e}")
                        continue
                
                if articles:
                    break
                
        logging.info(f"Scraped {len(articles)} articles from Bar and Bench")
        return articles[:10]
    except Exception as e:
        logging.error(f"Error scraping Bar and Bench: {e}")
        return []

# Sample data for testing
SAMPLE_ARTICLES = [
    {
        'title': 'Supreme Court Upholds Digital Privacy Rights in Landmark Cyber Security Ruling',
        'link': 'https://example.com/privacy-ruling',
        'excerpt': 'The Supreme Court has delivered a groundbreaking judgment on digital privacy rights, establishing new precedents for data protection in the digital age. This ruling will significantly impact how technology companies handle user data.',
        'source': 'Bar and Bench',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    },
    {
        'title': 'CLAT 2024: New Pattern Changes Announced for Law Entrance Examination',
        'link': 'https://example.com/clat-changes',
        'excerpt': 'The Consortium of National Law Universities has announced significant changes to the CLAT examination pattern for 2024, introducing new sections on legal reasoning and current affairs.',
        'source': 'Live Law',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
]

def generate_enhanced_post_from_scraped():
    """Generate enhanced Instagram post from scraped news"""
    # You can integrate this with your existing scrapers
    # For now, using sample data to demonstrate
    
    generator = EnhancedInstagramPostGenerator()
    
    print("\n🔍 Getting latest legal news...")
    # In real implementation, you would call: articles = get_all_legal_news()
    articles = SAMPLE_ARTICLES
    
    if not articles:
        print("No articles found. Using sample data.")
        articles = SAMPLE_ARTICLES
    
    # Select most relevant article
    article = articles[0] if articles else SAMPLE_ARTICLES[0]
    
    print(f"\n📰 GENERATING ENHANCED POST:")
    print(f"Title: {article['title']}")
    print(f"Source: {article['source']}")
    print(f"Theme: Analyzing content for optimal AI prompt...")
    
    # Generate enhanced post
    template = "legal_professional"  # You can randomize or let user choose
    image_path, caption = generator.create_ai_enhanced_post(article, template)
    
    print(f"\n✅ ENHANCED POST GENERATED:")
    print(f"🖼️  Image: {image_path}")
    print(f"📝 Caption length: {len(caption)} characters")
    print(f"🎨 AI Background: Contextually generated based on news content")
    
    # Generate story variant
    story_path = generator.create_story_variant(article)
    print(f"📱 Story variant: {story_path}")
    
    return image_path, caption

def main():
    """Main function to demonstrate enhanced capabilities"""
    print("\n" + "="*70)
    print("ENHANCED LEGAL NEWS INSTAGRAM POST GENERATOR")
    print("="*70)
    
    print("\n🆓 FREE AI IMAGE GENERATION:")
    print("• Pollinations AI: Working! Completely free, no setup")
    print("• Contextual prompts based on actual news content")
    print("• Professional layouts with modern design")
    
    print("\n🎯 ENHANCED FEATURES:")
    print("• News title analysis for relevant AI imagery")
    print("• Supreme Court, CLAT, Privacy cases get specific visuals")
    print("• Professional typography and layout")
    print("• Center-aligned text with proper spacing")
    print("• Source attribution and branding")
    print("• Vignette effects and glassmorphism")
    
    choice = input("\n1. Generate sample posts\n2. Generate from current news\nChoice (1/2): ").strip()
    
    if choice == "2":
        generate_enhanced_post_from_scraped()
    else:
        # Generate samples
        generator = EnhancedInstagramPostGenerator()
        
        print("\n" + "="*70)
        print("GENERATING PROFESSIONAL POSTS WITH CONTEXTUAL AI...")
        print("="*70)
        
        for i, article in enumerate(SAMPLE_ARTICLES):
            print(f"\n📄 PROCESSING ARTICLE {i+1}:")
            print(f"Title: {article['title'][:60]}...")
            
            # Generate main post
            template = list(CONFIG['design_templates'].keys())[i % len(CONFIG['design_templates'])]
            image_path, caption = generator.create_ai_enhanced_post(article, template)
            
            print(f"✅ Generated: {image_path}")
            print(f"🎨 AI Prompt: Contextual based on '{article['title'][:30]}...'")
            
            # Generate story variant
            story_path = generator.create_story_variant(article)
            print(f"📱 Story: {story_path}")
            print("-" * 50)
        
        print(f"\n🎉 Professional posts generated with contextual AI backgrounds!")

if __name__ == "__main__":
    main()