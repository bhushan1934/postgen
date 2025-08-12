import json
import os
import sys
from datetime import datetime
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
import requests

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/branding.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

branding = config["branding"]
themes = config["legal_themes"]

NEWS_API_KEY = "14ece35abe6e4f83af1ba1fe6e645490"
NEWS_SAVE_PATH = os.path.join(os.path.dirname(__file__), "../assets/latest_news.jpg")


def fetch_latest_news():
    """Fetch top trending news in India using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q=India&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"

    resp = requests.get(url)
    data = resp.json()

    if "articles" not in data or len(data["articles"]) == 0:
        raise ValueError("No articles found from NewsAPI")

    # Get first article with an image
    for article in data["articles"]:
        if article.get("urlToImage") and article.get("title"):
            headline = article["title"]
            img_url = article["urlToImage"]
            # Save image locally
            img_data = requests.get(img_url).content
            with open(NEWS_SAVE_PATH, "wb") as f:
                f.write(img_data)
            return headline, NEWS_SAVE_PATH

    raise ValueError("No valid article with image found.")


def match_theme(headline):
    for theme_name, theme_data in themes.items():
        if any(kw.lower() in headline.lower() for kw in theme_data["keywords"]):
            return theme_name, theme_data
    return "constitutional", themes["constitutional"]


def create_post(headline, image_path):
    theme_name, theme_data = match_theme(headline)
    template_style = list(branding["design_templates"].keys())[0]  # first template for now
    colors = branding["design_templates"][template_style]["colors"]
    text_color = branding["design_templates"][template_style]["text_color"]

    today_str = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(os.path.dirname(__file__), f"../output/{today_str}_{theme_name}.jpg")

    # Create gradient background
    with Image(width=1080, height=1080, background=Color(colors[0])) as base:
        with Image(width=1080, height=1080, background=Color(colors[1])) as overlay:
            overlay.alpha_channel = 'activate'
            base.composite_channel('all', overlay, 'blend', 0, 0)

        # Insert news image top is 120
        with Image(filename=image_path) as news_img:
            news_img.resize(1080, 760)
            base.composite(news_img, left=0, top=0)

        # Draw transparent bar for headline
        with Drawing() as draw:
            draw.fill_color = Color('rgba(0,0,0,0.45)')
            draw.rectangle(left=0, top=880, right=1080, bottom=1080)
            draw(base)

        # Add headline text
        with Drawing() as draw:
            draw.font = branding["font_primary"]
            draw.font_size = 36
            draw.fill_color = Color(text_color)
            draw.gravity = 'south'
            draw.text(0, 40, headline)
            draw(base)

        base.save(filename=output_path)
        print(f"✅ Post created: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments → auto fetch latest news
        headline, image_path = fetch_latest_news()
        print(f"📰 Using headline: {headline}")
        create_post(headline, image_path)
    elif len(sys.argv) == 3:
        # Manual mode
        create_post(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python generate_post.py              # auto fetch latest news")
        print("  python generate_post.py '<headline>' <image_path>  # manual mode")
