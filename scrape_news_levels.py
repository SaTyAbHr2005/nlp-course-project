import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import os

OUTPUT_FILE = 'scraped_simplification_pairs.csv'
MAX_PAGES_TO_SCRAPE = 100  # Scrape 100 pages for a solid dataset

seen_slugs = set()

def clean_text(text):
    # Remove random ads/UI text that might get caught in paragraphs
    text = re.sub(r'Difficult words:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Watch the video.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # News in Levels keeps article text inside div id="nContent"
        content = soup.find('div', id='nContent')
        if not content:
            return None
            
        paragraphs = content.find_all('p')
        
        # Combine paragraphs, ignoring empty ones or video placeholders
        text_chunks = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and not p.find('iframe'):
                text_chunks.append(text)
                
        full_text = " ".join(text_chunks)
        return clean_text(full_text)
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    print("======================================================")
    print(" ESL News Scraper (Level 3 -> Level 1)                ")
    print(" Source: News in Levels                               ")
    print("======================================================")
    
    # Initialize CSV
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['complex_text', 'simple_text'])
            
    total_pairs = 0
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for page_num in range(1, MAX_PAGES_TO_SCRAPE + 1):
        print(f"\n--- Scraping Page {page_num} ---")
        page_url = f"https://www.newsinlevels.com/page/{page_num}/"
        
        try:
            page_resp = requests.get(page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(page_resp.text, 'html.parser')
            
            # Find all article titles on the index page
            article_links = soup.find_all('div', class_='title')
            
            for link_div in article_links:
                a_tag = link_div.find('a')
                if not a_tag: continue
                    
                article_url = a_tag['href']
                
                # We need to extract the base name to guess Level 1 and Level 3 URLs
                # Example: https://www.newsinlevels.com/products/dog-in-the-sea-level-1/
                match = re.search(r'/products/(.+)-level-\d/?', article_url)
                if not match: continue
                    
                base_slug = match.group(1)
                
                if base_slug in seen_slugs:
                    continue
                seen_slugs.add(base_slug)
                
                level_3_url = f"https://www.newsinlevels.com/products/{base_slug}-level-3/"
                level_1_url = f"https://www.newsinlevels.com/products/{base_slug}-level-1/"
                
                # Fetch both levels
                time.sleep(1) # Be polite
                complex_text = get_article_text(level_3_url)
                time.sleep(1)
                simple_text = get_article_text(level_1_url)
                
                # If both successfully fetched, we have a perfect parallel pair!
                if complex_text and simple_text and len(complex_text) > 20 and len(simple_text) > 20:
                    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([complex_text, simple_text])
                        total_pairs += 1
                        
                    print(f"[SUCCESS] Scraped pair: {base_slug.replace('-', ' ')} (Total: {total_pairs})")
                else:
                    print(f"[SKIP] Missing levels for: {base_slug.replace('-', ' ')}")
                    
        except Exception as e:
            print(f"Error scraping page {page_num}: {e}")
            
    print(f"\nFinished! Total perfect pairs scraped: {total_pairs}")

if __name__ == "__main__":
    main()
