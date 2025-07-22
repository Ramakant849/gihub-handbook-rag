import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from collections import deque
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = os.getenv("BASE_URL", "https://handbook.gitlab.com/")
OUTPUT_DIR = "data"
MAX_PAGES_TO_SCRAPE = 100 # Limit for testing, remove or increase for full scrape

def save_content(url, content):
    """Saves the extracted content to a text file."""
    # Create a simple filename from the URL path
    path_parts = urlparse(url).path.strip('/').split('/')
    filename = '_'.join(path_parts) if path_parts[0] else "index"
    # Ensure filename is not too long or has invalid characters
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-')).strip()
    if not filename:
        filename = "index"
    
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    
    # Append a number if the file already exists
    counter = 1
    original_filepath = filepath
    while os.path.exists(filepath):
        filepath = f"{original_filepath.rsplit('.', 1)[0]}_{counter}.txt"
        counter += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved content from {url} to {filepath}")

def extract_text(soup):
    """Extracts main textual content, excluding navigation, headers, footers, and code blocks."""
    # Remove unwanted elements (adjust selectors based on GitLab Handbook structure)
    for element in soup(["nav", "header", "footer", "aside", "style", "script", "form", "img", "link", "meta", "svg"]):
        element.decompose()
    
    # More specific removal for common handbook elements (e.g., table of contents, sidebars)
    for selector in [
        ".handbook-toc", 
        ".js-vue-sidebar", 
        ".feedback",
        ".page-navigation",
        "div.by-author", # often contains non-content info
        "div.content-header", # page title might be redundant with file name
        "div.pagination", # page navigation
        "ul.accordion-nav", # navigation menus
        "div.header-hero", # hero sections
        "div.copy-link", # copy link buttons
        "div.c-contribute-banner", # contribution banners
        "div.js-search", # search bars
        "div.t-tag-list", # tag lists
        "ul.tags", # tags
        "a.btn", # buttons
        "div.alert", # alerts
        "div.grid-layout-section", # layout elements
        "div.overflow-auto", # often contains code blocks or tables that are not pure text
        "div.tooltip", # tooltips
        "div.dropdown", # dropdowns
        "input", "textarea", "select", # form elements
        "button", # buttons
        "[class*=\"navbar\"]", # anything with navbar in class
        "[class*=\"sidebar\"]", # anything with sidebar in class
        "[class*=\"footer\"]", # anything with footer in class
        "[class*=\"header\"]", # anything with header in class (be careful, might remove main content header)
        "[class*=\"breadcrumb\"]", # breadcrumbs
        "[class*=\"social\"]", # social media links
        "[data-qa-selector]", # QA selectors (often for interactive elements)
        "[role=\"navigation\"]", # navigation roles
        "[aria-hidden=\"true\"]", # hidden elements
        "figcaption", # image captions
        "figure", # figures
        "iframe", # embedded content
        "noscript", # noscript tags
        "code", "pre", # code blocks
        ".d-none", # display none elements
        ".gl-display-none", # gitlab specific display none
        ".md-toc", # markdown table of contents
    ]:
        for el in soup.select(selector):
            el.decompose()

    # Get text and clean up extra whitespace
    text = soup.get_text(separator=' ', strip=True)
    return text

def scrape_and_find_links(url):
    """Scrapes a single page and returns its content and a list of internal links."""
    print(f"Attempting to scrape: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content = extract_text(soup)
        links = []
        for a_tag in soup.find_all('a', href=True):
            link = urljoin(url, a_tag['href'])
            # Only follow links within the same domain and starting with BASE_URL
            if urlparse(link).netloc == urlparse(BASE_URL).netloc and link.startswith(BASE_URL):
                links.append(link)
        return content, links
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None, []

def main():
    # Create the data directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Queue for URLs to visit and set for visited URLs
    queue = deque([BASE_URL])
    visited = set()
    scraped_count = 0

    while queue and scraped_count < MAX_PAGES_TO_SCRAPE:
        current_url = queue.popleft()

        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"Scraping ({scraped_count + 1}/{MAX_PAGES_TO_SCRAPE}): {current_url}")
        content, new_links = scrape_and_find_links(current_url)

        if content:
            save_content(current_url, content)
            scraped_count += 1
            for link in new_links:
                if link not in visited:
                    queue.append(link)

    print(f"Scraping finished. Scraped {scraped_count} pages.")

if __name__ == "__main__":
    main() 