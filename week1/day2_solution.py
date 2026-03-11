# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=1.0.0",
#     "requests>=2.32.0",
#     "beautifulsoup4>=4.14.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Day 2 Homework Solution:
Upgrade the Day 1 website summarizer to use a local Ollama model
instead of the OpenAI API.

Usage:
    uv run week1/day2_solution.py [URL]

Examples:
    uv run week1/day2_solution.py
    uv run week1/day2_solution.py https://edwarddonner.com
    uv run week1/day2_solution.py https://cnn.com

Benefits of using Ollama (local model):
    - No API charges
    - Data doesn't leave your machine

Requirements:
    - Ollama must be running locally (http://localhost:11434)
    - Run: ollama pull llama3.2
    - Then start: ollama serve  (if not already running)
"""

import sys
import requests
from openai import OpenAI
from bs4 import BeautifulSoup

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"  # Change to "llama3.2:1b" for a lighter model, or "deepseek-r1:1.5b"
DEFAULT_URL = "https://edwarddonner.com"

# --- Scraper (same logic as scraper.py from day 1) ---
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}


def fetch_website_contents(url: str) -> str:
    """Fetch and return the title + text content of a webpage, truncated to 2000 chars."""
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:2_000]


# --- Prompts ---
SYSTEM_PROMPT = """
You are a helpful assistant that analyzes the contents of a website,
and provides a short, useful summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

USER_PROMPT_PREFIX = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.
"""


def build_messages(website_content: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_PREFIX + website_content},
    ]


# --- Main summarization logic ---
def summarize(url: str) -> str:
    """Fetch website content and summarize it using a local Ollama model."""
    print(f"Fetching: {url}")
    website_content = fetch_website_contents(url)

    # Connect to Ollama via its OpenAI-compatible endpoint
    ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    print(f"Summarizing with local model: {MODEL} ...")
    response = ollama_client.chat.completions.create(
        model=MODEL,
        messages=build_messages(website_content),
    )
    return response.choices[0].message.content


def check_ollama_running() -> bool:
    """Check if the Ollama server is running locally."""
    try:
        result = requests.get("http://localhost:11434", timeout=3)
        return b"Ollama is running" in result.content
    except requests.exceptions.ConnectionError:
        return False


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    if not check_ollama_running():
        print("ERROR: Ollama is not running.")
        print("Please start it with: ollama serve")
        print(f"Then pull the model with: ollama pull {MODEL}")
        sys.exit(1)

    summary = summarize(url)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()
