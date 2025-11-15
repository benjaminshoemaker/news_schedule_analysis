import os
import datetime as dt
from pathlib import Path
from string import Template

import feedparser
from openai import OpenAI

# -------- config --------

OPENAI_MODEL = "gpt-5-nano"
ENV_FILE = Path(".env.local")
FEEDS_FILE = Path("rss_feeds.txt")
PROMPT_TEMPLATE_FILE = Path("prompt_template.md")


def load_env_file(path: Path):
    """Load key=value pairs from a local env file without overriding existing vars."""
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


load_env_file(ENV_FILE)


def load_rss_feeds(path: Path):
    """Return a list of RSS feed URLs defined in a simple text file."""
    if not path.exists():
        raise FileNotFoundError(f"RSS feeds file not found: {path}")

    feeds = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        feeds.append(line)

    if not feeds:
        raise ValueError(f"No feeds defined in {path}")

    return feeds


def load_prompt_template(path: Path):
    """Load the user prompt template (with $today and $articles_block placeholders)."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    contents = path.read_text(encoding="utf-8")
    return Template(contents)


RSS_FEEDS = load_rss_feeds(FEEDS_FILE)
PROMPT_TEMPLATE = load_prompt_template(PROMPT_TEMPLATE_FILE)

MAX_ARTICLES = 15  # total across all feeds

# -------- helpers --------

def get_today_str():
    # Use UTC date for consistency in Actions; adjust if you want local time
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def fetch_articles():
    articles = []
    for feed_url in RSS_FEEDS:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            published = getattr(entry, "published", "") or getattr(entry, "updated", "")

            if not title or not link:
                continue

            articles.append(
                {
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": published,
                    "source": parsed.feed.get("title", feed_url),
                }
            )

    # Deduplicate by link
    unique = {}
    for a in articles:
        unique[a["link"]] = a
    articles = list(unique.values())

    # Sort by published if available (best-effort)
    def sort_key(a):
        return a.get("published", "")

    articles.sort(key=sort_key, reverse=True)

    return articles[:MAX_ARTICLES]


def build_articles_block(articles):
    """Compact representation of the articles for the LLM prompt."""
    lines = []
    for idx, a in enumerate(articles, start=1):
        lines.append(
            f"{idx}. Title: {a['title']}\n"
            f"   Source: {a['source']}\n"
            f"   Published: {a['published']}\n"
            f"   URL: {a['link']}\n"
            f"   Summary/Excerpt: {a['summary'][:500]}\n"
        )
    return "\n".join(lines)


def build_llm_prompt(articles):
    articles_block = build_articles_block(articles)
    return PROMPT_TEMPLATE.substitute(
        today=get_today_str(),
        articles_block=articles_block,
    )


def generate_report_md(articles):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    if not articles:
        # Fallback if feeds are empty
        return f"# Daily Research & Idea Report – {get_today_str()}\n\nNo articles found today."

    prompt = build_llm_prompt(articles)

    model_name = OPENAI_MODEL.strip()

    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You write clear, concise Markdown reports."},
            {"role": "user", "content": prompt},
        ],
    }

    if model_name.lower() != "gpt-5-nano":
        request_kwargs["temperature"] = 0.4

    completion = client.chat.completions.create(**request_kwargs)

    content = completion.choices[0].message.content
    return content


def save_report(markdown_text):
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    today = get_today_str()
    path = reports_dir / f"report-{today}.md"
    path.write_text(markdown_text, encoding="utf-8")
    print(f"Wrote report to {path}")


def main():
    articles = fetch_articles()
    report_md = generate_report_md(articles)
    save_report(report_md)


if __name__ == "__main__":
    main()
