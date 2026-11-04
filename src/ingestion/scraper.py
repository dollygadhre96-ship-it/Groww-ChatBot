"""Scrape only the exact URLs configured in config/config.yaml.

No link discovery, crawling, or internal-page traversal is performed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
USER_AGENT = "Groww-MF-FAQ-Research-Bot/1.0 (educational demo)"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled"
    # Remove non-content and common site chrome.
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        node.decompose()
    for selector in ["[role=navigation]", "[role=footer]", ".breadcrumb", ".sidebar"]:
        for node in soup.select(selector):
            node.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def scrape_one(session: requests.Session, url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "groww.in":
        raise ValueError(f"URL is not an approved HTTPS Groww URL: {url}")

    response = session.get(url, timeout=(10, 30), allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" not in content_type:
        raise ValueError(f"Unexpected content type for {url}: {content_type}")

    title, text = clean_html(response.text)
    if not text.strip():
        raise ValueError(f"No readable text extracted from {url}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    record = {
        "source_id": source_id,
        "title": title,
        "url": url,  # configured URL, not a discovered internal link
        "retrieved_at": fetched_at,
        "content_hash": digest,
        "text": text,
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{source_id}.html").write_text(response.text, encoding="utf-8")
    (PROCESSED_DIR / f"{source_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def scrape_configured_sources() -> dict[str, Any]:
    config = load_config()
    # Only this explicit list is scraped. No anchors are inspected or followed.
    urls = config.get("sources", {}).get("urls", [])
    if not urls:
        raise RuntimeError("No source URLs configured in config/config.yaml")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    results = {"success": [], "failed": []}

    for url in urls:
        try:
            record = None
            for attempt in range(3):
                try:
                    record = scrape_one(session, url)
                    break
                except (requests.RequestException, ValueError) as exc:
                    if attempt == 2:
                        raise
                    log.warning("Attempt %s failed for %s: %s", attempt + 1, url, exc)
                    time.sleep(1.5 * (attempt + 1))
            results["success"].append({
                "url": url, "source_id": record["source_id"],
                "title": record["title"], "characters": len(record["text"]),
            })
            log.info("Scraped %s (%s characters)", url, len(record["text"]))
        except Exception as exc:
            results["failed"].append({"url": url, "error": str(exc)})
            log.error("Failed %s: %s", url, exc)

    report_path = PROCESSED_DIR / "scrape_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Scrape report saved to %s", report_path)
    return results


if __name__ == "__main__":
    report = scrape_configured_sources()
    print(json.dumps(report, indent=2))
