from src.ingestion.scraper import clean_html, scrape_one
import pytest


def test_clean_html_removes_navigation_and_scripts():
    title, text = clean_html("""
      <html><head><title>Fund FAQ</title><script>bad()</script></head>
      <body><nav>Menu link</nav><main><h1>Expense ratio</h1><p>0.5%</p></main>
      <footer>Footer links</footer></body></html>
    """)
    assert title == "Fund FAQ"
    assert "Expense ratio" in text and "0.5%" in text
    assert "Menu link" not in text and "Footer links" not in text and "bad()" not in text


def test_scraper_rejects_non_groww_url():
    import requests
    with pytest.raises(ValueError):
        scrape_one(requests.Session(), "https://example.com/page")


def test_scraper_rejects_http_url():
    import requests
    with pytest.raises(ValueError):
        scrape_one(requests.Session(), "http://groww.in/page")
