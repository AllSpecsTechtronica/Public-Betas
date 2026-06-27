from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from mlops.scrap.selenium_search import (
    _download_urls,
    _dedupe_candidate_urls,
    _extract_bing_html_image_candidates,
    _normalize_source_mode,
)


def test_extract_bing_html_image_candidates_prefers_metadata_and_dedupes() -> None:
    html = """
    <html><body>
      <a class="iusc" m='{"murl":"https://images.example.com/full-a.jpg","turl":"https://thumbs.example.com/a.jpg"}'></a>
      <a class="iusc" m="{&quot;murl&quot;:&quot;https://images.example.com/full-b.png&quot;,&quot;turl&quot;:&quot;https://thumbs.example.com/b.png&quot;}"></a>
      <img class="mimg" src="https://thumbs.example.com/a.jpg" />
      <img class="mimg" data-src="https://images.example.com/from-data.webp" />
    </body></html>
    """

    urls = _extract_bing_html_image_candidates(html, max_collect=10)

    assert "https://images.example.com/full-a.jpg" in urls
    assert "https://thumbs.example.com/a.jpg" in urls
    assert "https://images.example.com/full-b.png" in urls
    assert "https://images.example.com/from-data.webp" in urls
    assert len(urls) == len(set(urls))


def test_dedupe_candidate_urls_normalizes_ampersands() -> None:
    base = ["https://images.example.com/photo?id=1&size=large"]
    extra = [
        "https://images.example.com/photo?id=1&amp;size=large",
        "https://images.example.com/photo?id=2&size=large",
        "",
    ]

    merged = _dedupe_candidate_urls(base, extra)

    assert merged == [
        "https://images.example.com/photo?id=1&size=large",
        "https://images.example.com/photo?id=2&size=large",
    ]


def test_normalize_source_mode_accepts_supported_aliases() -> None:
    assert _normalize_source_mode(None) == "auto"
    assert _normalize_source_mode("hybrid") == "auto"
    assert _normalize_source_mode("google") == "google-only"
    assert _normalize_source_mode("bing_only") == "bing-only"
    assert _normalize_source_mode("weird-value") == "auto"


def test_download_urls_uses_parallel_workers_without_overshooting_target() -> None:
    urls = [f"https://images.example.com/{idx}.jpg" for idx in range(8)]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_save(
        url: str,
        out_dir: Path,
        throttle_s: float,
        user_agent: str,
        request_timeout_s: float,
    ) -> Path | None:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        dest = out_dir / f"{Path(url).stem}.jpg"
        dest.write_bytes(b"image-bytes")
        with lock:
            active -= 1
        return dest

    with tempfile.TemporaryDirectory() as td, patch(
        "mlops.scrap.selenium_search._save_url",
        side_effect=fake_save,
    ):
        result = _download_urls(
            urls,
            4,
            Path(td),
            throttle_s=0.0,
            user_agent="ua",
            worker_count=4,
            request_timeout_s=1.0,
        )

    assert len(result.saved) == 4
    assert result.attempted == 4
    assert result.skipped == 0
    assert not result.cancelled
    assert max_active >= 2
