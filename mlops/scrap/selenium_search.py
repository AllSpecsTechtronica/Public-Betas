from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import html
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Google Images: thumbnails are often img.sFlh5c with src on encrypted-tbn*.gstatic.com (valid JPEG/WebP).
_GRID_IMG_SELECTOR = (
    "#islrg img[src], "
    "#islrg g-img img[src], "
    "div.islrc img[src], "
    "g-img img[src], "
    "img.sFlh5c[src*='gstatic'], "
    "img.sFlh5c[src*='encrypted-tbn']"
)

_REQUESTS_LOCAL = threading.local()


def _normalize_media_url(url: str) -> str:
    u = (url or "").strip()
    if "&amp;" in u:
        u = u.replace("&amp;", "&")
    return u


def _is_candidate_media_url(url: str) -> bool:
    u = _normalize_media_url(url)
    return u.startswith("http") or u.startswith("data:image/")


def _is_google_image_candidate_url(url: str) -> bool:
    u = _normalize_media_url(url)
    if not u.startswith("http"):
        return False
    if u.startswith("data:image/"):
        return True
    lowered = u.lower()
    if "encrypted-tbn" in lowered or "gstatic.com/images" in lowered:
        return True
    if "ggpht.com" in lowered or "googleusercontent.com" in lowered:
        return True
    return False


def _dedupe_candidate_urls(urls: list[str], extra: list[str]) -> list[str]:
    seen = {_normalize_media_url(u) for u in urls}
    merged = list(urls)
    for raw in extra:
        u = _normalize_media_url(raw)
        if not _is_candidate_media_url(u) or u in seen:
            continue
        seen.add(u)
        merged.append(u)
    return merged


def _normalize_source_mode(raw: str | None) -> str:
    mode = str(raw or "auto").strip().lower().replace("_", "-")
    if mode in {"bing", "bing-only"}:
        return "bing-only"
    if mode in {"google", "google-only"}:
        return "google-only"
    if mode in {"auto", "", "hybrid"}:
        return "auto"
    return "auto"


def _read_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except Exception:
        return default


def _read_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except Exception:
        return default


def _get_download_worker_count(max_count: int) -> int:
    default = 6
    cap = max(1, min(12, int(max_count) if max_count > 0 else 1))
    return _read_int_env(
        "MLOPS_SCRAP_DOWNLOAD_WORKERS",
        min(default, cap),
        minimum=1,
        maximum=cap,
    )


def _get_thread_requests_session(user_agent: str):
    import requests

    session = getattr(_REQUESTS_LOCAL, "session", None)
    current_ua = getattr(_REQUESTS_LOCAL, "user_agent", None)
    if session is None or current_ua != user_agent:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
        _REQUESTS_LOCAL.session = session
        _REQUESTS_LOCAL.user_agent = user_agent
    return session


def _pick_best_preview_src(imgs) -> str:
    """Prefer a non-thumbnail http(s) src when present; else use gstatic thumbnail (still a real image)."""
    best_non = ""
    best_tbn = ""
    for img_el in imgs:
        src = _normalize_media_url(img_el.get_attribute("src") or "")
        if not src.startswith("http"):
            continue
        if "encrypted-tbn" in src or "encrypted-tbn0" in src:
            if len(src) > len(best_tbn):
                best_tbn = src
        else:
            if len(src) > len(best_non):
                best_non = src
    return best_non or best_tbn or ""


def _interruptible_sleep(
    dur_s: float,
    *,
    poll_continue: Callable[[], bool] | None = None,
    step_s: float = 0.2,
) -> bool:
    """Sleep up to ``dur_s`` in short slices while ``poll_continue`` stays True."""
    if dur_s <= 0:
        return True if poll_continue is None else poll_continue()
    elapsed = 0.0
    while elapsed < dur_s:
        if poll_continue is not None and not poll_continue():
            return False
        slip = min(float(step_s), dur_s - elapsed)
        time.sleep(slip)
        elapsed += slip
    return True


def _cdp_stealth(driver) -> None:
    """Reduce trivial headless detection (best-effort; Google may still block)."""
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": (
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                ),
            },
        )
    except Exception:
        pass


def _count_grid_thumbnails(driver) -> int:
    from selenium.webdriver.common.by import By

    n = len(driver.find_elements(By.CSS_SELECTOR, _GRID_IMG_SELECTOR))
    if n > 0:
        return n
    try:
        n2 = driver.execute_script(
            """
            return document.querySelectorAll(
              '#islrg img[src], div.islrc img[src], img.sFlh5c[src*="encrypted-tbn"], img.sFlh5c[src*="gstatic"]'
            ).length;
            """
        )
        return int(n2 or 0)
    except Exception:
        return 0


def _scroll_results_ViewPort(driver) -> None:
    driver.execute_script(
        """
        const root = document.querySelector('#islrg') || document.querySelector('div.islrc');
        if (root) {
          root.scrollTop = root.scrollHeight;
        } else {
          window.scrollTo(0, document.body.scrollHeight);
        }
        """
    )


def _dismiss_google_consent_if_present(
    driver,
    *,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> None:
    """Click through common Google cookie / consent UIs so the image grid can render."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        StaleElementReferenceException,
    )

    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def click_if_any(by, selector: str) -> int:
        n = 0
        try:
            elems = driver.find_elements(by, selector)
        except Exception:
            return 0
        for el in elems:
            if poll_continue is not None and not poll_continue():
                return n
            try:
                if not el.is_displayed():
                    continue
            except Exception:
                continue
            try:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                WebDriverWait(driver, 3).until(EC.element_to_be_clickable(el))
                el.click()
                n += 1
                time.sleep(0.45)
            except (ElementClickInterceptedException, StaleElementReferenceException):
                try:
                    driver.execute_script("arguments[0].click();", el)
                    n += 1
                    time.sleep(0.45)
                except Exception:
                    pass
            except Exception:
                try:
                    driver.execute_script("arguments[0].click();", el)
                    n += 1
                    time.sleep(0.45)
                except Exception:
                    pass
        return n

    driver.switch_to.default_content()
    locators = [
        (By.CSS_SELECTOR, "button#L2AGLb"),
        (By.CSS_SELECTOR, "button[jsname='b3VHJd']"),
        (By.CSS_SELECTOR, "form[action*='consent'] button"),
        (By.XPATH, "//button[contains(., 'Accept all')]"),
        (By.XPATH, "//button[contains(., 'I agree')]"),
        (By.XPATH, "//button[contains(., 'Accept')]"),
    ]
    for _ in range(3):
        if poll_continue is not None and not poll_continue():
            return
        clicked = 0
        for by, sel in locators:
            clicked += click_if_any(by, sel)
        if clicked:
            p(f"Clicked through Google UI ({clicked} element(s)); waiting for image grid…")
        time.sleep(0.6)
        if _count_grid_thumbnails(driver) >= 4:
            return


def _wait_for_grid_thumbnails(
    driver,
    *,
    timeout_s: float = 22.0,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> int:
    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    deadline = time.time() + timeout_s
    last_log = 0.0
    while time.time() < deadline:
        if poll_continue is not None and not poll_continue():
            return 0
        n = _count_grid_thumbnails(driver)
        if n > 0:
            p(f"Found {n} result-thumbnail <img> node(s) in the grid.")
            return n
        now = time.time()
        if now - last_log > 4.0:
            p("Waiting for Google Images thumbnails (consent / slow render)…")
            last_log = now
        time.sleep(0.5)
    n = _count_grid_thumbnails(driver)
    if n == 0:
        p(
            "No thumbnails in #islrg / g-img after wait — Google may block headless, "
            "show captcha, or require consent. Try MLOPS_SCRAP_GOOGLE_HEADLESS=0 for a visible window."
        )
    return n


def _describe_google_surface(driver) -> str:
    """Short DOM summary for the 'zero thumbnails' case."""
    try:
        title = str(getattr(driver, "title", "") or "").strip()
    except Exception:
        title = ""
    try:
        current_url = str(getattr(driver, "current_url", "") or "").strip()
    except Exception:
        current_url = ""
    try:
        body_preview = str(
            driver.execute_script(
                """
                const txt = ((document.body && document.body.innerText) || '')
                  .replace(/\\s+/g, ' ')
                  .trim();
                return txt.slice(0, 280);
                """
            )
            or ""
        ).strip()
    except Exception:
        body_preview = ""
    try:
        iframe_count = int(
            driver.execute_script("return document.querySelectorAll('iframe').length;") or 0
        )
    except Exception:
        iframe_count = 0
    parts = [
        f"title={title!r}" if title else "title=<empty>",
        f"url={current_url!r}" if current_url else "url=<empty>",
        f"iframes={iframe_count}",
    ]
    if body_preview:
        parts.append(f"body={body_preview!r}")
    return "Google surface debug: " + " | ".join(parts)


@dataclass(frozen=True)
class SearchResult:
    saved: list[Path]
    attempted: int
    skipped: int
    cancelled: bool = False


def _build_driver(user_agent: str, *, on_progress: Callable[[str], None] | None = None) -> Any:
    """Lazy-import selenium so the rest of the package works without it installed."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service

        p(
            "Resolving Chromedriver (webdriver-manager; may download on first use — "
            "this can take a few minutes with no further lines until done)…"
        )
        service = Service(ChromeDriverManager().install())
        p("Chromedriver binary ready.")
    except Exception:
        service = None
        p("webdriver-manager unavailable or failed — using Selenium default driver resolution.")

    opts = Options()
    headless_env = str(os.environ.get("MLOPS_SCRAP_GOOGLE_HEADLESS", "1")).strip().lower()
    if headless_env not in {"0", "false", "no", "off"}:
        opts.add_argument("--headless=new")
    else:
        p("Headless disabled (MLOPS_SCRAP_GOOGLE_HEADLESS) — browser window will be visible.")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1920,2400")
    opts.add_argument(f"--user-agent={user_agent}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    if service is not None:
        return webdriver.Chrome(service=service, options=opts)
    return webdriver.Chrome(options=opts)


def _scroll_until_enough(
    driver,
    want: int,
    max_passes: int = 20,
    *,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> None:
    from selenium.webdriver.common.by import By

    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    last_count = 0
    stagnant = 0
    for i in range(max_passes):
        if poll_continue is not None and not poll_continue():
            return
        _scroll_results_ViewPort(driver)
        p(f"Scroll pass {i + 1}/{max_passes}: waiting for thumbnails…")
        if poll_continue is not None and not _interruptible_sleep(1.2, poll_continue=poll_continue):
            return
        thumbs = driver.find_elements(By.CSS_SELECTOR, _GRID_IMG_SELECTOR)
        count = len(thumbs)
        p(
            f"Scroll pass {i + 1}: found {count} grid thumbnail node(s) "
            f"(selector {_GRID_IMG_SELECTOR!r}; want about {want * 2}+ for harvest)"
        )
        if count >= want * 2:
            p("Enough thumbnails visible; stopping scroll early.")
            return
        if count == last_count:
            stagnant += 1
            if stagnant >= 3:
                p("Thumbnail count stagnant; stopping scroll.")
                return
        else:
            stagnant = 0
        last_count = count


def _harvest_image_urls(
    driver,
    want: int,
    *,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> tuple[list[str], bool]:
    """Collect image URLs from the Google Images grid.

    Thumbnails are usually ``encrypted-tbn*.gstatic.com`` — those are real image
    endpoints, not HTML result links. We read ``src`` from grid ``img`` nodes
    (including ``visibility:hidden`` preview tiles) and optionally click a few
    cells to discover larger ``src`` values when the direct list is thin.
    """
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import WebDriverException

    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    max_collect = max(want * 4, want + 20)
    seen: set[str] = set()
    urls: list[str] = []

    def push(u: str) -> None:
        u = _normalize_media_url(u)
        if not _is_google_image_candidate_url(u) or u in seen:
            return
        seen.add(u)
        urls.append(u)

    thumbs = driver.find_elements(By.CSS_SELECTOR, _GRID_IMG_SELECTOR)
    p(
        f"Harvest: collecting src from up to {len(thumbs)} grid <img> nodes "
        f"(encrypted-tbn / gstatic thumbnails count as real images)…"
    )
    for el in thumbs:
        if poll_continue is not None and not poll_continue():
            p(f"Harvest interrupted ({len(urls)} URLs)")
            return urls, True
        if len(urls) >= max_collect:
            break
        raw = el.get_attribute("src") or ""
        push(raw)

    if len(urls) < want and len(thumbs) < want * 2:
        try:
            js_urls = driver.execute_script(
                """
                const cap = arguments[0];
                const sel = (
                  '#islrg img[src], div.islrc img[src], g-img img[src], ' +
                  'img.sFlh5c[src*="gstatic"], img.sFlh5c[src*="encrypted-tbn"]'
                );
                const nodes = document.querySelectorAll(sel);
                const out = [];
                const seen = new Set();
                for (const img of nodes) {
                  let u = img.getAttribute('src') || '';
                  u = u.replace(/&amp;/g, '&');
                  if (!u.startsWith('http')) continue;
                  if (seen.has(u)) continue;
                  seen.add(u);
                  out.push(u);
                  if (out.length >= cap) break;
                }
                return out;
                """,
                max_collect,
            )
            if isinstance(js_urls, list):
                for u in js_urls:
                    if poll_continue is not None and not poll_continue():
                        return urls, True
                    if isinstance(u, str):
                        push(u)
        except Exception as exc:
            log.debug("scrap: JS URL harvest fallback failed: %s", exc)

    if len(urls) >= want:
        p(f"Harvest complete: {len(urls)} unique candidate URL(s) (direct src from grid).")
        return urls, False

    p(
        f"Direct src harvest only found {len(urls)} URL(s); trying click-to-expand "
        f"for up to {min(60, len(thumbs))} thumbnails…"
    )
    for idx, thumb in enumerate(thumbs[:60]):
        if poll_continue is not None and not poll_continue():
            p(f"Harvest interrupted ({len(urls)} URLs)")
            return urls, True
        if len(urls) >= max_collect:
            break
        if idx > 0 and idx % 20 == 0:
            p(f"Click-harvest progress: thumb {idx}/60, collected {len(urls)} URLs")
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", thumb)
            thumb.click()
        except WebDriverException:
            try:
                driver.execute_script("arguments[0].click();", thumb)
            except Exception:
                raw = thumb.get_attribute("src") or ""
                push(raw)
                continue
        except Exception:
            raw = thumb.get_attribute("src") or ""
            push(raw)
            continue
        if poll_continue is not None and not _interruptible_sleep(0.55, poll_continue=poll_continue):
            p(f"Harvest interrupted ({len(urls)} URLs)")
            return urls, True
        previews = driver.find_elements(By.CSS_SELECTOR, "img")
        best = _pick_best_preview_src(previews)
        if not best:
            best = _normalize_media_url(thumb.get_attribute("src") or "")
        push(best)

    p(f"Harvest complete: {len(urls)} unique candidate URL(s)")
    return urls, False


def _extract_bing_html_image_candidates(html_text: str, max_collect: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def push(raw: str) -> None:
        if len(out) >= max_collect:
            return
        u = _normalize_media_url(html.unescape(raw or ""))
        if not _is_candidate_media_url(u) or u in seen:
            return
        seen.add(u)
        out.append(u)

    for match in re.finditer(r"""\bm=(["'])(.*?)\1""", html_text, re.S):
        raw = html.unescape(match.group(2))
        if not raw or "murl" not in raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        for key in ("murl", "imgurl", "turl"):
            value = payload.get(key)
            if isinstance(value, str):
                push(value)
        if len(out) >= max_collect:
            return out

    for attr_name in ("data-src", "data-expanded-url", "src"):
        for match in re.finditer(rf"""\b{attr_name}=(["'])(.*?)\1""", html_text, re.S | re.I):
            push(match.group(2))
            if len(out) >= max_collect:
                return out
    return out


def _harvest_bing_image_urls(
    query: str,
    want: int,
    *,
    user_agent: str,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> tuple[list[str], bool]:
    import requests

    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    max_collect = max(want * 4, want + 20)
    urls: list[str] = []
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()
    cancelled = False

    for page_idx, first in enumerate((0, 35, 70, 105), start=1):
        if poll_continue is not None and not poll_continue():
            cancelled = True
            break
        params = {
            "q": query,
            "form": "HDRSC2",
            "first": str(first),
            "tsc": "ImageBasicHover",
        }
        page_url = "https://www.bing.com/images/search"
        p(f"Fallback source Bing: requesting page {page_idx} (first={first})…")
        try:
            resp = session.get(page_url, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            p(f"Fallback source Bing: request failed on page {page_idx}: {exc}")
            continue
        found = _extract_bing_html_image_candidates(resp.text, max_collect=max_collect)
        before = len(urls)
        urls = _dedupe_candidate_urls(urls, found)
        gained = len(urls) - before
        p(
            f"Fallback source Bing: page {page_idx} yielded {gained} new candidate URL(s) "
            f"({len(urls)} total)."
        )
        if len(urls) >= max_collect or (gained == 0 and page_idx >= 2):
            break

    return urls, cancelled


def _save_url(
    url: str,
    out_dir: Path,
    throttle_s: float,
    user_agent: str,
    request_timeout_s: float,
) -> Path | None:
    if url.startswith("data:image/"):
        try:
            header, b64 = url.split(",", 1)
            ext = ".jpg"
            m = re.match(r"data:image/([a-zA-Z0-9]+);base64", header)
            if m:
                fmt = m.group(1).lower()
                ext = ".jpg" if fmt in {"jpeg", "jpg"} else f".{fmt}"
            data = base64.b64decode(b64)
        except Exception:
            return None
    else:
        try:
            session = _get_thread_requests_session(user_agent)
            resp = session.get(url, timeout=request_timeout_s)
            resp.raise_for_status()
            data = resp.content
            ctype = (resp.headers.get("Content-Type") or "").lower()
            is_image = "image/" in ctype
            if not is_image and data:
                if data.startswith(b"\xff\xd8\xff"):
                    is_image = True
                elif data.startswith(b"\x89PNG\r\n\x1a\n"):
                    is_image = True
                elif len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
                    is_image = True
            if not is_image:
                return None
            ext = ".jpg"
            for cand in (".jpg", ".jpeg", ".png", ".webp"):
                if cand[1:] in ctype:
                    ext = ".jpg" if cand == ".jpeg" else cand
                    break
        except Exception:
            return None
        finally:
            time.sleep(max(0.0, throttle_s))

    if not data or len(data) < 4096:
        return None
    digest = hashlib.sha1(data).hexdigest()
    dest = out_dir / f"{digest}{ext}"
    if dest.exists():
        return dest
    try:
        dest.write_bytes(data)
    except Exception:
        return None
    return dest


def _download_urls(
    urls: list[str],
    max_count: int,
    out_dir: Path,
    *,
    throttle_s: float,
    user_agent: str,
    worker_count: int,
    request_timeout_s: float,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> SearchResult:
    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    if max_count <= 0 or not urls:
        return SearchResult(saved=[], attempted=0, skipped=0, cancelled=False)

    worker_count = max(1, min(int(worker_count), int(max_count), len(urls)))
    p(
        f"Downloading up to {max_count} images into {out_dir} "
        f"({worker_count} worker(s), throttle {throttle_s}s per request)…"
    )

    saved: list[Path] = []
    attempted = 0
    skipped = 0
    cancelled = False
    next_idx = 0
    futures: dict[Any, int] = {}

    def submit_more(executor: ThreadPoolExecutor) -> None:
        nonlocal next_idx
        while (
            next_idx < len(urls)
            and len(saved) + len(futures) < max_count
            and len(futures) < worker_count
        ):
            fut = executor.submit(
                _save_url,
                urls[next_idx],
                out_dir,
                throttle_s,
                user_agent,
                request_timeout_s,
            )
            futures[fut] = next_idx
            next_idx += 1

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="scrapdl") as executor:
        submit_more(executor)
        while futures:
            if poll_continue is not None and not poll_continue():
                cancelled = True
                break
            done, _ = wait(list(futures), timeout=0.25, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for fut in done:
                index = futures.pop(fut)
                attempted += 1
                try:
                    path = fut.result()
                except Exception:
                    path = None
                if path is None:
                    skipped += 1
                    if attempted == 1 or attempted % 10 == 0:
                        p(
                            f"Download {attempted}/{len(urls)}: skip (invalid / tiny / non-image) "
                            f"| saved={len(saved)} skipped={skipped}"
                        )
                else:
                    saved.append(path)
                    if len(saved) == 1 or len(saved) % 5 == 0 or len(saved) >= max_count:
                        p(
                            f"Download {attempted}/{len(urls)}: saved {path.name} "
                            f"| total saved={len(saved)} skipped={skipped}"
                        )
                if cancelled or len(saved) >= max_count:
                    continue
                if attempted == 1 or attempted % max(10, worker_count * 2) == 0:
                    p(
                        f"Download progress: checked {attempted}/{len(urls)} candidates "
                        f"(saved={len(saved)} skipped={skipped})"
                    )
                submit_more(executor)
            if len(saved) >= max_count:
                break

    if cancelled:
        for fut in futures:
            fut.cancel()
        p(
            f"Download cancellation requested after {attempted} candidate(s) "
            f"(saved={len(saved)} skipped={skipped})."
        )

    return SearchResult(saved=saved[:max_count], attempted=attempted, skipped=skipped, cancelled=cancelled)


def search_google_images(
    query: str,
    max_count: int,
    out_dir: Path,
    *,
    throttle_s: float = 0.4,
    user_agent: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> SearchResult:
    """Drive headless Chrome at Google Images, scroll, harvest URLs, download.

    ``poll_continue`` returns False to abort (cancel generation or operator stop).
    ``max_count`` may be 0 to skip the browser phase (resume when raw/ is full).

    Failures inside the loop are logged and counted as `skipped`, never raised
    upward — the goal is "best effort", not "all or nothing". Selenium and
    Chrome must be installed; otherwise this function will raise on driver
    construction, which the caller surfaces to the user.
    """
    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ua = user_agent or os.environ.get("MLOPS_SCRAP_USER_AGENT") or DEFAULT_USER_AGENT
    source_mode = _normalize_source_mode(os.environ.get("MLOPS_SCRAP_IMAGE_SOURCE"))
    google_wait_s = _read_float_env(
        "MLOPS_SCRAP_GOOGLE_WAIT_S",
        10.0 if source_mode == "auto" else 22.0,
        minimum=1.0,
    )
    google_reload_wait_s = _read_float_env(
        "MLOPS_SCRAP_GOOGLE_RELOAD_WAIT_S",
        0.0 if source_mode == "auto" else 18.0,
        minimum=0.0,
    )
    request_timeout_s = _read_float_env("MLOPS_SCRAP_REQUEST_TIMEOUT_S", 12.0, minimum=2.0)
    download_workers = _get_download_worker_count(max_count)

    saved: list[Path] = []
    attempted = 0
    skipped = 0
    cancelled = False
    urls: list[str] = []

    if max_count <= 0:
        p("Download target is satisfied (0 new files requested) — skipping browser phase.")
        return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=False)

    url = f"https://www.google.com/search?tbm=isch&q={quote_plus(query)}"
    driver = None
    browser_phase_failed = False
    if source_mode != "bing-only":
        p("Building headless Chrome WebDriver (Selenium)…")
        try:
            if poll_continue is not None and not poll_continue():
                cancelled = True
                return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=True)
            driver = _build_driver(ua, on_progress=on_progress)
            p("WebDriver ready.")
            _cdp_stealth(driver)
            p(f"GET {url[:80]}…")
            driver.get(url)
            if poll_continue is not None and not _interruptible_sleep(1.0, poll_continue=poll_continue):
                cancelled = True
                return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=True)
            _dismiss_google_consent_if_present(driver, on_progress=on_progress, poll_continue=poll_continue)
            n_grid = _wait_for_grid_thumbnails(
                driver,
                timeout_s=google_wait_s,
                on_progress=on_progress,
                poll_continue=poll_continue,
            )
            if n_grid == 0 and google_reload_wait_s > 0.0:
                p("No grid thumbnails yet — reloading once, then retrying consent + wait…")
                driver.refresh()
                if poll_continue is not None and not _interruptible_sleep(2.0, poll_continue=poll_continue):
                    cancelled = True
                    return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=True)
                _dismiss_google_consent_if_present(
                    driver,
                    on_progress=on_progress,
                    poll_continue=poll_continue,
                )
                n_grid = _wait_for_grid_thumbnails(
                    driver,
                    timeout_s=google_reload_wait_s,
                    on_progress=on_progress,
                    poll_continue=poll_continue,
                )
            if _count_grid_thumbnails(driver) == 0:
                p(_describe_google_surface(driver))
                if source_mode == "auto":
                    p("Google grid stayed empty during the quick probe; skipping deeper retries and falling back.")
            else:
                p("Letting image grid settle (1.5s)…")
                if poll_continue is not None and not _interruptible_sleep(1.5, poll_continue=poll_continue):
                    cancelled = True
                    return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=True)
                if poll_continue is not None and not poll_continue():
                    cancelled = True
                    return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=True)
                _scroll_until_enough(
                    driver,
                    max_count,
                    on_progress=on_progress,
                    poll_continue=poll_continue,
                )
                if poll_continue is not None and not poll_continue():
                    cancelled = True
                    return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=True)
                urls, harvest_cancelled = _harvest_image_urls(
                    driver,
                    max_count * 3,
                    on_progress=on_progress,
                    poll_continue=poll_continue,
                )
                if harvest_cancelled:
                    cancelled = True
                log.info("scrap: harvested %d google url candidates for %r", len(urls), query)
        except Exception as exc:
            browser_phase_failed = True
            p(f"Google browser phase failed: {exc}")
        finally:
            if driver is not None:
                p("Closing WebDriver session…")
                try:
                    driver.quit()
                except Exception:
                    pass
    else:
        p("Source mode is Bing-only; skipping the Selenium/Google phase.")

    if not cancelled and len(urls) < max_count and source_mode != "google-only":
        p(
            f"Google yielded {len(urls)} candidate URL(s); "
            "falling back to Bing image search HTML…"
        )
        bing_urls, bing_cancelled = _harvest_bing_image_urls(
            query,
            max_count * 3,
            user_agent=ua,
            on_progress=on_progress,
            poll_continue=poll_continue,
        )
        urls = _dedupe_candidate_urls(urls, bing_urls)
        cancelled = cancelled or bing_cancelled
        log.info("scrap: total merged url candidates after Bing fallback: %d", len(urls))

    if browser_phase_failed and not urls and not cancelled:
        p("No candidate URLs were harvested from Google or the Bing fallback.")

    download_result = _download_urls(
        urls,
        max_count,
        out_dir,
        throttle_s=throttle_s,
        user_agent=ua,
        worker_count=download_workers,
        request_timeout_s=request_timeout_s,
        on_progress=on_progress,
        poll_continue=poll_continue,
    )
    saved = download_result.saved
    attempted = download_result.attempted
    skipped = download_result.skipped
    cancelled = cancelled or download_result.cancelled

    return SearchResult(saved=saved, attempted=attempted, skipped=skipped, cancelled=cancelled)
