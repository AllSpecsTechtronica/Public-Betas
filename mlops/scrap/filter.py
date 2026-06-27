from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageResult:
    staged: list[Path]
    skipped_small: int
    skipped_dup: int
    skipped_unreadable: int


def _inspect_image(path: Path) -> tuple[tuple[int, int], str] | None:
    """Open the image once, returning dimensions and a stable dedupe key."""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(path) as im:
            im.load()
            dims = (int(im.width), int(im.height))
            try:
                import imagehash

                key = str(imagehash.phash(im.convert("RGB"), hash_size=8))
            except Exception:
                key = hashlib.sha1(path.read_bytes()).hexdigest()
            return dims, key
    except Exception:
        return None


def dedupe_and_stage(
    raw_dir: Path,
    staged_dir: Path,
    *,
    min_size: int = 0,
    on_progress: Callable[[str], None] | None = None,
    poll_continue: Callable[[], bool] | None = None,
) -> StageResult:
    """Read every image in `raw_dir`, optionally drop too-small files, and
    drop unreadable / near-duplicate files before copying survivors into
    `staged_dir` keyed by perceptual hash.

    Re-running on the same directories is idempotent: already-staged hashes are
    skipped without recopying.
    """
    def p(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    raw_dir = Path(raw_dir)
    staged_dir = Path(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = {p.stem for p in staged_dir.iterdir() if p.is_file()}
    staged: list[Path] = [p for p in staged_dir.iterdir() if p.is_file()]
    skipped_small = 0
    skipped_dup = 0
    skipped_unreadable = 0

    raw_files = sorted(x for x in raw_dir.iterdir() if x.is_file())
    min_edge = max(0, int(min_size or 0))
    size_policy = f"min edge {min_edge}px" if min_edge > 0 else "small-image filter disabled"
    p(f"Staging: {len(raw_files)} file(s) in raw/, {size_policy}, {len(staged)} already in staged/")

    for n, src in enumerate(raw_files, start=1):
        if poll_continue is not None and not poll_continue():
            p("Staging interrupted — partial staged set preserved.")
            break
        if n == 1 or n % 15 == 0:
            p(f"Staging scan {n}/{len(raw_files)}: {src.name}")
        inspected = _inspect_image(src)
        if inspected is None:
            skipped_unreadable += 1
            p(f"  skip unreadable: {src.name}")
            continue
        dims, key = inspected
        w, h = dims
        if min_edge > 0 and min(w, h) < min_edge:
            skipped_small += 1
            p(f"  skip small {w}x{h}: {src.name}")
            continue
        if key in seen:
            skipped_dup += 1
            continue
        seen.add(key)
        suffix = src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
        dest = staged_dir / f"{key}{suffix}"
        try:
            shutil.copyfile(src, dest)
            staged.append(dest)
            p(f"  staged -> {dest.name} ({w}x{h})")
        except Exception as exc:
            log.warning("stage copy failed for %s: %s", src, exc)
            skipped_unreadable += 1
            p(f"  skip copy error {src.name}: {exc}")

    p(
        f"Staging summary: {len(staged)} file(s) in staged/; "
        f"skipped small={skipped_small} dup={skipped_dup} unreadable={skipped_unreadable}"
    )
    return StageResult(
        staged=sorted(staged),
        skipped_small=skipped_small,
        skipped_dup=skipped_dup,
        skipped_unreadable=skipped_unreadable,
    )
