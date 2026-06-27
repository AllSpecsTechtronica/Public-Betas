from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from mlops.pipeline import registry as reg


@dataclass(frozen=True)
class LabeledItem:
    """One staged image with its YOLO-normalized boxes.

    Boxes are (class_index, cx, cy, w, h) with cx/cy/w/h in [0, 1].
    """
    image_path: Path
    boxes: tuple[tuple[int, float, float, float, float], ...]


def _format_label_line(box: tuple[int, float, float, float, float]) -> str:
    cls, cx, cy, w, h = box
    return f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if n <= 0:
        return [], []
    val_frac = max(0.0, min(1.0, float(val_frac)))
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val = sorted(idx[:n_val])
    train = sorted(idx[n_val:])
    return train, val


def emit_yolo_dataset(
    *,
    slug: str,
    classes: list[str],
    items: Iterable[LabeledItem],
    val_frac: float = 0.2,
    seed: int = 0,
) -> Path:
    """Materialize a YOLO dataset under `database/<slug>/`.

    Layout produced (matches what `detect_library_dataset_format` expects):
        database/<slug>/
            images/train/  val/
            labels/train/  val/
            classes.txt

    The slug must already exist (created by [create_library_dataset_root]).
    """
    if not classes:
        raise ValueError("classes is empty")
    items = [it for it in items if it.boxes]
    if not items:
        raise ValueError("no labeled items provided")

    base = reg.resolve_library_dataset_path(slug)
    tmp_dir = base / ".emit-tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    materialized_items: list[LabeledItem] = []
    images_root = (base / "images").resolve()
    try:
        for idx, item in enumerate(items):
            src = Path(item.image_path)
            try:
                src_resolved = src.resolve()
                inside_generated_images = src_resolved.is_relative_to(images_root)
            except Exception:
                inside_generated_images = False
            if inside_generated_images and src.exists():
                tmp_item_dir = tmp_dir / f"{idx:06d}"
                tmp_item_dir.mkdir(parents=True, exist_ok=True)
                tmp_img = tmp_item_dir / src.name
                shutil.copy2(src, tmp_img)
                materialized_items.append(LabeledItem(tmp_img, item.boxes))
            else:
                materialized_items.append(item)
        items = materialized_items
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    # Rebuild generated YOLO output from the current staged labels each emit.
    # raw/, staged/, scrap.json, and QA cache live alongside these directories
    # and must remain intact.
    for generated_dir in (base / "images", base / "labels"):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    img_train = base / "images" / "train"
    img_val = base / "images" / "val"
    lbl_train = base / "labels" / "train"
    lbl_val = base / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = _split_indices(len(items), val_frac, seed)
    assignments = [(i, "train") for i in train_idx] + [(i, "val") for i in val_idx]

    for idx, split in assignments:
        item = items[idx]
        src = Path(item.image_path)
        if not src.exists():
            continue
        for box in item.boxes:
            cls_idx = int(box[0])
            if cls_idx < 0 or cls_idx >= len(classes):
                raise ValueError(
                    f"class index {cls_idx} for {src.name} is outside classes.txt "
                    f"(0..{len(classes) - 1})"
                )
        img_dest_dir = img_train if split == "train" else img_val
        lbl_dest_dir = lbl_train if split == "train" else lbl_val
        img_dest = img_dest_dir / src.name
        shutil.copy2(src, img_dest)
        label_lines = [_format_label_line(b) for b in item.boxes]
        (lbl_dest_dir / (src.stem + ".txt")).write_text(
            "\n".join(label_lines) + "\n", encoding="utf-8"
        )

    (base / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    data_yaml = {
        "path": base.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(classes)},
    }
    (base / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return base
