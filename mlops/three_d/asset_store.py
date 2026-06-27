"""Filesystem-backed CV Ops 3D asset store.

The store owns the durable ``database/3D`` layout. Training and nerfstudio
execution are intentionally separate; this module creates stable draft assets
with manifests that later job runners can consume.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ASSET_BUCKETS: dict[str, str] = {
    "object": "objects",
    "scene": "scenes",
    "area": "areas",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class AssetCreateResult:
    """Result from staging a 3D asset shell."""

    asset_type: str
    slug: str
    root: Path
    manifest_path: Path
    created: bool
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "slug": self.slug,
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "created": self.created,
            "manifest": dict(self.manifest),
        }


def slugify_asset_name(value: object, *, fallback: str = "untitled-3d-asset") -> str:
    """Return a filesystem-safe slug for a CV Ops 3D asset."""

    slug = _SLUG_RE.sub("-", str(value or "").strip().lower()).strip("-")
    slug = slug[:80].strip("-")
    return slug or fallback


class ThreeDAssetStore:
    """Create and load 3D asset manifests under ``database/3D``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for bucket in ASSET_BUCKETS.values():
            (self.root / bucket).mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    def asset_root(self, asset_type: str, slug: str) -> Path:
        normalized_type = self.normalize_asset_type(asset_type)
        safe_slug = slugify_asset_name(slug)
        return self.root / ASSET_BUCKETS[normalized_type] / safe_slug

    def create_draft(
        self,
        payload: dict[str, Any],
        *,
        overwrite: bool = False,
        materialize_source: bool = True,
    ) -> AssetCreateResult:
        """Create folders and write ``manifest.json`` for a draft asset."""

        self.ensure_root()
        asset_type = self.normalize_asset_type(str(payload.get("asset_type") or "object"))
        name = str(payload.get("name") or "").strip()
        slug = slugify_asset_name(payload.get("slug") or name)
        target = self.asset_root(asset_type, slug)
        manifest_path = target / "manifest.json"
        created = not target.exists()

        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"3D asset already exists: {manifest_path}")

        for folder in ("inputs", "nerfstudio", "outputs"):
            (target / folder).mkdir(parents=True, exist_ok=True)

        now = time.time()
        manifest = self._normalize_manifest(
            payload,
            asset_type=asset_type,
            slug=slug,
            target=target,
            created_at=now,
        )
        if materialize_source:
            manifest["source"] = self._materialize_source(target, dict(manifest.get("source") or {}))
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            if isinstance(existing, dict):
                manifest["created_at"] = existing.get("created_at") or now
        manifest["updated_at"] = now

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return AssetCreateResult(
            asset_type=asset_type,
            slug=slug,
            root=target,
            manifest_path=manifest_path,
            created=created,
            manifest=manifest,
        )

    def _materialize_source(self, target: Path, source: dict[str, Any]) -> dict[str, Any]:
        """Represent the selected source inside ``inputs/`` using symlinks where possible."""

        source_kind = str(source.get("kind") or "").strip()
        source_path_text = str(source.get("path") or "").strip()
        source["materialized"] = False
        source["input_paths"] = []
        if not source_path_text:
            return source

        source_path = Path(source_path_text).expanduser()
        inputs_root = target / "inputs"
        source_root = inputs_root / "source"
        source_root.mkdir(parents=True, exist_ok=True)

        source_ref = {
            "kind": source_kind,
            "path": str(source_path),
            "created_at": time.time(),
            "mode": "symlink_or_copy",
        }
        (inputs_root / "source_reference.json").write_text(
            json.dumps(source_ref, indent=2),
            encoding="utf-8",
        )

        materialized: list[str] = []
        if source_kind == "image_folder" and source_path.is_dir():
            for child in sorted(source_path.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_file():
                    continue
                if child.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".tif",
                    ".tiff",
                    ".bmp",
                }:
                    continue
                dest = source_root / child.name
                self._link_or_copy(child, dest)
                materialized.append(str(dest))
        elif source_path.is_file():
            dest = source_root / source_path.name
            self._link_or_copy(source_path, dest)
            materialized.append(str(dest))

        source["materialized"] = bool(materialized)
        source["input_paths"] = materialized
        source["reference_path"] = str(inputs_root / "source_reference.json")
        return source

    @staticmethod
    def _link_or_copy(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(src.expanduser().resolve())
        except OSError:
            shutil.copy2(src, dest)

    def load_manifest(self, asset_type: str, slug: str) -> dict[str, Any]:
        path = self.asset_root(asset_type, slug) / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_assets(self) -> list[dict[str, Any]]:
        self.ensure_root()
        out: list[dict[str, Any]] = []
        for asset_type, bucket in ASSET_BUCKETS.items():
            base = self.root / bucket
            for path in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not path.is_dir():
                    continue
                manifest_path = path / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(manifest, dict):
                    out.append(
                        {
                            "asset_type": asset_type,
                            "slug": path.name,
                            "root": str(path),
                            "manifest_path": str(manifest_path),
                            "manifest": manifest,
                        }
                    )
        return out

    @staticmethod
    def normalize_asset_type(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in ASSET_BUCKETS:
            raise ValueError(f"unsupported 3D asset type: {value}")
        return normalized

    def _normalize_manifest(
        self,
        payload: dict[str, Any],
        *,
        asset_type: str,
        slug: str,
        target: Path,
        created_at: float,
    ) -> dict[str, Any]:
        manifest = dict(payload)
        manifest["version"] = int(manifest.get("version") or 1)
        manifest["status"] = str(manifest.get("status") or "draft")
        manifest["asset_type"] = asset_type
        manifest["slug"] = slug
        manifest["created_at"] = created_at

        database = dict(manifest.get("database") or {})
        database.update(
            {
                "sector_path": "/3D",
                "root": str(self.root),
                "target_path": str(target),
                "folders": ["inputs", "nerfstudio", "outputs"],
                "manifest_path": str(target / "manifest.json"),
            }
        )
        manifest["database"] = database

        manifest.setdefault("source", {"kind": "", "path": ""})
        manifest.setdefault("bounds_meters", {"width": 0.0, "depth": 0.0, "height": 0.0})
        manifest.setdefault("nerfstudio", {"command_plan": [], "wired": False})
        manifest.setdefault("artifacts", [])
        manifest.setdefault("lineage", [])
        return manifest
