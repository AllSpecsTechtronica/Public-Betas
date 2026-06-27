from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    detected: bool
    matched_paths: tuple[str, ...]
    roots: tuple[Path, ...]


@dataclass(frozen=True)
class PrivacyStatus:
    protected: bool
    providers: tuple[ProviderStatus, ...]


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _icloud_roots(home: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    primary = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    if primary.exists():
        roots.append(primary.resolve())
    fallback = home / "Library" / "Mobile Documents"
    if fallback.exists() and fallback.resolve() not in roots:
        roots.append(fallback.resolve())
    return tuple(roots)


def _onedrive_roots(home: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    direct_candidates = [
        home / "OneDrive",
        home / "OneDrive - Personal",
        home / "OneDrive - Business",
        home / "OneDrive - Microsoft",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            roots.append(candidate.resolve())
    cloud_storage = home / "Library" / "CloudStorage"
    if cloud_storage.exists():
        for child in sorted(cloud_storage.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir() and child.name.lower().startswith("onedrive"):
                resolved = child.resolve()
                if resolved not in roots:
                    roots.append(resolved)
    return tuple(roots)


def detect_privacy_status(storage_paths: dict[str, Path | str], home: Path | None = None) -> PrivacyStatus:
    resolved_paths = {label: _resolve(path) for label, path in storage_paths.items()}
    home_path = _resolve(home or Path.home())
    provider_roots = {
        "iCloud Drive": _icloud_roots(home_path),
        "Microsoft OneDrive": _onedrive_roots(home_path),
    }
    providers: list[ProviderStatus] = []
    protected = True
    for provider_name, roots in provider_roots.items():
        matched: list[str] = []
        for label, candidate in resolved_paths.items():
            if any(_is_relative_to(candidate, root) for root in roots):
                matched.append(label)
        if matched:
            protected = False
        providers.append(
            ProviderStatus(
                name=provider_name,
                detected=bool(roots),
                matched_paths=tuple(matched),
                roots=roots,
            )
        )
    return PrivacyStatus(protected=protected, providers=tuple(providers))
