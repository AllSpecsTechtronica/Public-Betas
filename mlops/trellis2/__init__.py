from __future__ import annotations

from .capability import Capabilities, detect
from .depth_anything_local import (
    depth_bundle_available,
    generate_depth_glb,
    resolve_depth_mlpackage,
)
from .client import DEFAULT_PARAMS, SamplingParams, Trellis2Client, Trellis2Error
from .jobs import Job, JobStatus, JobStore

__all__ = [
    "Capabilities",
    "DEFAULT_PARAMS",
    "Job",
    "JobStatus",
    "JobStore",
    "SamplingParams",
    "Trellis2Client",
    "Trellis2Error",
    "depth_bundle_available",
    "detect",
    "generate_depth_glb",
    "resolve_depth_mlpackage",
]
