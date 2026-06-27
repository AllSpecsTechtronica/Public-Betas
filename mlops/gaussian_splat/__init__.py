"""Local 3D without Trellis — Gaussian splatting pipeline."""

from __future__ import annotations

from mlops.gaussian_splat.replication import (
    CalibrationKind,
    ReplicationManifest,
    SourceKind,
    list_images_in_folder,
    new_job_id,
    prepare_replication_workspace,
    validate_image_folder,
    validate_video_file,
)
from mlops.gaussian_splat.true_gaussian import GaussianRunResult, run_true_gaussian_pipeline

PIPELINE_STATUS = "true_gaussian_splat"

__all__ = [
    "PIPELINE_STATUS",
    "CalibrationKind",
    "ReplicationManifest",
    "SourceKind",
    "list_images_in_folder",
    "new_job_id",
    "prepare_replication_workspace",
    "GaussianRunResult",
    "run_true_gaussian_pipeline",
    "validate_image_folder",
    "validate_video_file",
]
