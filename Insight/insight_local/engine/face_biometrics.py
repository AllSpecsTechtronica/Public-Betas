from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import MODELS_DIR


FACE_DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
FACE_ALIGN_SIZE = (112, 112)


@dataclass
class FaceSample:
    aligned_bgr: np.ndarray
    feature: np.ndarray
    quality: float
    detection_score: float
    bbox: tuple[int, int, int, int]


class FaceBiometricsEngine:
    """
    Local face detector + face recognizer built on YuNet and SFace.
    """

    def __init__(
        self,
        detector_model_path: Path = FACE_DETECTOR_MODEL,
        recognizer_model_path: Path = FACE_RECOGNIZER_MODEL,
    ) -> None:
        self._detector_model_path = Path(detector_model_path)
        self._recognizer_model_path = Path(recognizer_model_path)
        self._lock = threading.RLock()
        self._detector = None
        self._recognizer = None
        self._ready = False
        self._load_error: Optional[str] = None
        self._feature_dim = 0

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def ensure_loaded(self) -> bool:
        if self._ready:
            return True
        with self._lock:
            if self._ready:
                return True
            try:
                if not self._detector_model_path.exists():
                    raise FileNotFoundError(f"Face detector model not found: {self._detector_model_path}")
                if not self._recognizer_model_path.exists():
                    raise FileNotFoundError(f"Face recognizer model not found: {self._recognizer_model_path}")
                self._detector = cv2.FaceDetectorYN_create(
                    str(self._detector_model_path),
                    "",
                    (320, 320),
                    score_threshold=0.85,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self._recognizer = cv2.FaceRecognizerSF_create(
                    str(self._recognizer_model_path),
                    "",
                )
                self._load_error = None
                self._ready = True
            except Exception as exc:
                self._load_error = str(exc)
                self._ready = False
        return self._ready

    def extract_face(self, bgr_image: np.ndarray) -> Optional[FaceSample]:
        if bgr_image is None or bgr_image.size == 0:
            return None
        if not self.ensure_loaded():
            return None

        height, width = bgr_image.shape[:2]
        if width < 32 or height < 32:
            return None

        candidates: list[FaceSample] = []
        with self._lock:
            assert self._detector is not None
            assert self._recognizer is not None
            for detect_image, pad_x, pad_y in self._detection_candidates(bgr_image):
                detect_h, detect_w = detect_image.shape[:2]
                self._detector.setInputSize((detect_w, detect_h))
                _, faces = self._detector.detect(detect_image)
                if faces is None or len(faces) == 0:
                    continue
                best_face = self._pick_best_face(faces, detect_w, detect_h)
                try:
                    aligned = self._recognizer.alignCrop(detect_image, best_face)
                    feature = self._recognizer.feature(aligned)
                except Exception:
                    continue

                feature_vec = np.asarray(feature, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(feature_vec))
                if norm < 1e-6:
                    continue
                feature_vec = feature_vec / norm

                bbox = self._face_bbox_in_original_frame(best_face, width, height, pad_x, pad_y)
                if bbox is None:
                    continue
                detection_score = float(best_face[-1])
                quality = self._estimate_quality(aligned, bbox, (width, height), detection_score)
                candidates.append(
                    FaceSample(
                        aligned_bgr=aligned,
                        feature=feature_vec,
                        quality=quality,
                        detection_score=detection_score,
                        bbox=bbox,
                    )
                )

        if not candidates:
            return None
        best_sample = max(candidates, key=lambda sample: (sample.quality, sample.detection_score))
        self._feature_dim = int(best_sample.feature.shape[0])
        return best_sample

    @staticmethod
    def _detection_candidates(bgr_image: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        height, width = bgr_image.shape[:2]
        candidates: list[tuple[np.ndarray, int, int]] = [(bgr_image, 0, 0)]
        if min(width, height) < 480:
            pad_x = max(16, int(width * 0.18))
            pad_y = max(16, int(height * 0.18))
            padded = cv2.copyMakeBorder(
                bgr_image,
                pad_y,
                pad_y,
                pad_x,
                pad_x,
                cv2.BORDER_REPLICATE,
            )
            candidates.append((padded, pad_x, pad_y))
        return candidates

    @staticmethod
    def _face_bbox_in_original_frame(
        face: np.ndarray,
        frame_width: int,
        frame_height: int,
        pad_x: int,
        pad_y: int,
    ) -> Optional[tuple[int, int, int, int]]:
        x, y, w_box, h_box = [int(round(v)) for v in face[:4]]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_width, x + w_box - pad_x)
        y2 = min(frame_height, y + h_box - pad_y)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _pick_best_face(faces: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        frame_cx = frame_width * 0.5
        frame_cy = frame_height * 0.5
        best_score = -1.0
        best_face = faces[0]
        frame_area = float(max(1, frame_width * frame_height))
        for face in faces:
            x, y, w_box, h_box = [float(v) for v in face[:4]]
            det_score = float(face[-1])
            area_ratio = (w_box * h_box) / frame_area
            center_x = x + (w_box * 0.5)
            center_y = y + (h_box * 0.5)
            center_dist = np.hypot(center_x - frame_cx, center_y - frame_cy)
            max_dist = max(1.0, np.hypot(frame_cx, frame_cy))
            center_bonus = 1.0 - min(1.0, center_dist / max_dist)
            score = (det_score * 0.65) + (min(1.0, area_ratio * 6.0) * 0.25) + (center_bonus * 0.10)
            if score > best_score:
                best_score = score
                best_face = face
        return best_face

    @staticmethod
    def _estimate_quality(
        aligned_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_size: tuple[int, int],
        detection_score: float,
    ) -> float:
        if aligned_bgr is None or aligned_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        width, height = frame_size
        bx1, by1, bx2, by2 = bbox
        area_ratio = ((bx2 - bx1) * (by2 - by1)) / float(max(1, width * height))
        return float(
            max(
                0.0,
                min(
                    1.0,
                    (detection_score * 0.45)
                    + (min(1.0, area_ratio * 7.0) * 0.30)
                    + (min(1.0, sharpness / 900.0) * 0.25),
                ),
            )
        )


def encode_face_png(face_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", face_bgr)
    return encoded.tobytes() if ok else b""


def decode_face_png(blob: bytes) -> Optional[np.ndarray]:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image if image is not None and image.size > 0 else None
