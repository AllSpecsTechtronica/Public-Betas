from __future__ import annotations

from pathlib import Path
from typing import Any


_VIDEO_TEST_MODEL_SUFFIXES = {
    ".pt",
    ".torchscript",
    ".onnx",
    ".engine",
    ".mlmodel",
    ".mlpackage",
    ".tflite",
}
_YUNET_SCORE_THRESHOLD = 0.75
_YUNET_NMS_THRESHOLD = 0.30
_YUNET_TOP_K = 5000

# COCO 17-keypoint skeleton pairs (0-indexed) used for pose rendering
COCO_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9),                             # left arm
    (6, 8), (8, 10),                            # right arm
    (5, 11), (6, 12),                           # torso sides
    (11, 12),                                   # hips
    (11, 13), (13, 15),                         # left leg
    (12, 14), (14, 16),                         # right leg
]


def is_yunet_face_detector_model(model_path: str | Path) -> bool:
    name = Path(model_path).name.lower()
    return name.startswith("face_detection_yunet") and name.endswith(".onnx")


def is_sface_face_recognizer_model(model_path: str | Path) -> bool:
    name = Path(model_path).name.lower()
    return name.startswith("face_recognition_sface") and name.endswith(".onnx")


def is_supported_video_test_model(model_path: str | Path) -> bool:
    path = Path(model_path)
    suffix = path.suffix.lower()
    if suffix not in _VIDEO_TEST_MODEL_SUFFIXES:
        return False
    if is_sface_face_recognizer_model(path):
        return False
    if suffix == ".mlpackage":
        return path.is_dir()
    return path.is_file()


def _model_type_hint(model_path: str | Path) -> str:
    """Return 'pose', 'seg', or 'det' based on filename conventions."""
    name = Path(model_path).stem.lower()
    if "-pose" in name or "_pose" in name or name.endswith("pose"):
        return "pose"
    if "-seg" in name or "_seg" in name or name.endswith("seg"):
        return "seg"
    return "det"


# ---------------------------------------------------------------------------
# YOLO result extractors
# ---------------------------------------------------------------------------

def extract_yolo_detections(results: Any, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for res in results or []:
        boxes = getattr(res, "boxes", None)
        names = getattr(res, "names", {}) or {}
        if boxes is None:
            continue
        cls_t = getattr(boxes, "cls", None)
        xy_t = getattr(boxes, "xyxy", None)
        cf_t = getattr(boxes, "conf", None)
        if cls_t is None or xy_t is None:
            continue
        try:
            cls_list = cls_t.tolist()
            xy_list = xy_t.tolist()
            cf_list = cf_t.tolist() if cf_t is not None else [0.0] * len(cls_list)
        except Exception:
            continue
        for i, cls_id in enumerate(cls_list):
            if i >= len(xy_list):
                break
            x1, y1, x2, y2 = xy_list[i]
            out.append(
                {
                    "type": "det",
                    "label": str(names.get(int(cls_id), int(cls_id))),
                    "conf": float(cf_list[i]) if i < len(cf_list) else 0.0,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "frame_w": int(frame_w),
                    "frame_h": int(frame_h),
                }
            )
    return out


def extract_yolo_pose_detections(results: Any, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
    """Extract pose estimations from ultralytics YOLO-pose results.

    Each dict has the standard bbox fields plus:
      keypoints: list of [x, y, conf] per joint (COCO 17-kp order)
    """
    out: list[dict[str, Any]] = []
    for res in results or []:
        boxes = getattr(res, "boxes", None)
        kp_obj = getattr(res, "keypoints", None)
        names = getattr(res, "names", {}) or {}
        if boxes is None:
            continue
        xy_t = getattr(boxes, "xyxy", None)
        cf_t = getattr(boxes, "conf", None)
        cls_t = getattr(boxes, "cls", None)
        if xy_t is None:
            continue
        try:
            xy_list = xy_t.tolist()
            cf_list = cf_t.tolist() if cf_t is not None else [0.0] * len(xy_list)
            cls_list = cls_t.tolist() if cls_t is not None else [0] * len(xy_list)
        except Exception:
            continue

        # keypoints tensor: shape (N, K, 3) → [x, y, conf] per joint
        kp_list: list[list[list[float]]] = []
        if kp_obj is not None:
            kp_data = getattr(kp_obj, "data", None)
            if kp_data is not None:
                try:
                    kp_list = kp_data.tolist()
                except Exception:
                    pass
            if not kp_list:
                kp_xy = getattr(kp_obj, "xy", None)
                kp_conf = getattr(kp_obj, "conf", None)
                if kp_xy is not None:
                    try:
                        xy_kp = kp_xy.tolist()
                        conf_kp = kp_conf.tolist() if kp_conf is not None else None
                        for pi, person_xy in enumerate(xy_kp):
                            row: list[list[float]] = []
                            for ki, (kx, ky) in enumerate(person_xy):
                                kc = conf_kp[pi][ki] if conf_kp else 1.0
                                row.append([kx, ky, kc])
                            kp_list.append(row)
                    except Exception:
                        pass

        for i, (x1, y1, x2, y2) in enumerate(xy_list):
            kps: list[list[float]] = kp_list[i] if i < len(kp_list) else []
            label = str(names.get(int(cls_list[i]), "person")) if cls_list else "person"
            out.append(
                {
                    "type": "pose",
                    "label": label,
                    "conf": float(cf_list[i]) if i < len(cf_list) else 0.0,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "keypoints": kps,   # [[x, y, conf], ...]
                    "frame_w": int(frame_w),
                    "frame_h": int(frame_h),
                }
            )
    return out


def extract_yolo_seg_detections(results: Any, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
    """Extract segmentation results from ultralytics YOLO-seg results.

    Each dict has the standard bbox fields plus:
      mask_xy: list of [x, y] polygon contour points (normalized 0-1 coords)
    """
    out: list[dict[str, Any]] = []
    for res in results or []:
        boxes = getattr(res, "boxes", None)
        masks = getattr(res, "masks", None)
        names = getattr(res, "names", {}) or {}
        if boxes is None:
            continue
        xy_t = getattr(boxes, "xyxy", None)
        cf_t = getattr(boxes, "conf", None)
        cls_t = getattr(boxes, "cls", None)
        if xy_t is None:
            continue
        try:
            xy_list = xy_t.tolist()
            cf_list = cf_t.tolist() if cf_t is not None else [0.0] * len(xy_list)
            cls_list = cls_t.tolist() if cls_t is not None else [0] * len(xy_list)
        except Exception:
            continue

        # masks.xy: list of (N, 2) arrays, pixel coords in original image space
        mask_contours: list[list[list[float]]] = []
        if masks is not None:
            raw_xy = getattr(masks, "xy", None)
            if raw_xy is not None:
                try:
                    for contour in raw_xy:
                        if hasattr(contour, "tolist"):
                            mask_contours.append([[float(p[0]), float(p[1])] for p in contour.tolist()])
                        elif isinstance(contour, (list, tuple)):
                            mask_contours.append([[float(p[0]), float(p[1])] for p in contour])
                except Exception:
                    pass

        for i, (x1, y1, x2, y2) in enumerate(xy_list):
            contour: list[list[float]] = mask_contours[i] if i < len(mask_contours) else []
            label = str(names.get(int(cls_list[i]), int(cls_list[i]))) if cls_list else ""
            out.append(
                {
                    "type": "seg",
                    "label": label,
                    "conf": float(cf_list[i]) if i < len(cf_list) else 0.0,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "mask_xy": contour,   # [[x, y], ...] pixel coords in frame space
                    "frame_w": int(frame_w),
                    "frame_h": int(frame_h),
                }
            )
    return out


# ---------------------------------------------------------------------------
# ONNX inference backend (pose, seg, det)
# ---------------------------------------------------------------------------

class OnnxInferenceBackend:
    """Generic ONNX Runtime inference wrapper.

    Runs the session, interprets outputs by shape/name heuristics, and returns
    the same dict format as the YOLO extractors above.
    """

    # Confidence threshold for filtering raw ONNX outputs
    _CONF_THRESH = 0.25

    def __init__(self, model_path: str | Path, model_type: str = "det") -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed") from exc
        self._ort = ort
        self._model_type = model_type  # "det" | "pose" | "seg"
        resolved = Path(model_path).expanduser().resolve()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider"] + providers
        except Exception:
            pass
        self._session = ort.InferenceSession(str(resolved), sess_options=opts, providers=providers)
        inp = self._session.get_inputs()[0]
        self._input_name: str = inp.name
        self._input_shape: list = list(inp.shape)

    def predict(self, frame_bgr: Any) -> list[dict[str, Any]]:
        import numpy as np  # type: ignore
        h, w = frame_bgr.shape[:2]
        blob = self._preprocess(frame_bgr)
        outputs = self._session.run(None, {self._input_name: blob})
        if self._model_type == "pose":
            return self._parse_pose(outputs, w, h)
        if self._model_type == "seg":
            return self._parse_seg(outputs, w, h)
        return self._parse_det(outputs, w, h)

    def _preprocess(self, frame_bgr: Any) -> Any:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        # Determine target size from input shape [N, C, H, W] or [N, H, W, C]
        shape = self._input_shape
        if len(shape) == 4:
            if shape[1] == 3:     # NCHW
                tgt_h, tgt_w = int(shape[2]) if shape[2] > 0 else 640, int(shape[3]) if shape[3] > 0 else 640
            else:                 # NHWC
                tgt_h, tgt_w = int(shape[1]) if shape[1] > 0 else 640, int(shape[2]) if shape[2] > 0 else 640
        else:
            tgt_h, tgt_w = 640, 640

        resized = cv2.resize(frame_bgr, (tgt_w, tgt_h))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        # NCHW layout
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]
        return blob

    # ------------------------------------------------------------------
    # Output parsers
    # ------------------------------------------------------------------

    def _parse_det(self, outputs: list, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
        """Parse generic YOLO-style detection ONNX output.

        Supports two common layouts:
          - Shape (1, 5+C, N): transposed ultralytics export
          - Shape (1, N, 5+C): standard [x_c, y_c, w, h, conf, cls...]
        """
        import numpy as np  # type: ignore
        if not outputs:
            return []
        raw = np.array(outputs[0]).squeeze()  # remove batch dim
        if raw.ndim != 2:
            return []
        # ultralytics exports as (4+C, N) — transpose to (N, 4+C)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        out: list[dict[str, Any]] = []
        shape = self._input_shape
        tgt_w = int(shape[3]) if len(shape) == 4 and shape[1] == 3 and shape[3] > 0 else 640
        tgt_h = int(shape[2]) if len(shape) == 4 and shape[1] == 3 and shape[2] > 0 else 640
        for row in raw:
            if len(row) < 5:
                continue
            xc, yc, bw, bh = row[0], row[1], row[2], row[3]
            scores = row[4:]
            conf = float(np.max(scores))
            cls_id = int(np.argmax(scores))
            if conf < self._CONF_THRESH:
                continue
            # scale from model input resolution to original frame
            x1 = (xc - bw / 2) * frame_w / tgt_w
            y1 = (yc - bh / 2) * frame_h / tgt_h
            x2 = (xc + bw / 2) * frame_w / tgt_w
            y2 = (yc + bh / 2) * frame_h / tgt_h
            out.append({
                "type": "det",
                "label": str(cls_id),
                "conf": conf,
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
            })
        return out

    def _parse_pose(self, outputs: list, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
        """Parse YOLO-pose ONNX output.

        YOLOv8-pose exports shape (1, 56, N) or (1, N, 56):
          [xc, yc, w, h, conf, kp0_x, kp0_y, kp0_v, ..., kp16_x, kp16_y, kp16_v]
        """
        import numpy as np  # type: ignore
        if not outputs:
            return []
        raw = np.array(outputs[0]).squeeze()
        if raw.ndim != 2:
            return []
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        shape = self._input_shape
        tgt_w = int(shape[3]) if len(shape) == 4 and shape[1] == 3 and shape[3] > 0 else 640
        tgt_h = int(shape[2]) if len(shape) == 4 and shape[1] == 3 and shape[2] > 0 else 640
        out: list[dict[str, Any]] = []
        for row in raw:
            if len(row) < 5:
                continue
            xc, yc, bw, bh, conf = row[0], row[1], row[2], row[3], row[4]
            if float(conf) < self._CONF_THRESH:
                continue
            x1 = (xc - bw / 2) * frame_w / tgt_w
            y1 = (yc - bh / 2) * frame_h / tgt_h
            x2 = (xc + bw / 2) * frame_w / tgt_w
            y2 = (yc + bh / 2) * frame_h / tgt_h
            kps: list[list[float]] = []
            kp_data = row[5:]
            # groups of 3: [x, y, visibility]
            for ki in range(0, len(kp_data) - 2, 3):
                kx = float(kp_data[ki]) * frame_w / tgt_w
                ky = float(kp_data[ki + 1]) * frame_h / tgt_h
                kv = float(kp_data[ki + 2])
                kps.append([kx, ky, kv])
            out.append({
                "type": "pose",
                "label": "person",
                "conf": float(conf),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "keypoints": kps,
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
            })
        return out

    def _parse_seg(self, outputs: list, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
        """Parse YOLO-seg ONNX output.

        YOLOv8-seg exports two tensors:
          output0: (1, 4+C+32, N) — box + cls + mask coefficients
          output1: (1, 32, Mh, Mw) — prototype masks
        We decode a simplified polygon from the prototype * coefficients product.
        """
        import numpy as np  # type: ignore
        if not outputs:
            return []
        raw = np.array(outputs[0]).squeeze()
        if raw.ndim != 2:
            return self._parse_det([outputs[0]], frame_w, frame_h)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        protos = np.array(outputs[1]).squeeze() if len(outputs) > 1 else None

        shape = self._input_shape
        tgt_w = int(shape[3]) if len(shape) == 4 and shape[1] == 3 and shape[3] > 0 else 640
        tgt_h = int(shape[2]) if len(shape) == 4 and shape[1] == 3 and shape[2] > 0 else 640
        out: list[dict[str, Any]] = []
        num_classes = raw.shape[1] - 4 - (32 if protos is not None else 0)
        num_classes = max(1, num_classes)
        for row in raw:
            if len(row) < 5:
                continue
            xc, yc, bw, bh = row[0], row[1], row[2], row[3]
            cls_scores = row[4:4 + num_classes]
            conf = float(np.max(cls_scores))
            cls_id = int(np.argmax(cls_scores))
            if conf < self._CONF_THRESH:
                continue
            x1 = (xc - bw / 2) * frame_w / tgt_w
            y1 = (yc - bh / 2) * frame_h / tgt_h
            x2 = (xc + bw / 2) * frame_w / tgt_w
            y2 = (yc + bh / 2) * frame_h / tgt_h

            mask_xy: list[list[float]] = []
            if protos is not None and len(row) >= 4 + num_classes + 32:
                coeffs = row[4 + num_classes: 4 + num_classes + 32]
                try:
                    # protos shape: (32, Mh, Mw)
                    proto_h, proto_w = protos.shape[1], protos.shape[2]
                    mask = (coeffs[:, None, None] * protos).sum(axis=0)
                    mask = 1.0 / (1.0 + np.exp(-mask))  # sigmoid
                    # crop mask to bbox region and trace a simple bounding contour
                    bx1 = max(0, int(x1 * proto_w / frame_w))
                    by1 = max(0, int(y1 * proto_h / frame_h))
                    bx2 = min(proto_w, int(x2 * proto_w / frame_w))
                    by2 = min(proto_h, int(y2 * proto_h / frame_h))
                    patch = mask[by1:by2, bx1:bx2]
                    if patch.size > 0:
                        binary = (patch > 0.5).astype(np.uint8) * 255
                        try:
                            import cv2  # type: ignore
                            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                cnt = max(contours, key=lambda c: cv2.contourArea(c))
                                for pt in cnt[:, 0]:
                                    px = float(pt[0] + bx1) * frame_w / proto_w
                                    py = float(pt[1] + by1) * frame_h / proto_h
                                    mask_xy.append([px, py])
                        except Exception:
                            pass
                except Exception:
                    pass
            out.append({
                "type": "seg",
                "label": str(cls_id),
                "conf": conf,
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "mask_xy": mask_xy,
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
            })
        return out


class YuNetFaceDetectorBackend:
    """OpenCV YuNet adapter that returns the same detection dict shape as YOLO."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = _YUNET_SCORE_THRESHOLD,
        nms_threshold: float = _YUNET_NMS_THRESHOLD,
        top_k: int = _YUNET_TOP_K,
    ) -> None:
        import cv2  # type: ignore

        resolved = Path(model_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Face detector model not found: {resolved}")
        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN_create(
            str(resolved),
            "",
            (320, 320),
            score_threshold=float(score_threshold),
            nms_threshold=float(nms_threshold),
            top_k=int(top_k),
        )

    def predict(self, frame: Any) -> list[dict[str, Any]]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        height, width = frame.shape[:2]
        if width < 32 or height < 32:
            return []
        self._detector.setInputSize((int(width), int(height)))
        _ret, faces = self._detector.detect(frame)
        out: list[dict[str, Any]] = []
        if faces is None:
            return out
        for face in faces:
            try:
                x, y, w_box, h_box = [float(v) for v in face[:4]]
            except Exception:
                continue
            conf = float(face[-1]) if len(face) else 0.0
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + w_box)
            y2 = min(float(height), y + h_box)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                {
                    "label": "face",
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "frame_w": int(width),
                    "frame_h": int(height),
                }
            )
        out.sort(key=lambda det: float(det.get("conf", 0.0)), reverse=True)
        return out
