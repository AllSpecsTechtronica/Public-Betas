"""Model Converter panel — converts .pth/.pt to ONNX or TorchScript.

Runs conversion in a QThread so the UI stays live. Output is saved into
assets/models/ alongside the existing registry weights so collect_video_test_models()
picks it up immediately.
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...ui.theme import text_css, theme_rgba
from ..detection_backends import is_supported_video_test_model

# Suffixes the panel will accept as drops. .pth / .pt go into the converter as
# source weights; the rest are already in a registry-runnable format and get
# copied straight into assets/models/.
_DROP_DIRECT_SUFFIXES = {".onnx", ".torchscript", ".engine", ".mlmodel", ".tflite", ".mlpackage"}
_DROP_CONVERT_SUFFIXES = {".pt", ".pth"}

_MODELS_DIR = Path(__file__).resolve().parents[4] / "assets" / "models"

_SOURCE_FORMATS = [".pth", ".pt"]
_TARGET_FORMATS = ["onnx", "torchscript"]

_INPUT_SHAPE_HELP = "e.g. 1,3,640,640"

# Sentinel object returned by _try_ultralytics_export to signal that the
# export + file copy was already completed inside that method, so the
# normal _export_onnx/_export_torchscript path should be skipped.
class _UltralyticsDone:
    pass
_ULTRALYTICS_HANDLED = _UltralyticsDone()


# ---------------------------------------------------------------------------
# Detection model wrapper for ONNX export
# ---------------------------------------------------------------------------

class _DetectionModelWrapper:
    """Wraps a torchvision detection model so torch.onnx.export can trace it.

    Torchvision detection models return list[dict[str,Tensor]] which the ONNX
    exporter cannot serialize. This wrapper runs in eval mode and flattens the
    output to (boxes, scores, labels) tensors — enough for downstream inference.
    """

    def __init__(self, model) -> None:
        self._model = model

    def __call__(self, images):
        import torch  # type: ignore
        outputs = self._model(images)
        if not outputs:
            return (
                torch.zeros(0, 4),
                torch.zeros(0),
                torch.zeros(0, dtype=torch.long),
            )
        det = outputs[0]
        return det.get("boxes", torch.zeros(0, 4)), \
               det.get("scores", torch.zeros(0)), \
               det.get("labels", torch.zeros(0, dtype=torch.long))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _ConvertWorker(QObject):
    finished = pyqtSignal(str)   # output path
    failed   = pyqtSignal(str)   # error message
    progress = pyqtSignal(str)   # status line updates

    def __init__(
        self,
        source_path: str,
        target_fmt: str,
        output_path: str,
        input_shape: tuple[int, ...],
        opset: int,
    ) -> None:
        super().__init__()
        self._source = source_path
        self._target_fmt = target_fmt
        self._output = output_path
        self._input_shape = input_shape
        self._opset = opset

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._convert()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    def _convert(self) -> None:
        import torch  # type: ignore

        src = Path(self._source)
        out = Path(self._output)
        self.progress.emit(f"Loading {src.name} …")

        checkpoint = torch.load(str(src), map_location="cpu", weights_only=False)
        model = self._extract_module(checkpoint, src)
        if model is None:
            return  # _extract_module already emitted failed
        if isinstance(model, _UltralyticsDone):
            return  # ultralytics export + finished signal already fired

        if self._target_fmt == "onnx":
            self._export_onnx(model, out)
        elif self._target_fmt == "torchscript":
            self._export_torchscript(model, out)
        else:
            self.failed.emit(f"Unknown target format: {self._target_fmt}")

    def _extract_module(self, checkpoint, src: Path):
        """Return an eval-mode nn.Module from whatever was saved, or emit failed."""
        import torch  # type: ignore

        # 1. Ultralytics full checkpoint — has a 'model' key with the nn.Module embedded
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.progress.emit("Detected ultralytics checkpoint — extracting model …")
            raw = checkpoint["model"]
            try:
                return raw.float().eval()
            except Exception:
                return raw

        # 2. Already an nn.Module (torch.save(model, path))
        if hasattr(checkpoint, "parameters"):
            self.progress.emit("Full nn.Module detected.")
            try:
                return checkpoint.float().eval()
            except Exception:
                return checkpoint

        # 3. State-dict dict — try to identify and reconstruct the architecture
        state_dict = None
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # bare state dict: all values are tensors
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint

        if state_dict is not None:
            self.progress.emit("State dict detected — identifying architecture …")
            model = self._reconstruct_from_state_dict(state_dict)
            if model is not None:
                return model
            # could not reconstruct — fall through to ultralytics attempt
            self.progress.emit("Architecture not auto-detected from state dict keys.")

        # 4. Ultralytics YOLO loader — handles .pt YOLO saves natively and exports
        #    directly; skip for .pth since ultralytics only accepts .pt
        if src.suffix.lower() == ".pt":
            self.progress.emit("Trying ultralytics YOLO loader …")
            result = self._try_ultralytics_export(src)
            if result is not None:
                return result  # sentinel: already exported, finished emitted
            # if None returned and running, means export path handled internally;
            # if failed was emitted there, we're done

        self.failed.emit(
            f"Could not load '{src.name}' as a usable nn.Module.\n\n"
            "Supported checkpoint shapes:\n"
            "  • ultralytics YOLO / YOLO-pose / YOLO-seg  (.pt)\n"
            "  • torch.save(model) full saves\n"
            "  • torchvision detection: RetinaNet / FasterRCNN / SSD state dicts\n"
            "  • torchvision segmentation: MaskRCNN state dicts\n"
            "  • torchvision pose: KeypointRCNN state dicts\n"
            "  • torchvision classification: ResNet / MobileNet / EfficientNet /\n"
            "    VGG / DenseNet / ViT state dicts\n\n"
            "For other architectures, export manually and add the .onnx to assets/models/."
        )
        return None

    def _reconstruct_from_state_dict(self, state_dict: dict):
        """Identify the architecture from state dict key patterns and reconstruct it."""
        keys = list(state_dict.keys())
        first = keys[0] if keys else ""

        try:
            import torchvision.models as tvm  # type: ignore
            import torchvision.models.detection as tvd  # type: ignore
        except ImportError:
            self.progress.emit("torchvision not available — cannot auto-reconstruct.")
            return None

        arch, constructor = self._identify_torchvision_arch(keys, first, tvm, tvd)
        if constructor is None:
            return None

        self.progress.emit(f"Identified architecture: {arch} — reconstructing …")
        try:
            model = constructor()
            # torchvision checkpoints may have keys prefixed with 'module.' (DataParallel)
            cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
            model.eval()
            return model
        except Exception as exc:
            self.progress.emit(f"Reconstruction failed for {arch}: {exc}")
            return None

    @staticmethod
    def _identify_torchvision_arch(keys, first, tvm, tvd):
        """Return (arch_name, constructor_callable) or (None, None)."""
        joined = " ".join(keys[:60])

        # ------------------------------------------------------------------
        # Pose estimation
        # ------------------------------------------------------------------
        # torchvision KeypointRCNN (COCO person keypoints)
        if "roi_heads.keypoint_predictor" in joined:
            return (
                "keypointrcnn_resnet50_fpn",
                lambda: tvd.keypointrcnn_resnet50_fpn(weights=None),
            )

        # ------------------------------------------------------------------
        # Detection models
        # ------------------------------------------------------------------
        if "backbone.body.layer1" in joined or "backbone.fpn" in joined:
            if "retinanet" in joined or "head.classification_head" in joined:
                return "retinanet_resnet50_fpn", lambda: tvd.retinanet_resnet50_fpn(weights=None)
            if "roi_heads" in joined:
                return "fasterrcnn_resnet50_fpn", lambda: tvd.fasterrcnn_resnet50_fpn(weights=None)
            if "ssd" in joined.lower():
                return "ssdlite320_mobilenet_v3_large", lambda: tvd.ssdlite320_mobilenet_v3_large(weights=None)

        # Mask RCNN (segmentation)
        if "roi_heads.mask_predictor" in joined:
            return (
                "maskrcnn_resnet50_fpn",
                lambda: tvd.maskrcnn_resnet50_fpn(weights=None),
            )

        # ------------------------------------------------------------------
        # Classification backbones
        # ------------------------------------------------------------------
        # ResNet family
        if first.startswith("layer1.") or "layer1.0.conv1" in joined:
            if "layer4.2.conv3" in joined:
                return "resnet50", lambda: tvm.resnet50(weights=None)
            if "layer4.0.conv1" in joined:
                return "resnet34", lambda: tvm.resnet34(weights=None)
            return "resnet18", lambda: tvm.resnet18(weights=None)

        # MobileNet
        if "features.0.0.weight" in joined and "classifier.1.weight" in joined:
            if "features.18" in joined:
                return "mobilenet_v2", lambda: tvm.mobilenet_v2(weights=None)
            return "mobilenet_v3_large", lambda: tvm.mobilenet_v3_large(weights=None)

        # EfficientNet
        if first.startswith("features.0.") and "features.7" in joined:
            return "efficientnet_b0", lambda: tvm.efficientnet_b0(weights=None)

        # VGG
        if first.startswith("features.0.weight") and "classifier.6" in joined:
            if len(keys) < 26:
                return "vgg16", lambda: tvm.vgg16(weights=None)
            return "vgg19", lambda: tvm.vgg19(weights=None)

        # DenseNet
        if "features.denseblock1" in joined:
            if "denseblock4.denselayer32" in joined:
                return "densenet201", lambda: tvm.densenet201(weights=None)
            return "densenet121", lambda: tvm.densenet121(weights=None)

        # ViT
        if "encoder.layers.encoder_layer_0" in joined:
            if len(keys) > 200:
                return "vit_l_16", lambda: tvm.vit_l_16(weights=None)
            return "vit_b_16", lambda: tvm.vit_b_16(weights=None)

        return None, None

    def _try_ultralytics_export(self, src: Path):
        """Run ultralytics export directly — returns _ULTRALYTICS_HANDLED sentinel or None on failure."""
        try:
            from ultralytics import YOLO  # type: ignore
            import shutil
            yolo = YOLO(str(src))
            out = Path(self._output)
            if self._target_fmt == "onnx":
                self.progress.emit("Exporting via ultralytics to ONNX …")
                exported = yolo.export(
                    format="onnx",
                    imgsz=list(self._input_shape[-2:]) if len(self._input_shape) >= 2 else [640, 640],
                    opset=self._opset,
                    dynamic=False,
                )
            else:
                self.progress.emit("Exporting via ultralytics to TorchScript …")
                exported = yolo.export(format="torchscript")
            exported_path = Path(str(exported))
            if exported_path.exists() and str(exported_path) != str(out):
                shutil.copy2(exported_path, out)
            self.finished.emit(str(out))
            return _ULTRALYTICS_HANDLED
        except Exception as exc:
            self.progress.emit(f"ultralytics loader failed: {exc}")
            return None

    def _export_onnx(self, model, out: Path) -> None:
        import torch  # type: ignore

        self.progress.emit(f"Exporting to ONNX (opset {self._opset}) …")
        self.progress.emit("Using legacy ONNX exporter (dynamo=False) …")
        shape = self._input_shape or (1, 3, 640, 640)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Torchvision detection models (RetinaNet, FasterRCNN, SSD…) expect a
        # list[Tensor] input and return list[dict] — torch.onnx.export can't
        # trace through the dict output directly.  We wrap the model to produce
        # a flat tuple of tensors that the ONNX exporter can handle.
        if self._is_torchvision_detection(model):
            self.progress.emit("Detection model detected — using wrapped export …")
            model = _DetectionModelWrapper(model)
            dummy = [torch.zeros(shape[1], shape[2], shape[3])]  # list[C,H,W]
            self._torch_onnx_export(
                torch,
                model,
                (dummy,),
                str(out),
                opset_version=self._opset,
                input_names=["images"],
                do_constant_folding=True,
            )
        else:
            dummy = torch.zeros(*shape)
            self._torch_onnx_export(
                torch,
                model,
                dummy,
                str(out),
                opset_version=self._opset,
                input_names=["images"],
                output_names=["output0"],
                dynamic_axes={
                    "images": {0: "batch"},
                    "output0": {0: "batch"},
                },
            )
        self.progress.emit("ONNX export complete.")
        self.finished.emit(str(out))

    def _torch_onnx_export(self, torch_module, *args, **kwargs) -> None:
        kwargs.setdefault("dynamo", False)
        try:
            torch_module.onnx.export(*args, **kwargs)
        except TypeError as exc:
            if "dynamo" not in str(exc):
                raise
            kwargs.pop("dynamo", None)
            torch_module.onnx.export(*args, **kwargs)

    @staticmethod
    def _is_torchvision_detection(model) -> bool:
        cls = type(model).__mro__
        names = {c.__name__ for c in cls}
        return bool(names & {"RetinaNet", "FasterRCNN", "FCOS", "SSD", "SSDLite320",
                             "MaskRCNN", "KeypointRCNN", "GeneralizedRCNN"})

    def _export_torchscript(self, model, out: Path) -> None:
        import torch  # type: ignore

        self.progress.emit("Tracing to TorchScript …")
        shape = self._input_shape or (1, 3, 640, 640)
        dummy = torch.zeros(*shape)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            traced = torch.jit.trace(model, dummy, strict=False)
        except Exception:
            self.progress.emit("Trace failed, trying torch.jit.script …")
            traced = torch.jit.script(model)
        traced.save(str(out))
        self.progress.emit("TorchScript export complete.")
        self.finished.emit(str(out))


# ---------------------------------------------------------------------------
# Session (thread lifecycle manager)
# ---------------------------------------------------------------------------

class _ConvertSession:
    def __init__(self) -> None:
        self._thread: Optional[QThread] = None
        self._worker: Optional[_ConvertWorker] = None

    def start(
        self,
        *,
        source_path: str,
        target_fmt: str,
        output_path: str,
        input_shape: tuple[int, ...],
        opset: int,
        on_finished,
        on_failed,
        on_progress,
    ) -> None:
        self.stop()
        thread = QThread()
        worker = _ConvertWorker(
            source_path=source_path,
            target_fmt=target_fmt,
            output_path=output_path,
            input_shape=input_shape,
            opset=opset,
        )
        worker.moveToThread(thread)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.progress.connect(on_progress)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def stop(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is None:
            return
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
        except RuntimeError:
            pass

    def is_running(self) -> bool:
        try:
            return self._thread is not None and self._thread.isRunning()
        except RuntimeError:
            return False


# ---------------------------------------------------------------------------
# Panel widget
# ---------------------------------------------------------------------------

class ModelConverterPanel(QWidget):
    """Tab panel: upload a .pth/.pt, pick an output format, convert and register."""

    modelRegistered = pyqtSignal(str)  # emitted with output path when done

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session = _ConvertSession()
        self._source_path = ""

        # Enable drag-and-drop for .pt / .pth (convert) and .onnx etc. (direct copy)
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # -- status bar --
        self._status = QLabel("Upload a .pth or .pt file to convert, or drop any supported model here.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"font-size: 10px; color: {text_css(0.84)};")
        root.addWidget(self._status)

        # -- drop hint --
        self._drop_hint = QLabel(
            "Drop .pt / .pth here to convert, or .onnx / .torchscript / .mlpackage / "
            ".engine / .tflite / .mlmodel to register directly."
        )
        self._drop_hint.setWordWrap(True)
        self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_hint_base_style = (
            f"font-size: 10px; color: {text_css(0.55)}; "
            f"border: 1px dashed {theme_rgba('accent_dark', 0.35)}; "
            f"background: {theme_rgba('panel', 0.25)}; padding: 6px; border-radius: 4px;"
        )
        self._drop_hint_active_style = (
            f"font-size: 10px; color: {text_css(0.95)}; "
            f"border: 1px dashed {theme_rgba('accent_dark', 0.9)}; "
            f"background: {theme_rgba('accent_dark', 0.25)}; padding: 6px; border-radius: 4px;"
        )
        self._drop_hint.setStyleSheet(self._drop_hint_base_style)
        root.addWidget(self._drop_hint)

        # -- source file --
        src_frame = QFrame()
        src_frame.setStyleSheet(
            f"QFrame {{ background: {theme_rgba('panel', 0.45)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.18)}; }}"
        )
        src_layout = QVBoxLayout(src_frame)
        src_layout.setContentsMargins(8, 8, 8, 8)
        src_layout.setSpacing(6)

        src_lbl = QLabel("Source weights")
        src_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; "
            f"color: {theme_rgba('accent_dark', 0.9)}; border: none; background: transparent;"
        )
        src_layout.addWidget(src_lbl)

        src_row = QHBoxLayout()
        self._source_label = QLabel("No file selected.")
        self._source_label.setStyleSheet(
            f"font-size: 10px; color: {text_css(0.6)}; border: none; background: transparent;"
        )
        self._source_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        src_row.addWidget(self._source_label, stretch=1)
        self._browse_btn = QPushButton("[BROWSE]")
        self._browse_btn.clicked.connect(self._on_browse)
        src_row.addWidget(self._browse_btn)
        src_layout.addLayout(src_row)
        root.addWidget(src_frame)

        # -- conversion options --
        opts_frame = QFrame()
        opts_frame.setStyleSheet(
            f"QFrame {{ background: {theme_rgba('panel', 0.45)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.18)}; }}"
        )
        opts_layout = QVBoxLayout(opts_frame)
        opts_layout.setContentsMargins(8, 8, 8, 8)
        opts_layout.setSpacing(6)

        opts_lbl = QLabel("Conversion options")
        opts_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; "
            f"color: {theme_rgba('accent_dark', 0.9)}; border: none; background: transparent;"
        )
        opts_layout.addWidget(opts_lbl)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Target format:"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItem("ONNX (.onnx)", userData="onnx")
        self._fmt_combo.addItem("TorchScript (.torchscript)", userData="torchscript")
        self._fmt_combo.currentIndexChanged.connect(self._on_fmt_changed)
        fmt_row.addWidget(self._fmt_combo, stretch=1)
        opts_layout.addLayout(fmt_row)

        shape_row = QHBoxLayout()
        shape_lbl = QLabel("Input shape:")
        shape_lbl.setToolTip("NCHW, comma-separated. Required for ONNX/TorchScript trace.")
        shape_row.addWidget(shape_lbl)
        self._shape_edit = QLineEdit()
        self._shape_edit.setPlaceholderText(_INPUT_SHAPE_HELP)
        self._shape_edit.setText("1,3,640,640")
        shape_row.addWidget(self._shape_edit, stretch=1)
        opts_layout.addLayout(shape_row)

        opset_row = QHBoxLayout()
        opset_row.addWidget(QLabel("ONNX opset:"))
        self._opset_spin = QSpinBox()
        self._opset_spin.setRange(9, 20)
        self._opset_spin.setValue(17)
        opset_row.addWidget(self._opset_spin)
        opset_row.addStretch(1)
        self._opset_row_widget = QWidget()
        self._opset_row_widget.setLayout(opset_row)
        opts_layout.addWidget(self._opset_row_widget)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Output name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("leave blank to auto-name")
        name_row.addWidget(self._name_edit, stretch=1)
        opts_layout.addLayout(name_row)

        root.addWidget(opts_frame)

        # -- convert button --
        self._convert_btn = QPushButton("[CONVERT + REGISTER]")
        self._convert_btn.setEnabled(False)
        self._convert_btn.clicked.connect(self._on_convert)
        root.addWidget(self._convert_btn)

        # -- log --
        log_lbl = QLabel("Log")
        log_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {text_css(0.5)}; border: none;"
        )
        root.addWidget(log_lbl)
        self._log = QLabel("")
        self._log.setWordWrap(True)
        self._log.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._log.setStyleSheet(
            f"font-size: 9px; color: {text_css(0.55)}; "
            f"background: {theme_rgba('panel', 0.3)}; border: none; padding: 4px;"
        )
        self._log.setMinimumHeight(60)
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._log, stretch=1)

        self._on_fmt_changed()

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def _drop_paths(self, event) -> list[Path]:
        """Return supported file paths from a drag event, in drop order."""
        md = event.mimeData()
        if not md.hasUrls():
            return []
        out: list[Path] = []
        allowed = _DROP_DIRECT_SUFFIXES | _DROP_CONVERT_SUFFIXES
        for url in md.urls():
            if not url.isLocalFile():
                continue
            p = Path(url.toLocalFile())
            suffix = p.suffix.lower()
            if suffix in allowed:
                # .mlpackage is a directory bundle, others must be files
                if suffix == ".mlpackage" and not p.is_dir():
                    continue
                if suffix != ".mlpackage" and not p.is_file():
                    continue
                out.append(p)
        return out

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if self._drop_paths(event):
            event.acceptProposedAction()
            self._drop_hint.setStyleSheet(self._drop_hint_active_style)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drop_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, _event) -> None:  # type: ignore[override]
        self._drop_hint.setStyleSheet(self._drop_hint_base_style)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        self._drop_hint.setStyleSheet(self._drop_hint_base_style)
        paths = self._drop_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()

        # Split by handling kind
        convert_targets = [p for p in paths if p.suffix.lower() in _DROP_CONVERT_SUFFIXES]
        direct_targets = [p for p in paths if p.suffix.lower() in _DROP_DIRECT_SUFFIXES]

        # Direct registration: copy each into assets/models/, emit modelRegistered
        registered = 0
        for src in direct_targets:
            try:
                self._register_direct(src)
                registered += 1
            except Exception as exc:
                self._append_log(f"[ERROR] {src.name}: {exc}")

        # Convert path: load the LAST .pt/.pth into the source field (single-source converter)
        if convert_targets:
            target = convert_targets[-1]
            self._set_source(target)
            extra = ""
            if len(convert_targets) > 1:
                extra = (
                    f" ({len(convert_targets) - 1} other .pt/.pth file(s) ignored — "
                    "drop one at a time to convert each)"
                )
            self._status.setText(f"Loaded: {target.name}{extra}")

        if registered and not convert_targets:
            self._status.setText(
                f"Registered {registered} model{'s' if registered != 1 else ''} into the catalog."
            )
        elif registered:
            self._status.setText(
                f"Registered {registered} model{'s' if registered != 1 else ''}; "
                f"source set to {convert_targets[-1].name}."
            )

    def _register_direct(self, src: Path) -> None:
        """Copy an already-runnable model file/bundle into assets/models/."""
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = _MODELS_DIR / src.name
        # If a file with this exact name already exists in the registry, leave it alone
        if dest.exists() and dest.resolve() == src.resolve():
            self._append_log(f"{src.name} is already in the registry — skipped.")
            self.modelRegistered.emit(str(dest))
            return
        if dest.exists():
            # Pick a non-clobbering name: stem-1.ext, stem-2.ext, ...
            n = 1
            while True:
                candidate = _MODELS_DIR / f"{src.stem}-{n}{src.suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                n += 1
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        # Sanity check: ensure the registry sees it
        if not is_supported_video_test_model(dest):
            self._append_log(f"[WARN] {dest.name} copied but registry does not recognize it.")
        else:
            self._append_log(f"Registered {dest.name} into the model catalog.")
        self.modelRegistered.emit(str(dest))

    def _set_source(self, p: Path) -> None:
        """Populate the source-weights field for a .pt/.pth (shared with browse)."""
        self._source_path = str(p)
        self._source_label.setText(p.name)
        if not self._name_edit.text().strip():
            self._name_edit.setText(p.stem)
        self._convert_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select source weights",
            "",
            "PyTorch weights (*.pth *.pt);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        self._set_source(p)
        self._status.setText(f"Loaded: {p.name}  ({p.stat().st_size // 1024} KB)")

    def _on_fmt_changed(self) -> None:
        fmt = self._fmt_combo.currentData() or "onnx"
        self._opset_row_widget.setVisible(fmt == "onnx")

    def _on_convert(self) -> None:
        if not self._source_path:
            self._status.setText("Select a source file first.")
            return
        if self._session.is_running():
            self._status.setText("Conversion already running…")
            return

        fmt = self._fmt_combo.currentData() or "onnx"
        ext = ".onnx" if fmt == "onnx" else ".torchscript"
        stem = self._name_edit.text().strip() or Path(self._source_path).stem
        # avoid overwriting the source
        out_name = stem if stem.endswith(ext) else stem + ext
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_MODELS_DIR / out_name)

        shape = self._parse_shape()
        if shape is None:
            self._status.setText(f"Invalid input shape — use format: {_INPUT_SHAPE_HELP}")
            return

        opset = self._opset_spin.value()
        self._convert_btn.setEnabled(False)
        self._log.setText("")
        self._status.setText("Converting…")

        self._session.start(
            source_path=self._source_path,
            target_fmt=fmt,
            output_path=output_path,
            input_shape=shape,
            opset=opset,
            on_finished=self._on_done,
            on_failed=self._on_failed,
            on_progress=self._on_progress,
        )

    def _on_done(self, output_path: str) -> None:
        p = Path(output_path)
        size_kb = p.stat().st_size // 1024 if p.exists() else 0
        self._status.setText(f"Done — {p.name}  ({size_kb} KB) added to registry.")
        self._append_log(f"[DONE] {output_path}")
        self._convert_btn.setEnabled(True)
        self.modelRegistered.emit(output_path)

    def _on_failed(self, message: str) -> None:
        self._status.setText("Conversion failed — see log.")
        self._append_log(f"[ERROR] {message}")
        self._convert_btn.setEnabled(True)

    def _on_progress(self, message: str) -> None:
        self._status.setText(message)
        self._append_log(message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_shape(self) -> Optional[tuple[int, ...]]:
        raw = self._shape_edit.text().strip()
        if not raw:
            return (1, 3, 640, 640)
        try:
            parts = tuple(int(x.strip()) for x in raw.split(","))
            if len(parts) < 2 or any(p <= 0 for p in parts):
                return None
            return parts
        except (ValueError, TypeError):
            return None

    def _append_log(self, line: str) -> None:
        current = self._log.text()
        lines = current.split("\n") if current else []
        lines.append(line)
        # keep last 20 lines
        self._log.setText("\n".join(lines[-20:]))
