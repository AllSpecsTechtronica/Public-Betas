"""llm_fine_tuning.py - local LLM LoRA fine-tuning backbone for CV Ops."""
from __future__ import annotations

import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .. import registry as _reg
from ..backbone import BackboneBase, BackboneCell, BackboneContext, CellResult


def _resolve_repo_path(path_value: str) -> Path:
    raw = str(path_value or "").strip()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_reg.REPO_ROOT / p).resolve()


def _safe_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_reg.REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _next_run_dir(models_root: Path) -> Path:
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    if not runs:
        return models_root / "v1"
    latest = max(int(p.name[1:]) for p in runs)
    return models_root / f"v{latest + 1}"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def canonicalize_instruction_row(row: dict[str, Any], *, source: str) -> dict[str, str] | None:
    """Convert supported JSONL shapes into a single text sample."""
    if not isinstance(row, dict):
        return None
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        lines: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip() or "user"
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(str(part.get("text") if isinstance(part, dict) else part) for part in content)
            text = str(content or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        if lines:
            return {"text": "\n".join(lines).strip(), "source": source}

    prompt = str(row.get("prompt") or "").strip()
    response = str(row.get("response") or row.get("completion") or "").strip()
    if prompt and response:
        return {"text": f"user: {prompt}\nassistant: {response}", "source": source}

    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    output = str(row.get("output") or row.get("answer") or "").strip()
    if instruction and output:
        user = instruction if not input_text else f"{instruction}\n\n{input_text}"
        return {"text": f"user: {user}\nassistant: {output}", "source": source}
    return None


def _load_jsonl_examples(path: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    files: list[Path]
    root = path.resolve()
    if root.is_file():
        files = [root] if root.suffix.lower() == ".jsonl" else []
    else:
        files = [
            p for p in sorted(root.iterdir(), key=lambda x: x.name.lower())
            if p.is_file() and p.suffix.lower() == ".jsonl" and not p.name.startswith(".")
        ]
    examples: list[dict[str, str]] = []
    file_stats: list[dict[str, Any]] = []
    for file_path in files:
        seen = 0
        accepted = 0
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                seen += 1
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                item = canonicalize_instruction_row(obj, source=f"jsonl:{file_path.name}:{line_no}")
                if item is None:
                    continue
                accepted += 1
                examples.append(item)
        file_stats.append({
            "path": _safe_rel(file_path),
            "rows": seen,
            "accepted": accepted,
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        })
    return examples, file_stats


def _feedback_paths(cfg: Any) -> list[Path]:
    raw = cfg.backbone_config or {}
    paths: list[Path] = []
    explicit = str(raw.get("feedback_path") or "").strip()
    if explicit:
        paths.append(_resolve_repo_path(explicit))
    paths.extend([
        _reg.REPO_ROOT / "Insight" / "assets" / "model_feedback" / "console_model_feedback.jsonl",
        _reg.REPO_ROOT / "assets" / "model_feedback" / "console_model_feedback.jsonl",
    ])
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _load_feedback_examples(cfg: Any) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    examples: list[dict[str, str]] = []
    stats: list[dict[str, Any]] = []
    for path in _feedback_paths(cfg):
        if not path.is_file():
            continue
        seen = 0
        accepted = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                seen += 1
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                notes = str(obj.get("notes") or "").strip()
                rec = str(obj.get("recommendation") or "").strip()
                if not notes and not rec:
                    continue
                context = str(obj.get("context") or obj.get("scenario") or "").strip()
                issue = str(obj.get("issue_type") or "").strip()
                severity = str(obj.get("severity") or "").strip()
                prompt = (
                    "Use this operator feedback to improve local model behavior.\n"
                    f"Context: {context or 'unspecified'}\n"
                    f"Issue: {issue or 'unspecified'}\n"
                    f"Severity: {severity or 'unspecified'}\n"
                    f"Notes: {notes or rec}"
                )
                response = rec or notes
                examples.append({
                    "text": f"user: {prompt}\nassistant: {response}",
                    "source": f"feedback:{path.name}:{line_no}",
                })
                accepted += 1
        stats.append({"path": _safe_rel(path), "rows": seen, "accepted": accepted})
    return examples, stats


def load_training_examples(cfg: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    bcfg = cfg.backbone_config or {}
    sources = {s.lower() for s in (_as_list(bcfg.get("sources")) or ["jsonl", "feedback"])}
    examples: list[dict[str, str]] = []
    manifest: dict[str, Any] = {"sources": sorted(sources), "jsonl_files": [], "feedback_files": []}
    if "jsonl" in sources:
        jsonl_examples, jsonl_stats = _load_jsonl_examples(Path(str(cfg.dataset_path)))
        examples.extend(jsonl_examples)
        manifest["jsonl_files"] = jsonl_stats
    if "feedback" in sources:
        feedback_examples, feedback_stats = _load_feedback_examples(cfg)
        examples.extend(feedback_examples)
        manifest["feedback_files"] = feedback_stats
    return examples, manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_prepared_texts(path: Path) -> list[str]:
    texts: list[str] = []
    if not path.is_file():
        return texts
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            text = str(obj.get("text") or "").strip() if isinstance(obj, dict) else ""
            if text:
                texts.append(text)
    return texts


def _update_scenario_weights_yaml(config_path: Path, weights_ref: str) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {config_path}")
    raw["weights"] = weights_ref
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


class _PrepareInstructionDataCell(BackboneCell):
    name = "Prepare Instruction Data"
    description = "Load JSONL and feedback examples, then write prepared train/val splits"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        cfg = ctx.scenario_config
        examples, manifest = load_training_examples(cfg)
        if not examples:
            raise RuntimeError("No usable LLM fine-tuning examples found")
        models_root = (_reg.MLOPS_ROOT / "models" / cfg.name).resolve()
        models_root.mkdir(parents=True, exist_ok=True)
        run_dir = _next_run_dir(models_root)
        run_dir.mkdir(parents=True, exist_ok=False)

        val_count = max(1, int(math.ceil(len(examples) * 0.1))) if len(examples) >= 10 else 0
        train_rows = examples[:-val_count] if val_count else list(examples)
        val_rows = examples[-val_count:] if val_count else []
        train_path = run_dir / "prepared_train.jsonl"
        val_path = run_dir / "prepared_val.jsonl"
        manifest_path = run_dir / "dataset_manifest.json"
        _write_jsonl(train_path, train_rows)
        _write_jsonl(val_path, val_rows)
        manifest.update({
            "scenario": cfg.name,
            "dataset": str(cfg.dataset),
            "dataset_path": str(cfg.dataset_path),
            "example_count": len(examples),
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Prepared {len(train_rows)} train and {len(val_rows)} validation LLM example(s).")
        return CellResult(
            cell_name=self.name,
            status="done",
            output=f"prepared examples: train={len(train_rows)} val={len(val_rows)}",
            elapsed_ms=0,
            data={
                "run_dir": str(run_dir),
                "prepared_train": str(train_path),
                "prepared_val": str(val_path),
                "manifest": manifest,
                "train_count": len(train_rows),
                "val_count": len(val_rows),
            },
        )


class _TrainLoraAdapterCell(BackboneCell):
    name = "Train LoRA Adapter"
    description = "Fine-tune a PEFT LoRA adapter or create dry-run adapter artifacts"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        cfg = ctx.scenario_config
        prior = prev[-1].data if prev and isinstance(prev[-1].data, dict) else {}
        run_dir = Path(str(prior.get("run_dir") or ""))
        if not run_dir:
            raise RuntimeError("run_dir missing from data preparation cell")
        adapter_dir = run_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        train_path = Path(str(prior.get("prepared_train") or ""))
        val_path = Path(str(prior.get("prepared_val") or ""))
        train_texts = _read_prepared_texts(train_path)
        val_texts = _read_prepared_texts(val_path)
        bcfg = dict(cfg.backbone_config or {})
        dry_run = bool(bcfg.get("dry_run", False))
        base_model = str(bcfg.get("base_model") or cfg.base_model or "").strip()
        if not base_model:
            raise RuntimeError("backbone_config.base_model is required")

        started = time.time()
        if dry_run:
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({
                    "base_model_name_or_path": base_model,
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "dry_run": True,
                }, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            adapter_path = adapter_dir / "adapter_model.safetensors"
            adapter_path.write_bytes(b"CVOPS_LLM_DRY_RUN_ADAPTER" + b"\0" * 64)
            print("Dry-run enabled; wrote placeholder LoRA adapter artifacts.")
            return CellResult(
                cell_name=self.name,
                status="done",
                output=f"adapter: {_safe_rel(adapter_path)}",
                elapsed_ms=0,
                data={
                    "run_dir": str(run_dir),
                    "adapter_dir": str(adapter_dir),
                    "adapter_path": str(adapter_path),
                    "train_count": len(train_texts),
                    "val_count": len(val_texts),
                    "dry_run": True,
                    "train_runtime_s": round(time.time() - started, 3),
                },
            )

        missing = [
            pkg for pkg in ("torch", "transformers", "peft", "safetensors")
            if importlib.util.find_spec(pkg) is None
        ]
        if missing:
            raise RuntimeError(
                "LLM fine-tuning dependencies are missing: "
                + ", ".join(missing)
                + ". Install transformers, peft, torch, accelerate, and safetensors, or set dry_run=true."
            )
        if importlib.util.find_spec("accelerate") is None:
            raise RuntimeError("LLM fine-tuning requires accelerate for Transformers Trainer.")

        import torch  # type: ignore[import-not-found]
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        max_seq = int(bcfg.get("max_seq_length") or 1024)
        epochs = float(bcfg.get("epochs") or 1)
        batch_size = int(bcfg.get("batch_size") or 1)
        grad_accum = int(bcfg.get("gradient_accumulation_steps") or 4)
        learning_rate = float(bcfg.get("learning_rate") or 2e-4)
        local_files_only = bool(bcfg.get("local_files_only", False))
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=local_files_only)
        model.config.use_cache = False
        target_modules = _as_list(bcfg.get("target_modules")) or ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=int(bcfg.get("lora_r") or 8),
            lora_alpha=int(bcfg.get("lora_alpha") or 16),
            lora_dropout=float(bcfg.get("lora_dropout") or 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

        class _TextDataset(torch.utils.data.Dataset):  # type: ignore[name-defined]
            def __init__(self, texts: list[str]) -> None:
                self._texts = list(texts)

            def __len__(self) -> int:
                return len(self._texts)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return tokenizer(
                    self._texts[idx],
                    truncation=True,
                    max_length=max_seq,
                    padding=False,
                )

        args = TrainingArguments(
            output_dir=str(run_dir / "trainer"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": _TextDataset(train_texts),
            "eval_dataset": _TextDataset(val_texts) if val_texts else None,
            "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        }
        try:
            trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
        except TypeError:
            trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
        result = trainer.train()
        model.save_pretrained(str(adapter_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(run_dir / "tokenizer"))
        adapter_path = adapter_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise RuntimeError("PEFT training completed but adapter_model.safetensors was not written")
        print(f"Trained LoRA adapter from {len(train_texts)} example(s).")
        return CellResult(
            cell_name=self.name,
            status="done",
            output=f"adapter: {_safe_rel(adapter_path)}",
            elapsed_ms=0,
            data={
                "run_dir": str(run_dir),
                "adapter_dir": str(adapter_dir),
                "adapter_path": str(adapter_path),
                "train_count": len(train_texts),
                "val_count": len(val_texts),
                "dry_run": False,
                "train_runtime_s": round(time.time() - started, 3),
                "trainer_metrics": dict(getattr(result, "metrics", {}) or {}),
            },
        )


class _PackageOllamaModelfileCell(BackboneCell):
    name = "Package Ollama Modelfile"
    description = "Write Modelfile, metrics, and model registry entry"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        cfg = ctx.scenario_config
        merged: dict[str, Any] = {}
        for result in prev:
            if isinstance(result.data, dict):
                merged.update(result.data)
        run_dir = Path(str(merged.get("run_dir") or ""))
        adapter_path = Path(str(merged.get("adapter_path") or ""))
        if not run_dir or not adapter_path.is_file():
            raise RuntimeError("adapter artifact missing")
        bcfg = dict(cfg.backbone_config or {})
        ollama_base = str(bcfg.get("ollama_base_model") or bcfg.get("base_model") or cfg.base_model or "").strip()
        if not ollama_base:
            raise RuntimeError("ollama_base_model is required to write Modelfile")
        system = str(bcfg.get("system") or "").strip()
        modelfile = run_dir / "Modelfile"
        lines = [
            f"FROM {ollama_base}",
            "ADAPTER ./adapter",
        ]
        if system:
            lines.append(f'SYSTEM """{system}"""')
        lines.append("")
        modelfile.write_text("\n".join(lines), encoding="utf-8")

        adapter_ref = _safe_rel(adapter_path)
        _update_scenario_weights_yaml(Path(str(cfg.config_path)), adapter_ref)
        metrics = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "backbone_type": "llm_fine_tuning",
            "scenario": cfg.name,
            "base_model": str(bcfg.get("base_model") or cfg.base_model or ""),
            "ollama_base_model": ollama_base,
            "adapter_path": adapter_ref,
            "adapter_dir": _safe_rel(adapter_path.parent),
            "modelfile": _safe_rel(modelfile),
            "train_examples": int(merged.get("train_count") or 0),
            "val_examples": int(merged.get("val_count") or 0),
            "dry_run": bool(merged.get("dry_run", False)),
            "train_runtime_s": merged.get("train_runtime_s", 0),
            "metrics": {
                "task": "llm_fine_tuning",
                "train_examples": int(merged.get("train_count") or 0),
                "val_examples": int(merged.get("val_count") or 0),
                "dry_run": bool(merged.get("dry_run", False)),
            },
            "trainer_metrics": dict(merged.get("trainer_metrics") or {}),
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
        try:
            from .. import model_registry as _model_registry

            _model_registry.MODEL_REGISTRY_PATH = _reg.MLOPS_ROOT / "model_registry.json"
            _model_registry.register_model_version(
                scenario=cfg.name,
                run_version=run_dir.name,
                artifacts={
                    "run_dir": str(run_dir),
                    "weights": str(adapter_path),
                    "adapter_dir": str(adapter_path.parent),
                    "modelfile": str(modelfile),
                    "metrics_path": str(run_dir / "metrics.json"),
                },
                lineage={
                    "base_model": str(bcfg.get("base_model") or cfg.base_model or ""),
                    "dataset": str(cfg.dataset),
                    "dataset_path": str(cfg.dataset_path),
                    "sources": list((merged.get("manifest") or {}).get("sources") or []),
                },
                metrics={"task": "llm_fine_tuning", "train_examples": metrics["train_examples"], "dry_run": metrics["dry_run"]},
                set_candidate=True,
            )
        except Exception as exc:
            print(f"[WARN] model registry update failed: {exc}")
        summary = f"LLM adapter packaged for Ollama ({run_dir.name})"
        print(summary)
        return CellResult(
            cell_name=self.name,
            status="done",
            output=f"Modelfile: {_safe_rel(modelfile)}",
            elapsed_ms=0,
            data={
                "run_dir": str(run_dir),
                "result_path": str(run_dir),
                "weights": str(adapter_path),
                "model_version": run_dir.name,
                "signal": {"flag": False, "summary": summary, "metrics": metrics["metrics"]},
                "metrics": metrics,
            },
        )


class _LlmInferNotSupportedCell(BackboneCell):
    name = "Prompt Test Not Implemented"
    description = "LLM prompt-test inference is intentionally out of ViewPort for v1"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        msg = "llm_fine_tuning scenarios are train-only in CV Ops v1"
        print(msg)
        return CellResult(cell_name=self.name, status="error", output=msg, elapsed_ms=0)


class LlmFineTuningBackbone(BackboneBase):
    backbone_type = "llm_fine_tuning"

    def __init__(self, config: Any) -> None:
        self._config = config
        self._job_type = "infer"

    @property
    def cells(self) -> list[BackboneCell]:
        if self._job_type == "train":
            return [_PrepareInstructionDataCell(), _TrainLoraAdapterCell(), _PackageOllamaModelfileCell()]
        return [_LlmInferNotSupportedCell()]

    def run(self, ctx: BackboneContext) -> dict[str, Any]:
        self._job_type = ctx.job_type
        return super().run(ctx)

    def _build_result(self, ctx: BackboneContext, cell_results: list[CellResult]) -> dict[str, Any]:
        error = ""
        merged: dict[str, Any] = {}
        for result in cell_results:
            if isinstance(result.data, dict):
                merged.update(result.data)
            if result.status == "error" and not error:
                error = result.output or f"Cell '{result.cell_name}' failed"
        signal = merged.get("signal")
        if not isinstance(signal, dict):
            signal = {"flag": bool(error), "summary": error or "completed", "metrics": {}}
        return {
            "scenario": ctx.scenario_config.name,
            "model_version": str(merged.get("model_version") or ""),
            "weights": str(merged.get("weights") or ""),
            "summary": str(signal.get("summary") or ""),
            "detections": [],
            "elapsed_ms": sum(r.elapsed_ms for r in cell_results),
            "overlay_image": "",
            "signal": signal,
            "error": error,
            "artifact_policy": "path_only",
            "result_path": str(merged.get("result_path") or merged.get("run_dir") or ""),
            "backbone_data": {
                k: v for k, v in merged.items()
                if k not in {"signal", "weights", "model_version", "result_path"}
            },
        }
