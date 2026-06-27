"""Getting-started guide for new CV Ops users.

A small, self-contained read-only widget describing what CV Ops is, the
first-run workflow (Quick start), the core ideas (Key concepts), and a one-line
reference for each top-nav tab. Rendered as themed HTML in a ``QTextBrowser`` so
it can live anywhere a widget fits (the Overview area, above Live Activity).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QTextBrowser, QWidget

from .cvops_theme import cvops_color, cvops_rgba


def getting_started_html() -> str:
    bright = cvops_color("text_bright")
    signal = cvops_color("text_signal")
    accent = cvops_color("accent_active")
    rule = cvops_rgba("line_light", 0.16)
    chip = cvops_rgba("bg_panel", 0.35)

    def step(num: int, title: str, body: str) -> str:
        return (
            f'<table cellspacing="0" cellpadding="0" width="100%" '
            f'style="margin:0 0 8px 0;"><tr>'
            f'<td width="26" valign="top" style="color:{accent}; font-weight:700; '
            f'font-size:13px;">{num:02d}</td>'
            f'<td style="color:{bright};"><b>{title}</b>'
            f'<div style="color:{signal}; margin-top:2px;">{body}</div></td>'
            f"</tr></table>"
        )

    def two_col(name: str, desc: str) -> str:
        return (
            f"<tr>"
            f'<td valign="top" style="padding:3px 10px 3px 0; color:{accent}; '
            f'font-weight:600; white-space:nowrap;">{name}</td>'
            f'<td style="padding:3px 0; color:{signal};">{desc}</td>'
            f"</tr>"
        )

    def heading(text: str) -> str:
        return (
            f'<div style="font-size:13px; font-weight:700; color:{bright}; '
            f'margin:0 0 6px 0;">{text}</div>'
        )

    def divider() -> str:
        return f'<div style="border-top:1px solid {rule}; margin:12px 0 10px 0;"></div>'

    steps = "".join(
        [
            step(
                1,
                "Create a scenario",
                "Open the <b>Train</b> tab and add a new scenario. Pick a backbone "
                "type for your problem — <b>YOLO detection</b> (boxes), <b>image "
                "classification</b> (folder-per-class), <b>tabular</b> (CSV rows), "
                "<b>custom code</b> cells, <b>LLM fine-tune</b>, or <b>archival "
                "ingestion</b>. A scenario is the unit of work: it bundles your "
                "data, model, hyperparameters, and every run together.",
            ),
            step(
                2,
                "Add your data",
                "In <b>Collect &amp; Edit</b> you can upload &amp; label images, "
                "upload a tabular CSV, or use <b>Semantic Carve</b> — point at a "
                "folder of images, type what you want (\"schools\", \"orchards\"), "
                "and it carves a labeled dataset by meaning, no manual boxing. You "
                "can also attach an existing dataset from the Train tab's "
                "<b>Dataset Readiness</b> card.",
            ),
            step(
                3,
                "Set model &amp; guard",
                "Back in <b>Train</b>, choose a base model and tune the "
                "<b>Hyperparameters</b> suite (learning rate, augmentation, "
                "quality-stop, …) — all pre-wired, in one panel. Review the "
                "<b>System &amp; Guard</b> card: it auto-picks a safe device, "
                "storage root, and profile so a run won't exhaust your machine.",
            ),
            step(
                4,
                "Start training",
                "Click <b>Start Training</b> and watch the live console, loss, and "
                "accuracy curves stream in. <b>Stop</b> any time; finished runs are "
                "versioned and saved automatically.",
            ),
            step(
                5,
                "Review, compare &amp; promote",
                "Inspect outputs in <b>Data Viz</b>, <b>Model Gallery</b>, and "
                "<b>Run Comparison</b>; the <b>Flow</b> view draws the whole "
                "scenario as a progressive diagram. When a run passes its CI/CD "
                "gate it is <b>staged</b> — promote the challenger to <b>prod</b>, "
                "or <b>revert</b> in one click from the Model Lifecycle bar.",
            ),
            step(
                6,
                "See it run on video",
                "Open <b>Scope</b> to point your trained model at a live <b>camera</b> "
                "or a <b>pre-recorded video</b> and watch detections in real time — "
                "the proof your model works on real footage.",
            ),
        ]
    )

    concepts = "".join(
        [
            two_col("Scenario", "The unit of work — one problem, with its data, model, hyperparameters, and run history."),
            two_col("Backbone types", "YOLO detection, image classification, tabular, custom code, LLM fine-tune, archival ingestion."),
            two_col("Semantic Carve", "Turn any image folder into a labeled dataset by typing a query (CLIP-ranked) — in Collect &amp; Edit and the Archive page."),
            two_col("System &amp; Guard", "Auto-selects a safe device, storage, and resource profile so training stays within your machine's limits."),
            two_col("Model lifecycle", "Runs become versions: candidate → staging (gate passed) → prod. One-click promote and revert."),
            two_col("Provenance", "Every dataset and model carries W3C-PROV lineage — what came from what, fully traceable."),
        ]
    )

    tabs = "".join(
        [
            two_col("Ecosystem", "Live graph of scenarios, datasets, models, and runs and how they connect."),
            two_col("Collect &amp; Edit", "Onboard data: label images, edit tabular CSVs, or Semantic Carve a dataset from a folder."),
            two_col("Train", "The hub — create scenarios, set hyperparameters, train, review, and promote models."),
            two_col("Range", "Evaluate and range-test trained models across thresholds and inputs."),
            two_col("Queue", "Watch and manage queued and running training/ingestion jobs."),
            two_col("Database", "Browse the underlying catalog, datasets, snapshots, and job records."),
            two_col("Data Viz", "Visualization surfaces for datasets and runs, plus the scenario Flow diagram."),
            two_col("Notes", "Project notes plus the local AI assistant (chat, RAG, dictation, read-aloud)."),
            two_col("Cells", "Author custom training code cells for advanced or non-standard flows."),
            two_col("3D", "3D viewing surface for spatial data and models."),
            two_col("Scope", "<b>Model viewer</b> — run a trained model on a live camera or a pre-recorded video to watch it work."),
            two_col("Notifications", "Run alerts, gate results, and system messages."),
            two_col("Settings", "Appearance, runtime, dashboard, storage — and this guide."),
        ]
    )

    return (
        f'<div style="font-family:-apple-system,Segoe UI,sans-serif; font-size:12px; '
        f'color:{bright};">'
        f'<div style="color:{signal}; margin-bottom:10px;">Welcome to CV Ops — a '
        f"<b>local-first</b> workspace that takes you from raw data to a trained, "
        f"tracked, and deployable model without leaving your machine. Collect data, "
        f"train vision / tabular / LLM models, version every run with full "
        f"provenance, and watch your model run on real video. Follow the steps "
        f"below for your first model.</div>"
        f"{heading('Quick start')}"
        f"{steps}"
        f"{divider()}"
        f"{heading('Key concepts')}"
        f'<table cellspacing="0" cellpadding="0" width="100%">{concepts}</table>'
        f"{divider()}"
        f"{heading('Tab reference')}"
        f'<table cellspacing="0" cellpadding="0" width="100%">{tabs}</table>'
        f"{divider()}"
        f'<div style="background:{chip}; padding:8px 10px; color:{signal};">'
        f'<b style="color:{bright};">Tip:</b> Stuck? Ask the in-app assistant in the '
        f"<b>Notes</b> tab — it can explain any tab or walk you through a workflow. "
        f"New models are proven in <b>Scope</b> (camera or video) and shipped via the "
        f"Model Lifecycle in <b>Train</b>."
        f"</div>"
        f"</div>"
    )


class GettingStartedGuide(QTextBrowser):
    """Themed, read-only getting-started guide widget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsGuideBrowser")
        self.setOpenExternalLinks(True)
        self.setMinimumHeight(340)
        self.setStyleSheet(
            f"QTextBrowser#cvOpsGuideBrowser {{ background-color: {cvops_color('bg_void')}; "
            f"border: 1px solid {cvops_rgba('line_light', 0.16)}; border-radius: 6px; "
            f"padding: 6px; }}"
        )
        self.setHtml(getting_started_html())
