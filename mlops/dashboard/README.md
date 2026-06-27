# CV MLOps Dashboard

Technical dashboard for scenario + dataset inspection. Optimized for a technical audience:
lots of structured tables and raw JSON/YAML payloads, minimal styling.

## Run

From repo root:

```bash
python -m pip install -r mlops/dashboard/requirements.txt
python -m streamlit run mlops/dashboard/app.py
```

## What it shows

- Scenarios from `mlops/registry.json` with derived status (dataset/trained/ready/error).
- Scenario details (config + artifacts) and dataset inspection.
- Dataset library inspection (`database/` + image datasets under `mlops/datasets/`).
- Tabular dataset inspection for CSV files under `mlops/datasets/`.
- Basic health checks (missing base models/weights, empty datasets, config errors).
