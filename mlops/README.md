# CV MLOps Base

This directory contains scenario-driven CV runtime plumbing.

## Add a new runtime
1. Add a scenario config under `mlops/scenarios/`.
2. Add/update dataset under `mlops/datasets/<scenario>/`.
3. Train model: `python -m mlops.pipeline.train --scenario <name>`.
4. Add the scenario to `mlops/registry.json`.

No HUD Python changes are required for a new scenario.

## Dashboard (local)
For a technical, data-verbose dashboard (scenarios + datasets + artifacts):

- `python -m pip install -r mlops/dashboard/requirements.txt`
- `python -m streamlit run mlops/dashboard/app.py`
