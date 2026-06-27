# Dataset Database

This folder is the canonical on-disk location for datasets shown in the Insight CV Ops **Datasets** panel
and served by the CV Ops backend `/database` endpoints.

## Layout

Each dataset lives in its own directory under `database/`:

- `database/<dataset_name>/images/train/...`
- `database/<dataset_name>/images/val/...` (optional)
- `database/<dataset_name>/labels/train/...`
- `database/<dataset_name>/labels/val/...` (optional)

`data.yaml` is optional (training can generate `data.generated.yaml` when missing).

## Notes

- Datasets are typically large; by default this repo ignores `database/**` (except this README).
