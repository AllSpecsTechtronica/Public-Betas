# Image-to-3D Runtime Requirements

This pipeline is built for the existing Streamlit dashboard inside the Qt
`QWebEngineView` wrapper.

Install the additive dashboard requirements:

```bash
python -m pip install -r mlops/dashboard/requirements.txt
```

The depth stage uses `depth-anything/Depth-Anything-V2-Small-hf` through
Transformers. The model is downloaded lazily into the Hugging Face cache on the
first run. If the host is offline and the model is not already cached, the
pipeline fails at the depth stage and writes the failure into `provenance.json`.

On Apple Silicon, torch MPS is used when available. CPU is used as a fallback.
TRELLIS enhancement is optional and calls the existing hosted TRELLIS.2 client.
