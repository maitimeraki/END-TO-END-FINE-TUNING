# END-TO-END FINE-TUNING — Folder Guide (accurate)

This README documents the contents, conventions, and practical steps to run, develop, and promote experiments contained in this folder. It is written specifically for the repository layout present here.

Contents summary

- `api/` — Lightweight inference/test API and local serving entrypoint (`api/main.py`).
- `config.py` — Top-level training & runtime configuration used by notebooks and scripts.
- `docker/` — Example Dockerfiles for trainer and UI (`docker/api.Dockerfile`, `docker/ui.Dockerfile`).
- `cache/` & `unsloth_compiled_cache/` — compiled modules, temporary tokenizer artifacts and custom operators required for fast inference/training.
- `output/`, `outputs/`, `final_lora_adapter/` — training checkpoints, adapters and export destinations.
- `research/` — Jupyter notebooks for experiments and evaluation (`Reasoning_fine_tuning.ipynb`, `test_evaluation_inference.ipynb`).
- `src/` — project library code and trainer/service implementations used by notebooks.

Quick goals

- Run and reproduce an experiment from a notebook.
- Run the local `api` for quick validation of exported adapters.
- Export adapters/checkpoints into `final_lora_adapter/` for the inference service.

Environment setup

1. Create and activate a virtualenv matching the repository `requirements.txt`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

2. (Optional) Install GPU build of PyTorch that matches your CUDA driver; check `config.py` for version hints.

Running notebooks (recommended for reproducing experiments)

- Open the notebook(s) in `research/` or top-level notebooks with Jupyter or VS Code.
- Use small sample data or the repository's sample preprocessing to validate pipeline steps before full runs.

Data & preprocessing

- Place raw datasets under `END-TO-END-FINE-TUNE/data/` (create it if missing). The project expects tokenized/processed shards under a `processed/` subfolder when training at scale.
- If no preprocessing script exists, open the notebooks in `research/` and run the tokenization cells — they include the exact tokenizers and options used here.

Training / experiment workflow

1. Prepare data and config: update `config.py` or pass overrides from notebook/script.
2. Run the chosen trainer implementation in `src/service/` or the training cell in the notebook. Example (notebook-driven): open `research/Reasoning_fine_tuning.ipynb` and run cells sequentially.
3. Save artifacts to `output/` (intermediate) and `final_lora_adapter/` (exported adapters).

Example command patterns (adjust per your trainer implementation):

```powershell
# small local run (CPU or single GPU)
python -m src.service.train --config config.py --data data/processed --output output/run-001

# resume
python -m src.service.train --resume output/run-001
```

Exporting adapter/checkpoint for inference

- Use the project's export helper (if present in `src/`) or follow export cells in the notebook to produce a directory under `final_lora_adapter/`.
- Include a `manifest.json` alongside the exported adapter with fields: `version`, `base_model`, `tokenizer`, `commit`, `created_at`.

Local API for validation

- The folder contains an API entry in `api/main.py`. For local validation, prefer running with `uvicorn`:

```powershell
cd END-TO-END-FINE-TUNE
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

If `python -m api.main` fails, run via `uvicorn` to get better error trace and hot reload support.

Docker (trainer and UI)

- `docker/api.Dockerfile` and `docker/ui.Dockerfile` are provided as examples. For GPU training, use NVIDIA container runtime and CUDA base images.

Build & run example (trainer image):

```bash
docker build -f docker/api.Dockerfile -t etft-trainer:latest .
docker run --gpus all -v /path/to/data:/data etft-trainer:latest python -m src.service.train --config config.py
```

Observability & logs

- Training logs and artifacts are written to `output/` and `logs/mlartifacts/`.
- Keep `trainer_state.json` (already present in `outputs/`) with each run for resuming and audit.

Compiled code / caches

- `unsloth_compiled_cache/` contains compiled extensions and model-specific accelerated operators — do not delete if experiments depend on them.

Good practices specific to this folder

- Always run the small reproducibility notebook in `research/` before large experiments.
- Record the git commit and `config.py` copy in each `output/` run directory.
- Keep exported adapters in `final_lora_adapter/` and only point the inference service to stable versions.

Troubleshooting

- If import errors reference compiled modules, ensure Python version matches the compiled cache's build; consider rebuilding compiled extensions.
- If GPU training crashes, check CUDA / PyTorch compatibility and GPU memory usage; reduce batch size or use gradient accumulation.

Next steps I can take

- Add a `manifest.json` generator to export flows.
- Create a small wrapper script `run_local_training.sh` to standardize runs.
- Commit this README update and open a PR.

---

Maintainer: add your name and contact info here.
