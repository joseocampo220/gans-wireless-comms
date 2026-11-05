# GANs for Wireless Communications (MFTR)

This repository explores **generative adversarial networks (GANs)** for **wireless fading distributions**, with a focus on approximating the **Multi-Cluster Fluctuating Two-Ray (MFTR)** fading envelope distribution using a **conditional GAN (cGAN)**.

## What has been tried so far
- **Unconditional GANs** for simple 1D fading distributions (e.g., Rayleigh, Nakagami).
- A **continuous-condition cGAN** trained on synthetic MFTR samples to generate envelope samples conditioned on MFTR parameters (with experiments emphasizing a sweep over $\mu$).
- Evaluation focused on **interpolation vs extrapolation** (in-range vs out-of-range conditions) using:
	- QQ plots and density overlays
	- Kolmogorov–Smirnov statistic (KS)
	- 1D Wasserstein distance
	- Quantile MSE/MAE

## Thesis report
The approved thesis source is [report/main.tex](report/main.tex), with front matter
and numbered sections under `report/`. The tracked PDF is
[report/build/main.pdf](report/build/main.pdf). Figures are in
[report/images/](report/images/); the existing figure-generation scripts write there.

Build from `report/` with `latexmk -pdf -interaction=nonstopmode -file-line-error main.tex`.
The build requires pdfLaTeX, BibTeX, `IEEEtran.bst`, latexmk and the LaTeX packages
used by the document. It writes the PDF and SyncTeX data to `report/build/`.
See [report/README.md](report/README.md) for VS Code preview and cleanup details.

## Key code locations (MFTR conditional model)
- Core conditional model:
	- [src/cgans/cgan.py](src/cgans/cgan.py)
	- [src/cgans/components.py](src/cgans/components.py)
- MFTR conditional evaluation + metrics:
	- [src/cgans/mftr_evaluation.py](src/cgans/mftr_evaluation.py)
- MFTR conditional dataset utilities:
	- [src/local_datasets/mftr_uniform_conditional_dataset.py](src/local_datasets/mftr_uniform_conditional_dataset.py)

## References (useful starting points)
- MFTR model paper (arXiv): https://arxiv.org/abs/2212.02448
- GANs: https://arxiv.org/abs/1406.2661
- Conditional GANs: https://arxiv.org/abs/1411.1784
- 1D distributions with GANs (“GAN connoisseur”): NeurIPS 2017

Bibliography entries live in [report/references.bib](report/references.bib).

## Reproducing results (uses `uv`)
Create the environment and run commands via `uv` (see [pyproject.toml](pyproject.toml)).

### Run tests / lint
- `make test`
- `make lint`
- `make format`

### Train models (Makefile)
- Train MFTR cGAN (main experiment pattern):
	- `make train-cgan-mftr`
- Train unconditional baselines:
	- `make train-rayleigh`
	- `make train-nakagami`
	- `make train-mftr`

Training outputs/logs go under [logs/](logs/).

### Regenerate paper figures (no retraining)
- Conditional evaluation grids (interpolation/extrapolation) for version\_16:
	- `uv run python scripts/regenerate_conditional_eval_plots.py --version-dir logs/mftr/cgan/version_16 --keep 3`
- Training curves (losses + discriminator probabilities) for version\_16:
	- `uv run python scripts/plot_version16_training_curves.py`

## Interactive Gradio app
A lightweight interactive page is available to:
- summarize the project (goal, architecture, references, tried methods, next steps), and
- run conditional MFTR comparisons for curated trained versions.

Run it with :
- `make run` to run the app in production mode and create a public sharing url
- `make dev` to setup hot reloading

Then access the app locally in `localhost:7860`.

Main app modules:
- [src/app/main.py](src/app/main.py): UI layout and callbacks
- [src/app/artifacts.py](src/app/artifacts.py): model-version discovery and metadata cards
- [src/app/inference.py](src/app/inference.py): checkpoint loading, generation, and metrics
- [src/app/plots.py](src/app/plots.py): QQ and distribution comparison plots + diagrams
- [src/app/content.py](src/app/content.py): concise overview text blocks

## Results snapshot (from the latest MFTR cGAN run)
- In-range conditional generation (interpolation) matches the theoretical MFTR reference closely.
- Extrapolation beyond the training range is **less reliable**, with similar median errors but larger **worst-case** discrepancies at larger $\mu$.
- Loss/probability curves stabilize early, so they are not sufficient alone to assess final sample quality.
