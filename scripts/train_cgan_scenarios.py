"""Training script for scenario-conditioned CGAN.

This script trains a CGAN using empirical scenario data from the standardized
`data/scenarios/` directory. It is a sibling to `train_cgan.py` and reuses the
same model, trainer, and logging infrastructure; the only difference is that
dataset construction is driven by selected scenario IDs and a sample-source
switch rather than a wide synthetic parameter grid.

Organization
1. `_save_training_details_scenarios`: saves training metadata to JSON, extended
   with scenario-specific fields (scenario IDs, source, params method).
2. `main`: parses args, builds the scenario-conditional dataset, constructs the
   val dataset (same scenarios, smaller sample count), and runs the Lightning
   trainer with the standard CGAN model.

Usage example
-------------
python scripts/train_cgan_scenarios.py \\
    --train_scenarios 1 2 3 4 5 \\
    --source empirical_pdf \\
    --samples_per_scenario 10000 \\
    --epochs 200 \\
    --logs_dir ../logs

The left-out scenario (e.g. 6) is not referenced at training time and can be
selected manually in the Gradio evaluation app for visual comparison.
"""

import json
import os
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from cgans import CGAN
from config import get_scenario_cgan_args, _parse_devices
from local_datasets import ScenarioConditionalDataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    """Recursively convert an object to a JSON-serialisable type."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _save_training_details(
    *,
    log_dir: str,
    args,
    train_dataset: ScenarioConditionalDataset,
    val_dataset: ScenarioConditionalDataset,
) -> None:
    """Write training metadata to `train_details.json` and save condition tensors.

    The JSON includes all hyperparameters plus scenario-specific fields:
    - `scenario_source.train_scenario_ids`
    - `scenario_source.source`
    - `scenario_source.params_method`
    - `scenario_source.samples_per_scenario`
    These fields are consumed by the Gradio app to display the training report card.
    """
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_conds_path = out_dir / "train_conds_raw.pt"
    val_conds_path = out_dir / "val_conds_raw.pt"
    torch.save(train_dataset.conds_raw.detach().cpu(), train_conds_path)
    torch.save(val_dataset.conds_raw.detach().cpu(), val_conds_path)

    details = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "train_cgan_scenarios",
        "seed": int(args.seed),
        "platform": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "lightning_version": L.__version__,
            "numpy_version": np.__version__,
        },
        "training_hyperparameters": vars(args),
        "scenario_source": {
            "train_scenario_ids": args.train_scenarios,
            "source": args.source,
            "params_method": args.params_method,
            "samples_per_scenario": int(args.samples_per_scenario),
            "scenarios_root": str(Path(args.scenarios_root).resolve()),
        },
        "datasets": {
            "train": {
                "num_scenarios": len(train_dataset.scenario_ids),
                "samples_per_scenario": train_dataset.samples_per_scenario,
                "num_samples_total": len(train_dataset),
                "normalize_conds": bool(args.normalize_conds),
            },
            "val": {
                "num_scenarios": len(val_dataset.scenario_ids),
                "samples_per_scenario": val_dataset.samples_per_scenario,
                "num_samples_total": len(val_dataset),
                "normalize_conds": bool(args.normalize_conds),
            },
        },
        "dataloaders": {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "train_shuffle": True,
            "val_shuffle": False,
        },
        "saved_artifacts": {
            "train_conds_raw": str(train_conds_path),
            "val_conds_raw": str(val_conds_path),
            "details_json": str(out_dir / "train_details.json"),
        },
    }

    details_path = out_dir / "train_details.json"
    details_path.write_text(
        json.dumps(_to_jsonable(details), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _is_global_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _configure_runtime_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=(
            r"functools\.partial will be a method descriptor in future Python "
            r"versions; wrap it in enum\.member\(\) if you want to preserve "
            r"the old behavior"
        ),
        module=r"torch\.distributed\.algorithms\.ddp_comm_hooks(\..*)?",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _configure_runtime_warning_filters()
    args = get_scenario_cgan_args()

    scenarios_root = Path(args.scenarios_root)

    L.seed_everything(seed=args.seed, workers=True)

    model = CGAN(
        latent_dim=args.latent_dim,
        cond_dim=args.cond_dim,
        cond_emb_dim=args.cond_emb_dim,
        g_every_k_steps=args.g_every_k_steps,
        distribution_name="mftr",
        val_metric_max_samples=int(args.val_metric_max_samples),
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        betas_g=tuple(args.betas_g),
        betas_d=tuple(args.betas_d),
    )

    # Training dataset: full sample count for selected scenarios.
    train_dataset = ScenarioConditionalDataset(
        scenario_ids=args.train_scenarios,
        scenarios_root=scenarios_root,
        samples_per_scenario=args.samples_per_scenario,
        source=args.source,
        method=args.params_method,
        dist_type=args.dist_type,
        seed=args.seed,
    )

    # Validation dataset: same scenarios, reduced sample count for speed.
    # 1000 samples per scenario is enough to compute val_wasserstein reliably.
    val_samples = max(1000, args.samples_per_scenario // 10)
    val_dataset = ScenarioConditionalDataset(
        scenario_ids=args.train_scenarios,
        scenarios_root=scenarios_root,
        samples_per_scenario=val_samples,
        source=args.source,
        method=args.params_method,
        dist_type=args.dist_type,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset),
        num_workers=args.num_workers,
        shuffle=False,
    )

    is_rank_zero = _is_global_rank_zero()
    logger = (
        TensorBoardLogger(save_dir=args.logs_dir, name="mftr/cgan_scenarios")
        if is_rank_zero
        else False
    )

    devices = _parse_devices(args.devices)
    # Use DDP only when more than one device is requested.
    strategy = "ddp_find_unused_parameters_true" if devices != 1 else "auto"

    trainer = L.Trainer(
        max_epochs=args.epochs,
        strategy=strategy,
        accelerator="auto",
        devices=devices,
        logger=logger,
        deterministic=True,
        benchmark=False,
        log_every_n_steps=50,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero and trainer.logger and trainer.logger.log_dir:
        _save_training_details(
            log_dir=trainer.logger.log_dir,
            args=args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )


if __name__ == "__main__":
    main()
