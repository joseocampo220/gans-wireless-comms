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
from torch.utils.data import ConcatDataset, DataLoader

from cgans import CGAN
from config import get_scenario_cgan_args, _parse_devices, _parse_range
from local_datasets import MftrUniformConditionalDataset, ScenarioConditionalDataset

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
    train_conds_raw: torch.Tensor | None = None,
    val_conds_raw: torch.Tensor | None = None,
    mix_mftr_info: dict | None = None,
) -> None:
    """Write training metadata to `train_details.json` and save condition tensors.

    The JSON includes all hyperparameters plus scenario-specific fields:
    - `scenario_source`: scenario IDs, source, params method, samples per scenario
    - `mix_mftr`: (optional) MFTR uniform mixing configuration

    When `mix_mftr_info` is provided, ``train_dataset`` and ``val_dataset`` may
    be wrapped in a ``ConcatDataset`` and their ``conds_raw`` / scenario-specific
    attributes extracted from the passed keyword arguments instead.
    """
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve conds_raw: explicit tensors take precedence (needed for
    # ConcatDataset wraps), fall back to dataset attributes.
    tcr = train_conds_raw if train_conds_raw is not None else train_dataset.conds_raw
    vcr = val_conds_raw if val_conds_raw is not None else val_dataset.conds_raw
    train_conds_path = out_dir / "train_conds_raw.pt"
    val_conds_path = out_dir / "val_conds_raw.pt"
    torch.save(tcr.detach().cpu(), train_conds_path)
    torch.save(vcr.detach().cpu(), val_conds_path)

    # Extract scenario-specific metadata, which may live on the wrapped dataset
    # when mixing or directly on the dataset otherwise.
    if mix_mftr_info is not None:
        scenario_source = mix_mftr_info.get("scenario_source", {})
        train_spec = mix_mftr_info.get("train_dataset", {})
        val_spec = mix_mftr_info.get("val_dataset", {})
    else:
        scenario_source = {
            "train_scenario_ids": args.train_scenarios,
            "source": args.source,
            "params_method": args.params_method,
            "samples_per_scenario": int(args.samples_per_scenario),
            "scenarios_root": str(Path(args.scenarios_root).resolve()),
            "cond_ranges": {
                name: list(train_dataset.cond_ranges[i])
                for i, name in enumerate(["m", "mu", "K", "delta", "omega"])
            },
        }
        train_spec = {
            "num_scenarios": len(train_dataset.scenario_ids),
            "samples_per_scenario": train_dataset.samples_per_scenario,
            "num_samples_total": len(train_dataset),
        }
        val_spec = {
            "num_scenarios": len(val_dataset.scenario_ids),
            "samples_per_scenario": val_dataset.samples_per_scenario,
            "num_samples_total": len(val_dataset),
        }

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
        "scenario_source": scenario_source,
        "datasets": {
            "train": {
                **train_spec,
                "normalize_conds": bool(args.normalize_conds),
            },
            "val": {
                **val_spec,
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

    # Add MFTR uniform mix block if mixing was enabled.
    if mix_mftr_info is not None:
        details["mix_mftr"] = {
            k: v
            for k, v in mix_mftr_info.items()
            if k not in ("scenario_source", "train_dataset", "val_dataset")
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


def _build_mixed_datasets(args, scenarios_root: Path) -> tuple[
    ScenarioConditionalDataset | ConcatDataset,
    ScenarioConditionalDataset | ConcatDataset,
    dict | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Build train/val datasets, optionally mixing in MFTR uniform samples.

    Returns
    -------
    train_dataset, val_dataset, mix_mftr_info, train_conds_raw, val_conds_raw
    When mixing is disabled (``not args.mix_mftr``), the last three
    values are ``None``.
    """
    # --- Always build scenario datasets first ---
    train_scenario = ScenarioConditionalDataset(
        scenario_ids=args.train_scenarios,
        scenarios_root=scenarios_root,
        samples_per_scenario=args.samples_per_scenario,
        source=args.source,
        method=args.params_method,
        dist_type=args.dist_type,
        normalize_conds=False,  # applied later if mixing
        seed=args.seed,
    )

    val_samples = max(1000, args.samples_per_scenario // 10)
    val_scenario = ScenarioConditionalDataset(
        scenario_ids=args.train_scenarios,
        scenarios_root=scenarios_root,
        samples_per_scenario=val_samples,
        source=args.source,
        method=args.params_method,
        dist_type=args.dist_type,
        normalize_conds=False,
        seed=args.seed + 1,
    )

    # No mixing — normalise scenarios directly and return.
    if not args.mix_mftr:
        if args.normalize_conds:
            # Rebuild with normalisation now that cond_ranges are known.
            train_scenario = ScenarioConditionalDataset(
                scenario_ids=args.train_scenarios,
                scenarios_root=scenarios_root,
                samples_per_scenario=args.samples_per_scenario,
                source=args.source,
                method=args.params_method,
                dist_type=args.dist_type,
                normalize_conds=True,
                seed=args.seed,
            )
            val_scenario = ScenarioConditionalDataset(
                scenario_ids=args.train_scenarios,
                scenarios_root=scenarios_root,
                samples_per_scenario=val_samples,
                source=args.source,
                method=args.params_method,
                dist_type=args.dist_type,
                normalize_conds=True,
                seed=args.seed + 1,
            )
        return train_scenario, val_scenario, None, None, None

    # --- Mixing enabled: build MFTR uniform dataset ---
    m_range = _parse_range(args.m_range, "--m_range")
    mu_range = _parse_range(args.mu_range, "--mu_range")
    K_range = _parse_range(args.K_range, "--K_range")
    delta_range = _parse_range(args.delta_range, "--delta_range")
    omega_range = _parse_range(args.omega_range, "--omega_range")
    mix_samples_per_combo = (
        args.mix_mftr_samples_per_combo
        if args.mix_mftr_samples_per_combo is not None
        else args.samples_per_scenario
    )

    mftr_ranges = {
        "m": list(m_range),
        "mu": list(mu_range),
        "K": list(K_range),
        "delta": list(delta_range),
        "omega": list(omega_range),
    }

    train_mftr = MftrUniformConditionalDataset(
        samples_per_combo=mix_samples_per_combo,
        combos=args.mix_mftr_combos,
        m=m_range,
        mu=mu_range,
        K=K_range,
        delta=delta_range,
        omega=omega_range,
        normalize_conds=False,
        dist_type=args.dist_type,
        seed=args.seed + 1000,
    )

    # Compute unified cond_ranges: union of scenario param bounds and MFTR
    # uniform ranges. These are used by both datasets at training *and*
    # persisted for inference.
    unified_cond_ranges: tuple[tuple[float, float], ...] = tuple(
        (
            min(
                float(train_scenario.conds_raw[:, i].min()),
                float(train_mftr.conds_raw[:, i].min()),
            ),
            max(
                float(train_scenario.conds_raw[:, i].max()),
                float(train_mftr.conds_raw[:, i].max()),
            ),
        )
        for i in range(5)
    )

    if args.normalize_conds:
        # Rebuild both scenario datasets with unified ranges.
        train_scenario = ScenarioConditionalDataset(
            scenario_ids=args.train_scenarios,
            scenarios_root=scenarios_root,
            samples_per_scenario=args.samples_per_scenario,
            source=args.source,
            method=args.params_method,
            dist_type=args.dist_type,
            normalize_conds=True,
            cond_ranges=unified_cond_ranges,
            seed=args.seed,
        )
        val_scenario = ScenarioConditionalDataset(
            scenario_ids=args.train_scenarios,
            scenarios_root=scenarios_root,
            samples_per_scenario=val_samples,
            source=args.source,
            method=args.params_method,
            dist_type=args.dist_type,
            normalize_conds=True,
            cond_ranges=unified_cond_ranges,
            seed=args.seed + 1,
        )
        train_mftr = MftrUniformConditionalDataset(
            samples_per_combo=mix_samples_per_combo,
            combos=args.mix_mftr_combos,
            m=m_range,
            mu=mu_range,
            K=K_range,
            delta=delta_range,
            omega=omega_range,
            normalize_conds=True,
            dist_type=args.dist_type,
            seed=args.seed + 1000,
        )
        # MFTR uniform dataset uses its own `self.ranges` for normalisation.
        # Override its conds to use the unified ranges instead.
        from local_datasets.mftr_uniform_conditional_dataset import (
            normalize_conds_minus1_1,
        )

        train_mftr.conds = normalize_conds_minus1_1(
            train_mftr.conds_raw, unified_cond_ranges
        )

    # Concat datasets.
    train_dataset: ConcatDataset = ConcatDataset([train_scenario, train_mftr])

    # Val mixing: smaller MFTR dataset.
    val_mftr = MftrUniformConditionalDataset(
        samples_per_combo=max(1000, mix_samples_per_combo // 10),
        combos=max(4, int(args.mix_mftr_combos) // 4),
        m=m_range,
        mu=mu_range,
        K=K_range,
        delta=delta_range,
        omega=omega_range,
        normalize_conds=False,
        dist_type=args.dist_type,
        seed=args.seed + 1001,
    )
    if args.normalize_conds:
        val_mftr.conds = normalize_conds_minus1_1(
            val_mftr.conds_raw, unified_cond_ranges
        )

    val_dataset: ConcatDataset = ConcatDataset([val_scenario, val_mftr])

    # Build metadata for _save_training_details.
    mix_mftr_info = {
        "mftr_combos": int(args.mix_mftr_combos),
        "mftr_samples_per_combo": mix_samples_per_combo,
        "mftr_ranges": mftr_ranges,
        "cond_ranges": {
            name: list(unified_cond_ranges[i])
            for i, name in enumerate(["m", "mu", "K", "delta", "omega"])
        },
        "scenario_source": {
            "train_scenario_ids": args.train_scenarios,
            "source": args.source,
            "params_method": args.params_method,
            "samples_per_scenario": int(args.samples_per_scenario),
            "scenarios_root": str(scenarios_root.resolve()),
            "cond_ranges": {
                name: list(unified_cond_ranges[i])
                for i, name in enumerate(["m", "mu", "K", "delta", "omega"])
            },
        },
        "train_dataset": {
            "num_scenarios": len(train_scenario.scenario_ids),
            "samples_per_scenario": train_scenario.samples_per_scenario,
            "num_samples_total": len(train_dataset),
        },
        "val_dataset": {
            "num_scenarios": len(val_scenario.scenario_ids),
            "samples_per_scenario": val_scenario.samples_per_scenario,
            "num_samples_total": len(val_dataset),
        },
    }

    train_conds_raw = torch.cat([train_scenario.conds_raw, train_mftr.conds_raw], dim=0)
    val_conds_raw = torch.cat([val_scenario.conds_raw, val_mftr.conds_raw], dim=0)

    return train_dataset, val_dataset, mix_mftr_info, train_conds_raw, val_conds_raw


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

    # Build datasets (scenario-only or scenario + MFTR uniform mix).
    (
        train_dataset,
        val_dataset,
        mix_mftr_info,
        train_conds_raw,
        val_conds_raw,
    ) = _build_mixed_datasets(args, scenarios_root)

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
            train_dataset=train_dataset,  # type: ignore[arg-type]
            val_dataset=val_dataset,  # type: ignore[arg-type]
            train_conds_raw=train_conds_raw,
            val_conds_raw=val_conds_raw,
            mix_mftr_info=mix_mftr_info,
        )


if __name__ == "__main__":
    main()
