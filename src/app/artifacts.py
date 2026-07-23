from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .config import CURATED_VERSIONS, LOGS_ROOT, _version_sort_key


@dataclass(frozen=True)
class VersionArtifacts:
    version: str
    version_dir: Path
    checkpoint: Path | None
    hparams: dict[str, Any]
    train_details: dict[str, Any]
    eval_metadata: dict[str, Any]
    conditional_metrics: dict[str, Any]
    status: str

    @property
    def ranges(self) -> dict[str, tuple[float, float]]:
        r = self.eval_metadata.get("ranges", {})
        if not r:
            r = self.train_details.get("distribution", {}).get("parameter_ranges", {})
        out: dict[str, tuple[float, float]] = {}
        for k in ["m", "mu", "K", "delta", "omega"]:
            v = r.get(k, [0.0, 0.0])
            out[k] = (float(v[0]), float(v[1]))
        return out

    @property
    def is_scenario_run(self) -> bool:
        """True when this version was trained with train_cgan_scenarios."""
        return self.train_details.get("run_type") == "train_cgan_scenarios"

    @property
    def scenario_source(self) -> dict[str, Any]:
        """Scenario-specific training metadata from train_details.json."""
        return self.train_details.get("scenario_source", {})


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hparams_yaml(path: Path) -> dict[str, Any]:
    """Minimal parser for the hparams.yaml format produced by Lightning.

    Handles scalar key:value lines and !!python/tuple blocks used for betas.
    """
    if not path.exists():
        return {}

    lines = path.read_text(encoding="utf-8").splitlines()
    out: dict[str, Any] = {}
    i = 0

    def _coerce(s: str) -> Any:
        s = s.strip()
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return s

    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line.lstrip().startswith("#"):
            i += 1
            continue

        if ":" not in line:
            i += 1
            continue

        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()

        if raw == "!!python/tuple":
            vals: list[Any] = []
            i += 1
            while i < len(lines):
                item = lines[i].strip()
                if not item.startswith("-"):
                    break
                vals.append(_coerce(item[1:].strip()))
                i += 1
            out[key] = tuple(vals)
            continue

        out[key] = _coerce(raw)
        i += 1

    return out


def _list_checkpoint_names(version_dir: Path) -> list[str]:
    """Return sorted checkpoint stems available in a version directory.

    Preferred names ("best", "last") come first, then any epoch-named files.
    """
    ckpt_dir = version_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    names = sorted({p.stem for p in ckpt_dir.glob("*.ckpt")})
    preferred = ["best", "last"]
    ordered = [n for n in preferred if n in names]
    ordered += [n for n in names if n not in preferred]
    return ordered


def _pick_checkpoint(version_dir: Path) -> Path | None:
    ckpt_dir = version_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Prefer best_model.ckpt, then best.ckpt, then the last epoch file.
    for preferred in ("best_model", "best"):
        candidate = ckpt_dir / f"{preferred}.ckpt"
        if candidate.exists():
            return candidate

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    return ckpts[-1]


def discover_versions(curated: list[str] | None = None) -> list[str]:
    requested = curated if curated is not None else CURATED_VERSIONS
    out: list[str] = []
    for v in requested:
        if (LOGS_ROOT / v).exists():
            out.append(v)
    return out


def _fresh_scan_versions(logs_root: Path) -> list[str]:
    """Live directory scan — does not rely on module-level caches."""
    if not logs_root.exists():
        return []
    return sorted(
        (
            p.name
            for p in logs_root.iterdir()
            if p.is_dir() and p.name.startswith("version_")
        ),
        key=_version_sort_key,
    )


def _load_single_artifacts(
    version_dir: Path,
    version_key: str,
    checkpoint_name: str | None = None,
) -> VersionArtifacts:
    """Build a VersionArtifacts for a single version directory and checkpoint.

    Parameters
    ----------
    version_dir
        Absolute path to the version directory.
    version_key
        Display key in the registry (e.g. ``\"version_0\"`` or ``\"scenarios/version_0\"``).
    checkpoint_name
        Checkpoint stem (``\"best\"``, ``\"last\"``, etc.). When ``None``, picks the
        best or last checkpoint automatically (backward compat).
    """
    test_results = version_dir / "test_results"

    hparams = _load_hparams_yaml(version_dir / "hparams.yaml")
    train_details = _load_json(version_dir / "train_details.json")
    eval_metadata = _load_json(test_results / "eval_metadata.json")
    conditional_metrics = _load_json(test_results / "conditional_metrics.json")

    if checkpoint_name is not None:
        ckpt = version_dir / "checkpoints" / f"{checkpoint_name}.ckpt"
        if not ckpt.exists():
            ckpt = None
    else:
        ckpt = _pick_checkpoint(version_dir)

    has_eval = bool(eval_metadata) and bool(conditional_metrics)
    status = "complete" if has_eval and ckpt else "partial"

    return VersionArtifacts(
        version=version_key,
        version_dir=version_dir,
        checkpoint=ckpt,
        hparams=hparams,
        train_details=train_details,
        eval_metadata=eval_metadata,
        conditional_metrics=conditional_metrics,
        status=status,
    )


def _expand_checkpoints(
    version_dir: Path,
    base_key: str,
) -> dict[str, VersionArtifacts]:
    """Create one registry entry per available checkpoint in a version directory.

    Returns a dict mapping display keys like ``\"version_0/best\"`` to their
    ``VersionArtifacts``.
    """
    names = _list_checkpoint_names(version_dir)
    if not names:
        return {}
    result: dict[str, VersionArtifacts] = {}
    for name in names:
        key = f"{base_key}/{name}"
        result[key] = _load_single_artifacts(version_dir, key, checkpoint_name=name)
    return result


def load_version_artifacts(version: str) -> VersionArtifacts:
    version_dir = LOGS_ROOT / version
    return _load_single_artifacts(version_dir, version)


def load_scenario_version_artifacts(version: str) -> VersionArtifacts:
    """Load artifacts for a scenario-trained model from the cgan_scenarios log dir."""
    from .config import SCENARIO_LOGS_ROOT
    version_dir = SCENARIO_LOGS_ROOT / version
    return _load_single_artifacts(version_dir, f"scenarios/{version}")


def load_registry(curated: list[str] | None = None) -> dict[str, VersionArtifacts]:
    versions = discover_versions(curated)
    return {v: load_version_artifacts(v) for v in versions}


def load_combined_registry() -> dict[str, VersionArtifacts]:
    """Load model versions from both cgan (MFTR) and cgan_scenarios log dirs.

    Each version is expanded into its available checkpoints (``best``, ``last``,
    etc.) so the UI lets the user pick which checkpoint to use.

    Uses fresh directory scans — does **not** rely on module-level caches, so
    calling this again picks up newly trained models without restarting the app.
    """
    from .config import SCENARIO_LOGS_ROOT

    registry: dict[str, VersionArtifacts] = {}
    for v in _fresh_scan_versions(LOGS_ROOT):
        registry.update(_expand_checkpoints(LOGS_ROOT / v, v))
    for v in _fresh_scan_versions(SCENARIO_LOGS_ROOT):
        registry.update(
            _expand_checkpoints(SCENARIO_LOGS_ROOT / v, f"scenarios/{v}")
        )
    return registry


def _fmt_range(name: str, r: tuple[float, float]) -> str:
    lo, hi = r
    if lo == hi:
        return f"- **{name}**: {lo:.3f}"
    return f"- **{name}**: [{lo:.3f}, {hi:.3f}]"


def model_card_markdown(art: VersionArtifacts) -> str:
    if art.is_scenario_run:
        return _scenario_model_card_markdown(art)

    hp = dict(art.hparams)
    td = art.train_details
    hp_td = td.get("training_hyperparameters", {})
    hp = {**hp_td, **hp} if hp_td else hp
    r = art.ranges

    ckpt = str(art.checkpoint.name) if art.checkpoint else "Not found"
    status = "Complete" if art.status == "complete" else "Partial"

    dl = td.get("dataloaders", {})
    split = td.get("split", {})
    ds_train = td.get("datasets", {}).get("train", {})
    ds_val = td.get("datasets", {}).get("val", {})
    platform = td.get("platform", {})

    rows = [
        f"### Model {art.version} ({status})",
        f"- **Checkpoint file**: {ckpt}",
        f"- **Saved at (UTC)**: {td.get('saved_at_utc', 'N/A')}",
        "#### Architecture",
        f"- **Latent vector size**: {hp.get('latent_dim', 'N/A')}",
        f"- **Condition input size**: {hp.get('cond_dim', 'N/A')}",
        f"- **Condition embedding size**: {hp.get('cond_emb_dim', 'N/A')}",
        "#### Training setup",
        f"- **Generator update frequency**: every {hp.get('g_every_k_steps', 'N/A')} discriminator steps",
        f"- **Generator learning rate**: {hp.get('lr_g', 'N/A')}",
        f"- **Discriminator learning rate**: {hp.get('lr_d', 'N/A')}",
        f"- **Generator Adam betas**: {hp.get('betas_g', 'N/A')}",
        f"- **Discriminator Adam betas**: {hp.get('betas_d', 'N/A')}",
        f"- **Condition normalization to [-1, 1]**: {ds_train.get('normalize_conds', hp.get('normalize_conds', art.eval_metadata.get('normalize_conds', 'N/A')))}",
        "#### Data and split",
        f"- **Train combos / val combos**: {split.get('train_combos', 'N/A')} / {split.get('val_combos', 'N/A')}",
        f"- **Train samples per combo**: {ds_train.get('samples_per_combo', hp.get('samples_per_combo', 'N/A'))}",
        f"- **Val samples per combo**: {ds_val.get('samples_per_combo', hp.get('val_samples_per_combo', 'N/A'))}",
        f"- **Total train samples**: {ds_train.get('num_samples_total', 'N/A')}",
        f"- **Batch size / workers**: {dl.get('batch_size', hp.get('batch_size', 'N/A'))} / {dl.get('num_workers', hp.get('num_workers', 'N/A'))}",
        f"- **Validation batch size**: {dl.get('val_batch_size', 'N/A')}",
        "#### Parameter ranges used during training",
        _fmt_range("m", r["m"]),
        _fmt_range("mu", r["mu"]),
        _fmt_range("K", r["K"]),
        _fmt_range("delta", r["delta"]),
        _fmt_range("omega", r["omega"]),
        "#### Runtime",
        f"- **Python / Torch / Lightning**: {platform.get('python_version', 'N/A')} / {platform.get('torch_version', 'N/A')} / {platform.get('lightning_version', 'N/A')}",
    ]
    return "\n".join(rows)


def _scenario_model_card_markdown(art: VersionArtifacts) -> str:
    """Model card for scenario-trained models (train_cgan_scenarios runs)."""
    td = art.train_details
    src = art.scenario_source
    hp = td.get("training_hyperparameters", {})
    ds_train = td.get("datasets", {}).get("train", {})
    dl = td.get("dataloaders", {})
    platform = td.get("platform", {})
    ckpt = str(art.checkpoint.name) if art.checkpoint else "Not found"
    status = "Complete" if art.status == "complete" else "Partial"

    train_ids = src.get("train_scenario_ids", [])
    rows = [
        f"### Model {art.version} — Scenario ({status})",
        f"- **Checkpoint file**: {ckpt}",
        f"- **Saved at (UTC)**: {td.get('saved_at_utc', 'N/A')}",
        "#### Scenario training conditions",
        f"- **Training scenarios**: {', '.join(str(s) for s in train_ids)}",
        f"- **Sample source**: {src.get('source', 'N/A')}",
        f"- **Params method**: {src.get('params_method', 'N/A')}",
        f"- **Samples per scenario**: {src.get('samples_per_scenario', 'N/A')}",
        f"- **Total training samples**: {ds_train.get('num_samples_total', 'N/A')}",
        f"- **Cond. normalization**: {ds_train.get('normalize_conds', 'N/A')}",
        "#### Architecture",
        f"- **Latent dim / Cond dim / Cond emb dim**: {hp.get('latent_dim', 'N/A')} / {hp.get('cond_dim', 'N/A')} / {hp.get('cond_emb_dim', 'N/A')}",
        "#### Training setup",
        f"- **Epochs / Batch size**: {hp.get('epochs', 'N/A')} / {dl.get('batch_size', hp.get('batch_size', 'N/A'))}",
        f"- **Generator LR / Discriminator LR**: {hp.get('lr_g', 'N/A')} / {hp.get('lr_d', 'N/A')}",
        f"- **G update every k steps**: {hp.get('g_every_k_steps', 'N/A')}",
        "#### Runtime",
        f"- **Python / Torch / Lightning**: {platform.get('python_version', 'N/A')} / {platform.get('torch_version', 'N/A')} / {platform.get('lightning_version', 'N/A')}",
    ]
    return "\n".join(rows)
