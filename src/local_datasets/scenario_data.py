"""Utilities for loading standardized scenario data used by scenario-conditioned training.

This module centralizes file parsing for scenario folders under `data/scenarios`.
It is intentionally simple and explicit so training scripts and app callbacks can
rely on one well-documented data contract.

Organization
1. Constants that define expected file names and condition ordering.
2. Dataclasses that represent parsed scenario content.
3. Small parsing helpers for each scenario artifact file.
4. Public loader functions to discover and load scenarios.

Data contract
- `estimated_params.csv` must use rows: `K,delta,mu,m,omega` and include
  method columns (for example `fminsearch` and `ga`).
- `pdf_data.csv` must include columns: `xi,pdf_empirical,pdf_fminsearch,pdf_ga`.
- `samples.csv` is a headerless file with one numeric sample per row.

By default, samples are not loaded into memory. `ScenarioData` keeps a reference
to `samples.csv` and can load values on demand.

The selected condition vector always follows model order:
`[m, mu, K, delta, omega]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import warnings

import numpy as np

SCENARIO_DIR_PREFIX = "scenario_"
PARAM_FILE_NAME = "estimated_params.csv"
PDF_FILE_NAME = "pdf_data.csv"
SAMPLES_FILE_NAME = "samples.csv"

PARAM_ORDER = ("m", "mu", "K", "delta", "omega")
_REQUIRED_PARAM_ROWS = ("K", "delta", "mu", "m", "omega")
_REQUIRED_PDF_COLUMNS = ("xi", "pdf_empirical", "pdf_fminsearch", "pdf_ga")


@dataclass(frozen=True)
class ScenarioData:
    """Container for one parsed scenario.

    Attributes
    - `scenario_id`: Numeric scenario identifier.
    - `method_names`: Methods available in `estimated_params.csv` columns.
    - `params_by_method`: Mapping `method -> {param_name: value}`.
    - `selected_method`: Method requested when loading this scenario.
    - `selected_condition`: Condition vector in order `[m, mu, K, delta, omega]`.
    - `xi`: X-axis values from empirical PDF file.
    - `pdf_empirical`: Empirical PDF values.
    - `pdf_fminsearch`: MFTR fit PDF values for fminsearch.
    - `pdf_ga`: MFTR fit PDF values for GA.
    - `samples_path`: Reference path to `samples.csv` for lazy loading.
    - `samples`: Empirical samples as a 1D array when eager loading is requested.
    """

    scenario_id: int
    method_names: tuple[str, ...]
    params_by_method: dict[str, dict[str, float]]
    selected_method: str
    selected_condition: np.ndarray
    xi: np.ndarray
    pdf_empirical: np.ndarray
    pdf_fminsearch: np.ndarray
    pdf_ga: np.ndarray
    samples_path: Path
    samples: np.ndarray | None = None

    def load_samples(self) -> np.ndarray:
        """Load empirical samples from `samples_path` on demand."""
        return _read_samples_csv(self.samples_path)


def list_available_scenarios(scenarios_root: Path | str) -> list[int]:
    """Return sorted scenario ids found under the provided root directory."""
    root = Path(scenarios_root)
    if not root.exists():
        return []

    scenario_ids: list[int] = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith(SCENARIO_DIR_PREFIX):
            continue
        suffix = path.name[len(SCENARIO_DIR_PREFIX) :]
        if suffix.isdigit():
            scenario_ids.append(int(suffix))
    return sorted(scenario_ids)


def load_scenario_data(
    scenario_id: int,
    scenarios_root: Path | str,
    *,
    method: str = "fminsearch",
    load_samples: bool = False,
) -> ScenarioData:
    """Load one scenario and return structured content.

    Raises
    - `FileNotFoundError` if a required file is missing.
    - `ValueError` if file schemas are invalid or if the selected method is absent.
    """
    scenario_dir = Path(scenarios_root) / f"{SCENARIO_DIR_PREFIX}{int(scenario_id)}"
    _ensure_required_files_exist(scenario_dir)

    method_names, params_by_method = _read_estimated_params_csv(
        scenario_dir / PARAM_FILE_NAME
    )
    if method not in params_by_method:
        raise ValueError(
            f"Method '{method}' not found in {scenario_dir / PARAM_FILE_NAME}; "
            f"available methods: {sorted(params_by_method)}"
        )

    xi, pdf_empirical, pdf_fminsearch, pdf_ga = _read_pdf_data_csv(
        scenario_dir / PDF_FILE_NAME
    )
    samples_path = scenario_dir / SAMPLES_FILE_NAME
    samples = _read_samples_csv(samples_path) if load_samples else None

    selected = params_by_method[method]
    selected_condition = np.array(
        [selected[p] for p in PARAM_ORDER],
        dtype=np.float32,
    )

    return ScenarioData(
        scenario_id=int(scenario_id),
        method_names=method_names,
        params_by_method=params_by_method,
        selected_method=method,
        selected_condition=selected_condition,
        xi=xi,
        pdf_empirical=pdf_empirical,
        pdf_fminsearch=pdf_fminsearch,
        pdf_ga=pdf_ga,
        samples_path=samples_path,
        samples=samples,
    )


def _ensure_required_files_exist(scenario_dir: Path) -> None:
    required = (PARAM_FILE_NAME, PDF_FILE_NAME, SAMPLES_FILE_NAME)
    missing = [name for name in required if not (scenario_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required scenario files in {scenario_dir}: {missing}"
        )


def _read_estimated_params_csv(
    csv_path: Path,
) -> tuple[tuple[str, ...], dict[str, dict[str, float]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {csv_path}")

        fieldnames = [name.strip() for name in reader.fieldnames]
        if "param" not in fieldnames:
            raise ValueError(f"Missing 'param' column in {csv_path}")

        methods = tuple(name for name in fieldnames if name != "param")
        if not methods:
            raise ValueError(f"No method columns found in {csv_path}")

        rows_by_param: dict[str, dict[str, float]] = {}
        for row in reader:
            param_name = str(row.get("param", "")).strip()
            if not param_name:
                continue

            parsed: dict[str, float] = {}
            for method in methods:
                raw = row.get(method)
                if raw is None or str(raw).strip() == "":
                    raise ValueError(
                        f"Missing value for param '{param_name}' method '{method}' "
                        f"in {csv_path}"
                    )
                parsed[method] = float(raw)
            rows_by_param[param_name] = parsed

    missing_rows = [name for name in _REQUIRED_PARAM_ROWS if name not in rows_by_param]
    if missing_rows:
        raise ValueError(
            f"Missing required parameter rows {missing_rows} in {csv_path}. "
            "Expected canonical rows: K, delta, mu, m, omega."
        )

    params_by_method: dict[str, dict[str, float]] = {}
    for method in methods:
        params_by_method[method] = {
            param_name: rows_by_param[param_name][method]
            for param_name in _REQUIRED_PARAM_ROWS
        }

    return methods, params_by_method


def _read_pdf_data_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi_values: list[float] = []
    empirical_values: list[float] = []
    fmin_values: list[float] = []
    ga_values: list[float] = []

    seen_rows: set[tuple[float, float, float, float]] = set()
    duplicates = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {csv_path}")

        missing_cols = [c for c in _REQUIRED_PDF_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} in {csv_path}")

        for row in reader:
            values = (
                float(row["xi"]),
                float(row["pdf_empirical"]),
                float(row["pdf_fminsearch"]),
                float(row["pdf_ga"]),
            )
            if values in seen_rows:
                duplicates += 1
                continue
            seen_rows.add(values)

            xi_values.append(values[0])
            empirical_values.append(values[1])
            fmin_values.append(values[2])
            ga_values.append(values[3])

    if duplicates:
        warnings.warn(
            f"Ignored {duplicates} duplicate PDF rows in {csv_path}",
            RuntimeWarning,
            stacklevel=2,
        )

    return (
        np.asarray(xi_values, dtype=np.float32),
        np.asarray(empirical_values, dtype=np.float32),
        np.asarray(fmin_values, dtype=np.float32),
        np.asarray(ga_values, dtype=np.float32),
    )


def _read_samples_csv(csv_path: Path) -> np.ndarray:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            values.append(float(row[0]))

    if not values:
        raise ValueError(f"No numeric sample values found in {csv_path}")

    return np.asarray(values, dtype=np.float32)
