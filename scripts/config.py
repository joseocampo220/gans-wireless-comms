import argparse
import json


def _parse_range(value: str, name: str) -> tuple[float, float]:
    """Parse a JSON array string like ``[low, high]`` into a float pair."""
    parsed = json.loads(value)
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or not all(isinstance(x, (int, float)) for x in parsed)
    ):
        raise ValueError(f"{name} must be a JSON array like [low, high]. Got: {value}")
    return float(parsed[0]), float(parsed[1])


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a GAN to approximate a Rayleigh distribution"
    )

    # ------------------- General  parameters -------------------
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="../logs",
        help="Directory to save logs and model checkpoints",
    )

    # ------------------- Training parameters -------------------
    parser.add_argument(
        "--gan_type",
        type=str,
        default="gan",
        help="Type of GAN to use",
        choices=["gan", "wgan_gp"],
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default="rayleigh",
        help="Name of the target distribution",
        choices=["rayleigh", "nakagami", "mftr"],
    )

    parser.add_argument(
        "--distribution_params",
        type=str,
        default="{'scale': 1.0}",
        help="Parameters of the target distribution in dictionary format",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate for the training dataset",
    )

    parser.add_argument(
        "--g_every_k_steps",
        "-k",
        type=int,
        default=3,
        help="Number of steps to wait for updating the generator",
    )

    parser.add_argument(
        "--latent_dim",
        "-d",
        type=int,
        default=1000,
        help="Latent dimension for the generator",
    )

    parser.add_argument(
        "--lambda_gp",
        type=float,
        default=10.0,
        help="Gradient penalty coefficient for WGAN-GP",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of max epochs",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use",
    )

    parser.add_argument(
        "--lr_g",
        type=float,
        default=0.0002,
        help="Learning rate for the generator optimizer",
    )

    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.0002,
        help="Learning rate for the discriminator optimizer",
    )

    parser.add_argument(
        "--betas_g",
        type=float,
        nargs=2,
        default=(0.5, 0.999),
        help="Betas for the generator optimizer",
    )

    parser.add_argument(
        "--betas_d",
        type=float,
        nargs=2,
        default=(0.5, 0.999),
        help="Betas for the discriminator optimizer",
    )

    # ------------------- Testing parameters -------------------
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=10000,
        help="Number of samples to generate for the testing dataset",
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        help="Checkpoint version for testing",
    )

    parser.add_argument(
        "--tests_save_path",
        type=str,
        default="../results/tests",
        help="Path to save testing results",
    )

    return parser.parse_args()


def get_cgan_args():
    parser = argparse.ArgumentParser(
        description="Train a CGAN on MFTR (conditional) samples"
    )

    parser.add_argument(
        "--logs_dir",
        type=str,
        default="../logs",
        help="Directory to save logs and model checkpoints",
    )

    parser.add_argument("--latent_dim", "-d", type=int, default=1000)
    parser.add_argument("--cond_dim", type=int, default=5)
    parser.add_argument("--cond_emb_dim", type=int, default=16)
    parser.add_argument(
        "--g_every_k_steps",
        "-k",
        type=int,
        default=3,
        help="Number of steps to wait for updating the generator",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr_g", type=float, default=0.0001)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--betas_g", type=float, nargs=2, default=(0.5, 0.999))
    parser.add_argument("--betas_d", type=float, nargs=2, default=(0.5, 0.999))

    parser.add_argument(
        "--val_metric_max_samples",
        type=int,
        default=20000,
        help="Max validation samples (pooled across ranks) used to compute val_wasserstein.",
    )

    parser.add_argument("--samples_per_combo", type=int, default=20000)
    parser.add_argument("--val_samples_per_combo", type=int, default=1000)

    # MFTR parameter ranges (JSON arrays like "[2.0, 10.0]")
    parser.add_argument("--combos", type=int, default=64)
    parser.add_argument("--val_combos", type=int, default=None)

    parser.add_argument("--m_range", type=str, default="[8.0, 8.0]")
    parser.add_argument("--mu_range", type=str, default="[7.0, 7.0]")
    parser.add_argument("--K_range", type=str, default="[8.0, 8.0]")
    parser.add_argument("--delta_range", type=str, default="[0.9, 0.9]")
    parser.add_argument("--omega_range", type=str, default="[2.0, 2.0]")

    parser.add_argument(
        "--normalize_conds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, normalize MFTR condition parameters to [-1, 1] using the training ranges.",
    )

    parser.add_argument(
        "--dist_type",
        type=str,
        default="amplitude",
        choices=["amplitude", "power"],
        help="Whether MFTR samples are amplitudes or powers",
    )

    # Evaluation (end-of-training)
    parser.add_argument(
        "--eval_max_mixture_samples",
        type=int,
        default=50000,
        help="Max number of mixture samples to use for end-of-training mixture evaluation",
    )
    parser.add_argument(
        "--eval_num_params_in",
        type=int,
        default=5,
        help="Number of in-range parameter sets to visualize/evaluate",
    )
    parser.add_argument(
        "--eval_num_params_out",
        type=int,
        default=5,
        help="Number of out-of-range parameter sets to visualize/evaluate",
    )
    parser.add_argument(
        "--eval_num_samples_per_param",
        type=int,
        default=5000,
        help="Number of samples to draw for each per-parameter evaluation",
    )

    parser.add_argument(
        "--eval_save_per_experiment_images",
        action="store_true",
        help=(
            "If set, also saves one image per conditional experiment under "
            "test_results/conditional_interpolation/ and test_results/conditional_extrapolation/"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _parse_devices(value: str):
    """Parse --devices as an int, comma-separated list of ints, or 'auto'."""
    if value == "auto":
        return "auto"
    if "," in value:
        return [int(x) for x in value.split(",")]
    return int(value)


def get_scenario_cgan_args():
    """CLI arguments for scenario-conditioned CGAN training.

    Scenario training uses empirical scenario data instead of a wide synthetic
    parameter grid. It shares all model/trainer hyperparameters with
    `get_cgan_args`, but replaces dataset construction arguments with:

    - `--scenarios_root`: path to the directory containing `scenario_<n>/` folders.
    - `--train_scenarios`: space-separated list of scenario ids to train on.
    - `--samples_per_scenario`: samples per scenario (default: 10000).
    - `--source`: `empirical_pdf` or `mftr_generated`.
    - `--params_method`: which estimation method to read params from.
    """
    parser = argparse.ArgumentParser(
        description="Train a CGAN on scenario-conditioned data"
    )

    # ---- Paths ----
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="../logs",
        help="Directory to save logs and model checkpoints",
    )
    parser.add_argument(
        "--scenarios_root",
        type=str,
        default="../data/scenarios",
        help="Root directory containing scenario_<n> folders",
    )

    # ---- Scenario selection ----
    parser.add_argument(
        "--train_scenarios",
        type=int,
        nargs="+",
        required=True,
        help="Space-separated list of scenario ids to use for training, e.g. --train_scenarios 1 2 3 4 5",
    )

    # ---- Dataset ----
    parser.add_argument(
        "--samples_per_scenario",
        type=int,
        default=10000,
        help="Number of samples to draw per scenario",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="empirical_pdf",
        choices=["empirical_pdf", "mftr_generated"],
        help="Sample source: empirical_pdf (default) or mftr_generated",
    )
    parser.add_argument(
        "--params_method",
        type=str,
        default="fminsearch",
        choices=["fminsearch", "ga"],
        help="Estimated-parameter method to use from estimated_params.csv",
    )
    parser.add_argument(
        "--dist_type",
        type=str,
        default="amplitude",
        choices=["amplitude", "power"],
        help="Whether MFTR samples are amplitudes or powers (used for mftr_generated source)",
    )

    # ---- Mixed training (scenario + MFTR uniform) ----
    parser.add_argument(
        "--mix_mftr",
        action="store_true",
        default=False,
        help=(
            "If set, augment the scenario dataset with MFTR samples drawn from "
            "the uniform parameter ranges below. Default: False."
        ),
    )
    parser.add_argument(
        "--mix_mftr_combos",
        type=int,
        default=64,
        help=(
            "Number of MFTR parameter combos when mixing. "
            "Only relevant when --mix_mftr is set. Default: 64."
        ),
    )
    parser.add_argument(
        "--mix_mftr_samples_per_combo",
        type=int,
        default=None,
        help=(
            "Samples per MFTR uniform combo when mixing. "
            "Defaults to the value of --samples_per_scenario."
        ),
    )
    parser.add_argument(
        "--m_range",
        type=str,
        default="[1.0, 8.0]",
        help="JSON array [low, high] for 'm'",
    )
    parser.add_argument(
        "--mu_range",
        type=str,
        default="[1.0, 9.0]",
        help="JSON array [low, high] for 'mu'",
    )
    parser.add_argument(
        "--K_range",
        type=str,
        default="[0.0, 20.0]",
        help="JSON array [low, high] for 'K'",
    )
    parser.add_argument(
        "--delta_range",
        type=str,
        default="[0.0, 0.9]",
        help="JSON array [low, high] for 'delta'",
    )
    parser.add_argument(
        "--omega_range",
        type=str,
        default="[1.0, 2.0]",
        help="JSON array [low, high] for 'omega'",
    )

    # ---- Model ----
    parser.add_argument("--latent_dim", "-d", type=int, default=1000)
    parser.add_argument("--cond_dim", type=int, default=5)
    parser.add_argument("--cond_emb_dim", type=int, default=16)
    parser.add_argument(
        "--g_every_k_steps",
        "-k",
        type=int,
        default=3,
        help="Number of discriminator steps per generator step",
    )

    # ---- Training ----
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_g", type=float, default=0.0001)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--betas_g", type=float, nargs=2, default=(0.5, 0.999))
    parser.add_argument("--betas_d", type=float, nargs=2, default=(0.5, 0.999))
    parser.add_argument(
        "--normalize_conds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, normalize MFTR condition parameters to [-1, 1]. "
            "Scenario params can span very wide ranges, so this defaults to False."
        ),
    )
    parser.add_argument(
        "--val_metric_max_samples",
        type=int,
        default=20000,
        help="Max samples used to compute val_wasserstein",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help=(
            "Devices to train on. Accepts an int (e.g. '1'), a comma-separated "
            "list of GPU ids (e.g. '0,1'), or 'auto'. Default: '1'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()
