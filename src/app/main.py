from __future__ import annotations

import argparse
import numpy as np

import gradio as gr
from gradio.themes import Soft

from app.artifacts import VersionArtifacts, load_combined_registry, model_card_markdown
from app.config import (
    CURATED_VERSIONS,
    DEFAULT_TEST_PARAMS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    UI_PARAM_BOUNDS,
    SCENARIOS_DATA_ROOT,
)
from app.content import read_markdown
from app.inference import metrics_table, run_comparison, run_scenario_comparison
from app.plots import (
    make_cdf_figure,
    make_density_figure,
    make_scenario_density_figure,
    make_discriminator_architecture_diagram,
    make_generator_architecture_diagram,
    make_pipeline_diagram,
    make_qq_figure,
)


APP_CSS = """
.wrap {max-width: 1180px; margin: 0 auto;}
.smallnote {font-size: 0.92rem; color: #444;}

.callout-info {
    background-color: #f0f7ff;
    border-left: 5px solid #007bff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 4px;
    color: #333;
}
.callout-info strong {
    color: #007bff;
    display: block;
    margin-bottom: 5px;
}
"""

LATEX_DELIMITERS = [
  {"left": "$$", "right": "$$", "display": True},
  {"left": "$", "right": "$", "display": False},
  {"left": "\\(", "right": "\\)", "display": False},
  {"left": "\\[", "right": "\\]", "display": True}
]

def _slider_settings(name: str, art: VersionArtifacts) -> tuple[float, float, float, float, bool]:
    lo_ui, hi_ui = UI_PARAM_BOUNDS[name]
    value = max(lo_ui, min(hi_ui, _default_param_value(art, name)))
    step = max(1e-3, (hi_ui - lo_ui) / 500.0)
    return lo_ui, hi_ui, value, step, True


def _default_param_value(art: VersionArtifacts, name: str) -> float:
    _ = art
    return float(DEFAULT_TEST_PARAMS[name])


def _slider_update(name: str, art: VersionArtifacts):
    lo, hi, value, step, interactive = _slider_settings(name, art)
    return gr.update(
        minimum=lo,
        maximum=hi,
        value=value,
        step=step,
        interactive=interactive,
    )


def _on_model_or_extrapolation_change(
    version: str,
    registry: dict[str, VersionArtifacts],
):
    art = registry[version]
    return (
        model_card_markdown(art),
        _slider_update("m", art),
        _slider_update("mu", art),
        _slider_update("K", art),
        _slider_update("delta", art),
        _slider_update("omega", art),
    )


SCENARIO_CHOICES = ["None"] + [f"Scenario {i}" for i in range(1, 7)]


def _reload_registry(registry: dict) -> gr.update:
    """Reload both cgan and cgan_scenarios log dirs and refresh the version dropdown.

    The registry dict is mutated in-place so all existing callbacks that close
    over it automatically see the updated model list without restarting the app.
    """
    new = load_combined_registry()
    registry.clear()
    registry.update(new)
    versions = list(registry.keys())
    return gr.update(choices=versions, value=versions[0] if versions else None)


def _make_on_scenario_change(scenario_params: dict[int, dict[str, float]]):
    """Return a callback that autofills sliders from the selected scenario's params."""

    def _on_scenario_change(scenario_choice: str):
        if not scenario_choice or scenario_choice == "None":
            card = "*No scenario selected — comparing against theoretical MFTR distribution.*"
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), card

        sid = int(scenario_choice.split()[-1])
        params = scenario_params.get(sid, {})

        def _clamp(name: str, val: float) -> float:
            lo, hi = UI_PARAM_BOUNDS[name]
            return float(max(lo, min(hi, val)))

        card_lines = [
            f"#### Scenario {sid} — fminsearch parameters",
            f"- **m**: {params.get('m', '?'):.3f}",
            f"- **mu**: {params.get('mu', '?'):.4f}",
            f"- **K**: {params.get('K', '?'):.3f}",
            f"- **delta**: {params.get('delta', '?'):.4f}",
            f"- **omega**: {params.get('omega', '?'):.4f}",
        ]
        card = "\n".join(card_lines)
        return (
            gr.update(value=_clamp("m", params.get("m", 8.0))),
            gr.update(value=_clamp("mu", params.get("mu", 5.0))),
            gr.update(value=_clamp("K", params.get("K", 9.0))),
            gr.update(value=_clamp("delta", params.get("delta", 0.9))),
            gr.update(value=_clamp("omega", params.get("omega", 1.0))),
            card,
        )

    return _on_scenario_change


def _run_callback(
    version: str,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
    n_samples: int,
    seed: int,
    scenario_choice: str,
    registry: dict[str, VersionArtifacts],
):
    art = registry[version]
    scenario_id = None
    if scenario_choice and scenario_choice != "None":
        scenario_id = int(scenario_choice.split()[-1])

    if scenario_id is not None:
        real, generated, metrics, err = run_scenario_comparison(
            art=art,
            m=float(m), mu=float(mu), K=float(K),
            delta=float(delta), omega=float(omega),
            scenario_id=scenario_id,
            n_samples=int(n_samples),
            seed=int(seed),
        )
    else:
        real, generated, metrics, err = run_comparison(
            art=art,
            m=float(m), mu=float(mu), K=float(K),
            delta=float(delta), omega=float(omega),
            n_samples=int(n_samples),
            seed=int(seed),
        )

    if err is not None:
        return None, None, None, [], f"### Error\n{err}"

    plot_n = min(3500, real.shape[0], generated.shape[0])
    if plot_n < real.shape[0]:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(real.shape[0], size=plot_n, replace=False)
        real_plot = real[idx]
        gen_plot = generated[idx]
    else:
        real_plot = real
        gen_plot = generated

    qq_fig = make_qq_figure(real=real_plot, generated=gen_plot)

    if scenario_id is not None:
        density_fig = make_scenario_density_figure(
            scenario_samples=real_plot,
            generated=gen_plot,
            scenario_id=scenario_id,
        )
    else:
        density_fig = make_density_figure(
            generated=gen_plot,
            m=float(m), mu=float(mu), K=float(K),
            delta=float(delta), omega=float(omega),
        )

    cdf_fig = make_cdf_figure(real=real_plot, generated=gen_plot)
    rows = metrics_table(metrics)
    return qq_fig, density_fig, cdf_fig, rows, ""


def build_app() -> gr.Blocks:
    registry = load_combined_registry()
    if not registry:
        raise RuntimeError("No curated model versions found under logs/mftr/cgan")

    # Preload scenario params for autofill (fminsearch method, all available scenarios).
    from local_datasets.scenario_data import load_scenario_data, list_available_scenarios
    scenario_params: dict[int, dict[str, float]] = {}
    for sid in list_available_scenarios(SCENARIOS_DATA_ROOT):
        try:
            sc = load_scenario_data(sid, SCENARIOS_DATA_ROOT, method="fminsearch")
            c = sc.selected_condition  # [m, mu, K, delta, omega]
            scenario_params[sid] = {
                "m": float(c[0]), "mu": float(c[1]), "K": float(c[2]),
                "delta": float(c[3]), "omega": float(c[4]),
            }
        except Exception:
            pass

    versions = list(registry.keys())
    default_version = versions[0]
    default_art = registry[default_version]

    on_scenario_change = _make_on_scenario_change(scenario_params)

    with gr.Blocks(title="MFTR cGAN Interactive Demo") as demo:
        gr.Markdown(
            "# MFTR Conditional GAN Interactive Page\n"
            "Compact project overview + interactive model comparison against theoretical MFTR."
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                with gr.Row(elem_classes=["wrap"]):
                    with gr.Column(scale=7):
                        gr.Markdown(read_markdown("overview.md"), latex_delimiters=LATEX_DELIMITERS)
                    with gr.Column(scale=5):
                        gr.Plot(value=make_pipeline_diagram, label="Pipeline")
                        gr.HTML(value="<h3 style='color: #0f766e;'>Generator Architecture</h3>")
                        gr.HTML(value=make_generator_architecture_diagram, label="Generator (torchview)")
                        gr.HTML(value="<h3 style='color: #7c2d12;'>Discriminator Architecture</h3>")
                        gr.HTML(value=make_discriminator_architecture_diagram, label="Discriminator (torchview)")

            with gr.Tab("Model in Action"):
                with gr.Row(elem_classes=["wrap"]):
                    with gr.Column(scale=4):
                        with gr.Row():
                            version_dd = gr.Dropdown(
                                choices=versions,
                                value=default_version,
                                label="Model version",
                                scale=5,
                            )
                            reload_btn = gr.Button("↺ Reload", scale=1, min_width=80)
                        model_card = gr.Markdown(model_card_markdown(default_art))

                        scenario_dd = gr.Dropdown(
                            choices=SCENARIO_CHOICES,
                            value="None",
                            label="Scenario (optional — autofills sliders and uses empirical samples as reference)",
                        )
                        scenario_card = gr.Markdown(
                            "*No scenario selected — comparing against theoretical MFTR distribution.*"
                        )

                        m_lo, m_hi, m_val, m_step, m_inter = _slider_settings("m", default_art)
                        mu_lo, mu_hi, mu_val, mu_step, mu_inter = _slider_settings("mu", default_art)
                        k_lo, k_hi, k_val, k_step, k_inter = _slider_settings("K", default_art)
                        d_lo, d_hi, d_val, d_step, d_inter = _slider_settings("delta", default_art)
                        o_lo, o_hi, o_val, o_step, o_inter = _slider_settings("omega", default_art)

                        m_slider = gr.Slider(label="m", minimum=m_lo, maximum=m_hi, value=m_val, step=m_step, interactive=m_inter)
                        mu_slider = gr.Slider(label="mu", minimum=mu_lo, maximum=mu_hi, value=mu_val, step=mu_step, interactive=mu_inter)
                        k_slider = gr.Slider(label="K", minimum=k_lo, maximum=k_hi, value=k_val, step=k_step, interactive=k_inter)
                        delta_slider = gr.Slider(label="delta", minimum=d_lo, maximum=d_hi, value=d_val, step=d_step, interactive=d_inter)
                        omega_slider = gr.Slider(label="omega", minimum=o_lo, maximum=o_hi, value=o_val, step=o_step, interactive=o_inter)

                        n_samples = gr.Slider(minimum=1000, maximum=20000, value=DEFAULT_NUM_SAMPLES, step=500, label="Number of samples")
                        seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")
                        run_btn = gr.Button("Generate and Compare", variant="primary")

                    with gr.Column(scale=8):
                        with gr.Row():
                            qq_plot = gr.Plot(label="QQ plot")
                            density_plot = gr.Plot(label="Histogram + KDE")
                        cdf_plot = gr.Plot(label="Empirical CDF comparison")
                        metrics_df = gr.Dataframe(
                            headers=["metric", "value", "meaning"],
                            datatype=["str", "str", "str"],
                            label="Metrics",
                            interactive=False,
                        )
                        error_md = gr.Markdown(visible=True)

                slider_outputs = [m_slider, mu_slider, k_slider, delta_slider, omega_slider]

                on_change_outputs = [model_card] + slider_outputs
                version_dd.change(
                    fn=lambda v: _on_model_or_extrapolation_change(v, registry),
                    inputs=[version_dd],
                    outputs=on_change_outputs,
                )

                reload_btn.click(
                    fn=lambda: _reload_registry(registry),
                    inputs=[],
                    outputs=[version_dd],
                )

                scenario_dd.change(
                    fn=on_scenario_change,
                    inputs=[scenario_dd],
                    outputs=slider_outputs + [scenario_card],
                )

                run_btn.click(
                    fn=lambda v, m, mu, k, d, o, n, s, sc: _run_callback(
                        v, m, mu, k, d, o, n, s, sc, registry
                    ),
                    inputs=[
                        version_dd, m_slider, mu_slider, k_slider,
                        delta_slider, omega_slider, n_samples, seed, scenario_dd,
                    ],
                    outputs=[qq_plot, density_plot, cdf_plot, metrics_df, error_md],
                )

    return demo


app = build_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Launch the app publicly")
    args = parser.parse_args()

    app.launch(
        theme=Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="zinc"),
        css=APP_CSS,
        share=args.share,
    )
