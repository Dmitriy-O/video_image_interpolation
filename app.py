"""
Адаптивна інтерполяція відео на основі model-based кластеризації.

Інтерактивна демонстрація курсового проєкту (Gradio).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.feature_extraction import extract_frame_features, discretize_features
from src.clustering import (
    run_ward_clustering,
    run_kmodes_clustering,
    get_cluster_statistics,
    compare_clusterings,
    compute_cluster_profiles,
)
from src.interpolation import (
    compute_temporal_interp_errors,
    simulate_adaptive_policies,
    recommend_policies_from_profiles,
)
from src.visualization import (
    plot_cluster_timeline_plotly,
    plot_temporal_psnr,
    plot_policy_comparison,
    get_cluster_representatives,
)

DEFAULT_FEATURES = [
    "mean_b", "mean_g", "mean_r",
    "std_b", "std_g", "std_r",
    "brightness", "contrast", "texture_laplacian",
    "motion_mean", "motion_std",
]

POLICY_OPTIONS = [
    "linear", "mild", "moderate", "conservative",
    "strong", "very_strong", "hold",
    "biased", "motion_aware", "forward_biased"
]

APP_CSS = """
/* ——— ноутбук: 1024px – 1600px ——— */
@media (max-width: 1023px) {
    .laptop-only-hint {
        display: flex !important;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 2rem;
        text-align: center;
        font-size: 1.05rem;
        color: #475569;
        background: #f8fafc;
    }
    .laptop-app { display: none !important; }
}
@media (min-width: 1024px) {
    .laptop-only-hint { display: none !important; }
}

.gradio-container {
    min-width: 1024px !important;
    max-width: 1280px !important;
    width: calc(100vw - 48px) !important;
    margin: 0 auto !important;
    padding: 0 8px 20px !important;
    background: linear-gradient(180deg, #f5f7fb 0%, #eef1f8 50%, #f3f5fa 100%) !important;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif !important;
}

.app-hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 0.4rem 0.85rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    margin-bottom: 0.85rem;
}
.app-hero .mark {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(145deg, #5b8def 0%, #7c9cf5 100%);
    box-shadow: 0 4px 14px rgba(91, 141, 239, 0.3);
    flex-shrink: 0;
}
.app-hero h1 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #1e293b;
}
.app-hero .subtitle {
    margin: 0.1rem 0 0;
    font-size: 0.8rem;
    color: #64748b;
}

.panel-card {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(226, 232, 240, 0.85) !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 24px rgba(30, 41, 59, 0.06) !important;
    padding: 1rem 1.1rem !important;
    backdrop-filter: blur(6px);
}
.panel-card h5, .panel-card .prose h5 {
    margin-top: 0 !important;
    color: #334155 !important;
    font-weight: 600 !important;
}

.workspace-card {
    background: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid #e8ecf4 !important;
    box-shadow: 0 8px 28px rgba(30, 41, 59, 0.05) !important;
    padding: 0.85rem 1rem 1rem !important;
    margin-top: 0.35rem !important;
}

.results-strip {
    gap: 1rem !important;
    margin: 0.85rem 0 0.5rem !important;
    align-items: stretch !important;
}
.results-strip > .gap { flex: 0 0 38% !important; max-width: 38% !important; }
.results-strip > .gap:last-child { flex: 1 1 62% !important; max-width: 62% !important; }
.metrics-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.65rem;
}
.metric-chip {
    background: linear-gradient(180deg, #fff 0%, #f8fafc 100%);
    border: 1px solid #e8edf5;
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.metric-chip:hover {
    box-shadow: 0 4px 14px rgba(91, 141, 239, 0.12);
    transform: translateY(-1px);
}
.metric-chip .label {
    font-size: 0.68rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.metric-chip .value {
    font-size: 1.12rem;
    font-weight: 600;
    color: #1e293b;
    margin-top: 0.2rem;
}
.status-ok {
    color: #047857;
    background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-size: 0.88rem;
    line-height: 1.45;
}
.status-err {
    color: #b91c1c;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-size: 0.88rem;
}

.tab-nav button {
    font-weight: 500 !important;
    padding: 0.55rem 1.1rem !important;
    border-radius: 8px 8px 0 0 !important;
}
.tabitem {
    padding-top: 1rem !important;
}
.tabitem .plot-container { min-height: 300px; }

footer { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#eef1f8",
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_shadow="0 4px 18px rgba(30, 41, 59, 0.05)",
    block_title_text_weight="600",
    block_label_text_color="#64748b",
    button_primary_background_fill="#5b8def",
    button_primary_background_fill_hover="#4a7de0",
    button_large_radius="12px",
    button_large_text_weight="600",
    input_background_fill="#f8fafc",
    slider_color="#5b8def",
)


def _format_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return profiles
    out = profiles.copy()
    out = out.rename(columns={"cluster": "Кластер", "size": "Розмір"})
    rename = {
        c: c.replace("_mean", " (сер.)").replace("_std", " (σ)")
        for c in out.columns
        if c not in ("Кластер", "Розмір")
    }
    return out.rename(columns=rename)


def _format_comparison(cmp: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"Метрика": "Adjusted Rand Index (ARI)", "Значення": cmp["adjusted_rand_index"]},
        {"Метрика": "Normalized Mutual Information (NMI)", "Значення": cmp["normalized_mutual_info"]},
        {"Метрика": "Точна згода (%)", "Значення": cmp["exact_agreement_percent"]},
    ])


def _policies_to_df(policy_map: dict[int, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Кластер": list(policy_map.keys()),
        "Стратегія": list(policy_map.values()),
    })


def _df_to_policies(df: pd.DataFrame | None) -> dict[int, str]:
    if df is None or len(df) == 0:
        return {}
    return {int(row["Кластер"]): str(row["Стратегія"]) for _, row in df.iterrows()}


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _status_html(message: str, *, error: bool = False) -> str:
    cls = "status-err" if error else "status-ok"
    return f'<div class="{cls}">{message}</div>'


def _summary_html(n_frames: int, k: int, ari: float, worst_psnr: float) -> str:
    return f"""
    <div class="metrics-bar">
        <div class="metric-chip"><div class="label">Кадрів</div><div class="value">{n_frames}</div></div>
        <div class="metric-chip"><div class="label">Кластерів</div><div class="value">{k}</div></div>
        <div class="metric-chip"><div class="label">ARI</div><div class="value">{ari:.3f}</div></div>
        <div class="metric-chip"><div class="label">Мін. PSNR</div><div class="value">{worst_psnr:.1f} dB</div></div>
    </div>
    """


def _resolve_video_path(video) -> str | None:
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("path") or video.get("name")
    return str(video)


def _load_frames(video_path: str | None, max_frames: int, target_size: tuple[int, int] = (320, 180)) -> tuple[list, str]:
    video_path = _resolve_video_path(video_path)
    if not video_path:
        return [], "Завантажте відеозапис."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Не вдалося відкрити відеофайл."

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, target_size))
    cap.release()

    if len(frames) < 12:
        return [], f"Недостатньо кадрів ({len(frames)}). Потрібно щонайменше 12."

    note = ""
    if len(frames) > max_frames:
        step = max(1, len(frames) // max_frames)
        frames = frames[::step]
        note = f" Субдискретизація: крок {step}."

    return frames, note


def process_video(
    video_path: str | None,
    n_clusters: int,
    use_motion: bool,
    n_bins: int,
    random_seed: int,
    max_frames: int,
    sample_step: int,
):
    """Повний пайплайн; повертає outputs + стан для Gradio State."""
    empty_plot = None
    empty_df = pd.DataFrame()
    empty_gallery: list = []
    null_state = None

    frames, load_msg = _load_frames(video_path, max_frames)
    if not frames:
        return (
            _status_html(load_msg, error=True),
            "",
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_plot,
            empty_plot,
            empty_plot,
            empty_gallery,
            empty_df,
            null_state,
            int(random_seed),
            int(sample_step),
        )

    feature_df = extract_frame_features(frames, use_motion=use_motion)
    feat_for_cluster = feature_df[["frame_idx"] + [f for f in DEFAULT_FEATURES if f in feature_df.columns]]
    feature_disc = discretize_features(feat_for_cluster, n_bins=n_bins)

    ward_labels = run_ward_clustering(feat_for_cluster, n_clusters, random_state=random_seed)
    kmodes_labels = run_kmodes_clustering(feature_disc, n_clusters, random_state=random_seed)

    profiles = compute_cluster_profiles(feature_df, ward_labels)
    clustering_cmp = compare_clusterings(ward_labels, kmodes_labels)
    _, temporal_summary = compute_temporal_interp_errors(
        frames, ward_labels, sample_step=sample_step, random_state=random_seed
    )

    ward_fig = plot_cluster_timeline_plotly(ward_labels, "Ward — розподіл міток у часі")
    kmodes_fig = plot_cluster_timeline_plotly(kmodes_labels, "KModes — розподіл міток у часі")
    temporal_fig = plot_temporal_psnr(temporal_summary)

    gallery = []
    for cl, items in get_cluster_representatives(frames, ward_labels, max_per_cluster=2).items():
        for idx, img in items:
            gallery.append((_bgr_to_rgb(img), f"Кластер {cl}, кадр {idx}"))

    # Return to all-linear as the standard starting point after analysis.
    # "Рекомендувати" will then suggest changes only where non-linear policies are actually better.
    policies = {int(c): "linear" for c in np.unique(ward_labels)}
    policy_df = _policies_to_df(policies)

    worst_psnr = (
        float(temporal_summary["Середній PSNR (temporal)"].min())
        if not temporal_summary.empty else 0.0
    )
    summary = _summary_html(
        len(frames), n_clusters,
        clustering_cmp["adjusted_rand_index"], worst_psnr,
    )
    status = _status_html(f"Готово — проаналізовано {len(frames)} кадрів.{load_msg}")

    ctx = {
        "frames": frames,
        "ward_labels": ward_labels,
        "profiles": profiles,
        "sample_step": int(sample_step),
        "random_seed": int(random_seed),
    }

    return (
        status,
        summary,
        get_cluster_statistics(ward_labels),
        get_cluster_statistics(kmodes_labels),
        _format_comparison(clustering_cmp),
        _format_profiles(profiles),
        temporal_summary,
        ward_fig,
        kmodes_fig,
        temporal_fig,
        gallery,
        policy_df,
        ctx,
        int(random_seed),
        int(sample_step),
    )


def recommend_policies_ui(ctx: dict | None):
    if ctx is None:
        return pd.DataFrame(columns=["Кластер", "Стратегія"]), _status_html("Спочатку запустіть аналіз.", error=True)
    raw_profiles = ctx.get("profiles")
    if raw_profiles is None or raw_profiles.empty:
        return pd.DataFrame(columns=["Кластер", "Стратегія"]), _status_html("Профілі недоступні.", error=True)

    rec = recommend_policies_from_profiles(
        raw_profiles,
        frames=ctx.get("frames"),
        labels=ctx.get("ward_labels"),
        sample_step=ctx.get("sample_step", 2),
        random_state=ctx.get("random_seed", 42),
    )
    return _policies_to_df(rec), _status_html("Рекомендації застосовано.")


def run_simulation(
    policy_df: pd.DataFrame | None,
    ctx: dict | None,
    sample_step: int,
    random_seed: int,
):
    if ctx is None:
        return None, _status_html("Спочатку запустіть аналіз.", error=True), None

    frames = ctx.get("frames")
    ward_labels = ctx.get("ward_labels")
    if frames is None or ward_labels is None:
        return None, _status_html("Контекст обробки порожній.", error=True), None

    policies = _df_to_policies(policy_df)
    if not policies:
        return None, _status_html("Задайте стратегії для кластерів.", error=True), None

    step = int(ctx.get("sample_step", sample_step))
    seed = int(ctx.get("random_seed", random_seed))

    sim = simulate_adaptive_policies(
        frames, ward_labels, policies,
        sample_step=step, random_state=seed,
        profiles=ctx.get("profiles"),   # enables motion-dependent fine-grained alpha
        # policy_smoothing_window=9  # stronger default temporal smoothing
    )

    fig = plot_policy_comparison(sim["per_cluster"])
    delta_color = "#166534" if sim["delta"] >= 0 else "#991b1b"
    delta_ssim_color = "#166534" if sim.get("delta_ssim", 0) >= 0 else "#991b1b"

    text = f"""
    <div class="metrics-bar">
        <div class="metric-chip">
            <div class="label">Базова (PSNR)</div>
            <div class="value">{sim['overall_uniform']:.2f} dB</div>
        </div>
        <div class="metric-chip">
            <div class="label">Адаптивна (PSNR)</div>
            <div class="value">{sim['overall_adaptive']:.2f} dB</div>
        </div>
        <div class="metric-chip">
            <div class="label">Δ PSNR</div>
            <div class="value" style="color:{delta_color}">{sim['delta']:+.2f} dB</div>
        </div>
    </div>
    <div class="metrics-bar" style="margin-top: 8px;">
        <div class="metric-chip">
            <div class="label">Базова (SSIM)</div>
            <div class="value">{sim.get('overall_uniform_ssim', 0):.4f}</div>
        </div>
        <div class="metric-chip">
            <div class="label">Адаптивна (SSIM)</div>
            <div class="value">{sim.get('overall_adaptive_ssim', 0):.4f}</div>
        </div>
        <div class="metric-chip">
            <div class="label">Δ SSIM</div>
            <div class="value" style="color:{delta_ssim_color}">{sim.get('delta_ssim', 0):+.4f}</div>
        </div>
    </div>
    <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">
        SSIM — додаткова метрика структурної подібності (вторинна до PSNR). 
        Політики застосовуються з temporal smoothing (сегментний режим) для зменшення хаотичних перемикань.
    </div>
    """

    export = {
        "policies": policies,
        "overall_uniform_db": sim["overall_uniform"],
        "overall_adaptive_db": sim["overall_adaptive"],
        "delta_db": sim["delta"],
    }
    export_path = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(export, export_path, ensure_ascii=False, indent=2)
    export_path.close()

    per_cluster = sim["per_cluster"] if not sim["per_cluster"].empty else pd.DataFrame()
    return fig, text, (export_path.name, per_cluster)


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="Адаптивна інтерполяція відео",
        theme=THEME,
        css=APP_CSS,
    ) as demo:
        gr.HTML(
            '<div class="laptop-only-hint">'
            "Цей інтерфейс розрахований на ноутбук.<br>"
            "Відкрийте на екрані шириною від <b>1024 px</b>."
            "</div>"
        )

        with gr.Column(elem_classes=["laptop-app"]):
            gr.HTML(
                """
                <div class="app-hero">
                    <div class="mark"></div>
                    <div>
                        <h1>Адаптивна інтерполяція відео</h1>
                        <p class="subtitle">Кластеризація контенту та адаптивні стратегії</p>
                    </div>
                </div>
                """
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    video_in = gr.Video(label="Відеозапис", height=220)
                with gr.Column(scale=6, elem_classes=["panel-card"]):
                    gr.Markdown("##### Параметри аналізу")

                    with gr.Row():
                        n_clusters = gr.Slider(2, 6, value=4, step=1, label="Кластерів (k)")
                        n_bins = gr.Slider(3, 8, value=5, step=1, label="Біни KModes")

                    with gr.Row():
                        use_motion = gr.Checkbox(value=True, label="Ознаки руху")
                        random_seed = gr.Number(value=42, label="Seed", precision=0)

                    with gr.Row():
                        max_frames = gr.Slider(60, 600, value=300, step=30, label="Макс. кадрів")
                        sample_step = gr.Slider(1, 5, value=2, step=1, label="Крок PSNR")

                    process_btn = gr.Button("Запустити аналіз", variant="primary", size="lg")

            with gr.Row(elem_classes=["results-strip"]):
                status_out = gr.HTML()
                summary_out = gr.HTML()

            state_ctx = gr.State()
            state_seed = gr.State(value=42)
            state_step = gr.State(value=2)

            with gr.Column(elem_classes=["workspace-card"]):
                with gr.Tabs():
                    with gr.Tab("Кластеризація"):
                        with gr.Row():
                            ward_stats = gr.Dataframe(label="Ward", interactive=False)
                            kmodes_stats = gr.Dataframe(label="KModes", interactive=False)
                        with gr.Row():
                            ward_plot = gr.Plot(label="Timeline — Ward")
                            kmodes_plot = gr.Plot(label="Timeline — KModes")
                        cmp_table = gr.Dataframe(label="Узгодженість (ARI · NMI)", interactive=False)

                    with gr.Tab("Профілі"):
                        with gr.Row():
                            with gr.Column(scale=5):
                                profiles_table = gr.Dataframe(
                                    label="Середні ознаки", interactive=False,
                                )
                            with gr.Column(scale=4):
                                rep_gallery = gr.Gallery(
                                    label="Репрезентативні кадри",
                                    columns=2,
                                    rows=2,
                                    height=340,
                                    object_fit="contain",
                                )

                    with gr.Tab("Якість"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                temporal_table = gr.Dataframe(
                                    label="Temporal PSNR", interactive=False,
                                )
                            with gr.Column(scale=6):
                                temporal_plot = gr.Plot(label="PSNR по кластерах")

                    with gr.Tab("Стратегії"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                policy_table = gr.Dataframe(
                                    label="Політики по кластерах",
                                    headers=["Кластер", "Стратегія"],
                                    datatype=["number", "str"],
                                    col_count=2,
                                    interactive=True,
                                )
                                with gr.Row():
                                    sim_btn = gr.Button("Порівняти", variant="primary", scale=2)
                                    rec_btn = gr.Button("Рекомендувати", scale=1)
                                sim_result_md = gr.HTML()
                                export_file = gr.File(label="Експорт")
                            with gr.Column(scale=6):
                                sim_plot = gr.Plot(label="Порівняння стратегій")
                                sim_table = gr.Dataframe(label="Деталізація", interactive=False)

                    with gr.Tab("Методологія"):
                        gr.Markdown(
                            """
                            | Етап | Опис |
                            |------|------|
                            | Ознаки | BGR, яскравість, контраст, Laplacian, різниця кадрів |
                            | Кластеризація | Ward (неперервні) · KModes (дискретні) |
                            | Валідація | ARI, NMI між розбиттями |
                            | Оцінка | Temporal triplets → PSNR + SSIM |
                            | Адаптація | fine-grained (0.10–0.62): linear, mild, moderate, conservative, strong, very_strong, hold, biased/motion_aware, forward_biased |
                            """
                        )

            process_btn.click(
                fn=process_video,
                inputs=[video_in, n_clusters, use_motion, n_bins, random_seed, max_frames, sample_step],
                outputs=[
                    status_out,
                    summary_out,
                    ward_stats,
                    kmodes_stats,
                    cmp_table,
                    profiles_table,
                    temporal_table,
                    ward_plot,
                    kmodes_plot,
                    temporal_plot,
                    rep_gallery,
                    policy_table,
                    state_ctx,
                    state_seed,
                    state_step,
                ],
            )

            rec_btn.click(
                fn=recommend_policies_ui,
                inputs=[state_ctx],
                outputs=[policy_table, sim_result_md],
            )

            def _sim_wrapper(policy_df, ctx, step, seed):
                fig, text, extra = run_simulation(policy_df, ctx, step, seed)
                if extra is None:
                    return fig, text, pd.DataFrame(), None
                path, per_cluster = extra
                return fig, text, per_cluster, path

            sim_btn.click(
                fn=_sim_wrapper,
                inputs=[policy_table, state_ctx, state_step, state_seed],
                outputs=[sim_plot, sim_result_md, sim_table, export_file],
            )

    return demo


if __name__ == "__main__":
    build_interface().launch(
        server_name="127.0.0.1",
        show_error=True,
        css=APP_CSS,
        theme=THEME,
    )