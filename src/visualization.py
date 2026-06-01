"""
Візуалізація результатів кластеризації (мінімалістичний академічний стиль).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import List

ACADEMIC_PALETTE = ["#1e40af", "#3b82f6", "#64748b", "#94a3b8", "#cbd5e1", "#0f172a"]


def _apply_matplotlib_academic_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4a4a4a",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#cccccc",
    })


def plot_cluster_timeline(labels: np.ndarray, title: str = "Розподіл кадрів за кластерами"):
    """Timeline кластерів (matplotlib)."""
    _apply_matplotlib_academic_style()
    fig, ax = plt.subplots(figsize=(11, 2.6))
    colors = [ACADEMIC_PALETTE[int(c) % len(ACADEMIC_PALETTE)] for c in labels]
    ax.scatter(range(len(labels)), labels, c=colors, s=18, alpha=0.85, edgecolors="none")
    ax.set_xlabel("Номер кадра")
    ax.set_ylabel("Кластер")
    ax.set_title(title, fontsize=11, fontweight="normal")
    ax.set_yticks(sorted(np.unique(labels)))
    return fig


def plot_cluster_timeline_plotly(labels: np.ndarray, title: str):
    """Інтерактивний timeline (Plotly, приглушена палітра)."""
    fig = go.Figure(
        data=go.Scatter(
            x=list(range(len(labels))),
            y=labels,
            mode="markers",
            marker=dict(
                size=6,
                color=labels,
                colorscale=[[0, ACADEMIC_PALETTE[0]], [1, ACADEMIC_PALETTE[-1]]],
                showscale=False,
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=320,
        margin=dict(l=48, r=24, t=40, b=36),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        xaxis_title="Номер кадра",
        yaxis_title="Кластер",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=11, color="#333"),
    )
    return fig


def plot_temporal_psnr(summary_df, y_col: str = "Середній PSNR (temporal)"):
    """Стовпчикова діаграма temporal PSNR по кластерах."""
    if summary_df is None or summary_df.empty:
        return None
    fig = px.bar(
        summary_df,
        x="Кластер",
        y=y_col,
        color_discrete_sequence=[ACADEMIC_PALETTE[0]],
    )
    fig.update_layout(
        height=360,
        margin=dict(l=48, r=24, t=44, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        showlegend=False,
        font=dict(size=11, color="#334155"),
        title=dict(text="Середній temporal PSNR за кластером", font=dict(size=13)),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def plot_policy_comparison(per_cluster_df):
    """Порівняння uniform vs adaptive PSNR."""
    if per_cluster_df is None or per_cluster_df.empty:
        return None
    fig = px.bar(
        per_cluster_df,
        x="Кластер",
        y=["Uniform PSNR", "Adaptive PSNR"],
        barmode="group",
        color_discrete_map={"Uniform PSNR": "#94a3b8", "Adaptive PSNR": "#2563eb"},
        labels={"value": "PSNR (dB)", "variable": "Метод"},
    )
    fig.update_layout(
        height=380,
        margin=dict(l=48, r=24, t=44, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(size=11, color="#333"),
        title=dict(text="Порівняння стратегій інтерполяції", font=dict(size=13)),
        legend_title="",
    )
    return fig


def get_cluster_representatives(
    frames: List[np.ndarray],
    labels: np.ndarray,
    max_per_cluster: int = 2,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Повертає {cluster_id: [(frame_idx, frame), ...]} для відображення в UI."""
    result: dict[int, list[tuple[int, np.ndarray]]] = {}
    for cluster in np.unique(labels):
        idxs = np.where(labels == cluster)[0]
        result[int(cluster)] = [(int(idxs[j]), frames[idxs[j]]) for j in range(min(max_per_cluster, len(idxs)))]
    return result