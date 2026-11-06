"""
Dashboard Charts (`app/dashboard/charts.py`).

Each builder receives data and returns a Plotly figure. Nothing here holds a result.

Conventions: one y-axis per chart, fixed categorical slots so a color always means the same thing,
hairline solid grids, thin marks, text in ink colors rather than series colors, and hover detail on
every mark. Dashed lines appear only for thresholds. Every chart has a table beside it on its page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import STAGES


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    deemphasis: str
    band: str
    series: Tuple[str, ...]


LIGHT = Theme(
    surface="#ffffff",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    deemphasis="#c3c2b7",
    band="rgba(137,135,129,0.14)",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
)
DARK = Theme(
    surface="#0e1117",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    deemphasis="#55554f",
    band="rgba(137,135,129,0.18)",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
)


def theme() -> Theme:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def _finish(fig: go.Figure, t: Theme, height: int, legend: bool = False, x_title: str = "", y_title: str = "", top: int = 28) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=64, t=top, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=t.ink2),
        hoverlabel=dict(bgcolor=t.surface, bordercolor=t.axis, font=dict(color=t.ink)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal", font=dict(color=t.ink2)),
        bargap=0.38,
    )
    axis = dict(gridcolor=t.grid, linecolor=t.axis, zeroline=False, tickfont=dict(color=t.muted), automargin=True)
    fig.update_xaxes(title=dict(text=x_title, font=dict(color=t.muted)), **axis)
    fig.update_yaxes(title=dict(text=y_title, font=dict(color=t.muted)), **axis)
    return fig


def _threshold(fig: go.Figure, t: Theme, value: float, text: str, vertical: bool = True) -> None:
    if vertical:
        fig.add_vline(value, line=dict(color=t.muted, width=1, dash="dash"))
        fig.add_annotation(x=value, y=1.0, yref="paper", yanchor="bottom", text=text, showarrow=False, font=dict(color=t.ink2, size=12))
    else:
        fig.add_hline(value, line=dict(color=t.muted, width=1, dash="dash"), annotation=dict(text=text, font=dict(color=t.ink2, size=12)))


def ranked_bars(
    names: Sequence[str],
    values: Sequence[float],
    unit: str,
    highlight: Optional[str] = None,
    threshold: Optional[float] = None,
    threshold_label: str = "",
    x_title: str = "",
) -> go.Figure:
    """Compare one measure across items. `highlight` is the single item the page is about."""
    t = theme()
    base = t.series[0] if highlight is None else t.deemphasis
    colors = [t.series[0] if n == highlight else base for n in names]
    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(names),
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:,.1f} {unit}" for v in values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=t.ink2),
            hovertemplate="%{y}<br>%{x:,.2f} " + unit + "<extra></extra>",
        )
    )
    if threshold is not None:
        _threshold(fig, t, threshold, threshold_label)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _finish(fig, t, height=84 + 34 * len(names), x_title=x_title)


def stacked_bars(frame: pd.DataFrame, y: str, parts: Sequence[str], x_title: str = "Median latency (ms)") -> go.Figure:
    """Show how a total splits into parts. A part keeps its slot in every chart."""
    t = theme()
    fig = go.Figure()
    for index, part in enumerate(parts):
        fig.add_trace(
            go.Bar(
                name=part.replace(" (ms)", ""),
                x=frame[part],
                y=frame[y],
                orientation="h",
                marker=dict(color=t.series[index % len(t.series)], line=dict(color=t.surface, width=2)),
                hovertemplate="%{y}<br>" + part.replace(" (ms)", "") + ": %{x:,.1f} ms<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _finish(fig, t, height=max(250, 130 + 34 * len(frame)), legend=True, x_title=x_title, top=76)


def stage_stack(frame: pd.DataFrame) -> go.Figure:
    wide = frame.pivot(index="Profile", columns="Stage", values="Median (ms)").reset_index()
    parts = [s for s in STAGES if s in wide.columns]
    return stacked_bars(wide, "Profile", parts)


def pass_spread(frame: pd.DataFrame, threshold: Optional[float], threshold_label: str) -> go.Figure:
    """Every timed pass per case, so the spread behind each median is visible."""
    t = theme()
    fig = go.Figure()
    for case, group in frame.groupby("case_id", sort=False):
        fig.add_trace(
            go.Box(
                x=group["e2e_total"],
                name=str(case),
                orientation="h",
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker=dict(color=t.series[0], size=6, line=dict(color=t.surface, width=1)),
                line=dict(color=t.muted, width=1),
                fillcolor="rgba(0,0,0,0)",
                hovertemplate=str(case) + "<br>%{x:,.1f} ms<extra></extra>",
            )
        )
    if threshold is not None:
        _threshold(fig, t, threshold, threshold_label)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _finish(fig, t, height=84 + 36 * frame["case_id"].nunique(), x_title="End-to-end latency per pass (ms)")


def patch_lines(frame: pd.DataFrame) -> go.Figure:
    t = theme()
    fig = go.Figure()
    for index, (backend, group) in enumerate(sorted(frame.groupby("Backend"), key=lambda item: item[0])):
        group = group.sort_values("Batch")
        fig.add_trace(
            go.Scatter(
                x=group["Batch"],
                y=group["Median (ms)"],
                name=backend,
                mode="lines+markers",
                line=dict(color=t.series[index % len(t.series)], width=2),
                marker=dict(size=9, color=t.series[index % len(t.series)], line=dict(color=t.surface, width=2)),
                hovertemplate=backend + "<br>Batch %{x}: %{y:,.2f} ms<extra></extra>",
            )
        )
    fig.update_xaxes(type="category")
    return _finish(fig, t, height=380, legend=True, x_title="Patches per call", y_title="Median latency (ms)", top=56)


def dice_deltas(frame: pd.DataFrame, tolerance: float) -> go.Figure:
    """Per-case Dice change from the reference, against the tolerance band."""
    t = theme()
    fig = go.Figure()
    fig.add_hrect(y0=-tolerance, y1=tolerance, fillcolor=t.band, line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=frame["Case"],
            y=frame["ΔDice"],
            mode="markers",
            marker=dict(size=11, color=t.series[0], line=dict(color=t.surface, width=2)),
            hovertemplate="%{x}<br>ΔDice %{y:+.6f}<extra></extra>",
        )
    )
    fig.add_hline(0, line=dict(color=t.axis, width=1))
    worst = frame.loc[frame["ΔDice"].abs().idxmax()]
    fig.add_annotation(x=worst["Case"], y=worst["ΔDice"], text=f"{worst['ΔDice']:+.5f}", showarrow=False, yshift=16, font=dict(color=t.ink2, size=12))
    span = max(float(frame["ΔDice"].abs().max()), tolerance) * 1.35
    fig.update_yaxes(range=[-span, span], tickformat="+.3f")
    fig.update_xaxes(tickangle=-40)
    return _finish(fig, t, height=380, y_title="ΔDice")


def bland_altman(frame: pd.DataFrame, stats: Dict[str, Any]) -> go.Figure:
    t = theme()
    fig = go.Figure(
        go.Scatter(
            x=frame["Mean volume (cm³)"],
            y=frame["ΔV (cm³)"],
            mode="markers",
            text=frame["Case"],
            marker=dict(size=11, color=t.series[0], line=dict(color=t.surface, width=2)),
            hovertemplate="%{text}<br>Mean %{x:,.1f} cm³<br>ΔV %{y:+.3f} cm³<extra></extra>",
        )
    )
    fig.add_hline(0, line=dict(color=t.axis, width=1))
    fig.add_hline(stats["bias"], line=dict(color=t.ink2, width=1.5), annotation=dict(text=f"Bias {stats['bias']:+.3f}", font=dict(color=t.ink2, size=12)))
    _threshold(fig, t, stats["loa_high"], f"Upper limit {stats['loa_high']:+.3f}", vertical=False)
    _threshold(fig, t, stats["loa_low"], f"Lower limit {stats['loa_low']:+.3f}", vertical=False)
    return _finish(fig, t, height=380, x_title="Mean of profile and reference volume (cm³)", y_title="ΔV (cm³)")


def sweep_scatter(frame: pd.DataFrame) -> go.Figure:
    """Three overlaps as hue, batch size as marker shape, so identity never rests on color alone."""
    t = theme()
    shapes = {1: "circle", 2: "square", 4: "triangle-up", 8: "diamond"}
    fig = go.Figure()
    for index, (overlap, group) in enumerate(frame.groupby("overlap")):
        fig.add_trace(
            go.Scatter(
                x=group["median_sw_latency_ms"],
                y=group["mean_dice"],
                name=f"Overlap {overlap:g}",
                mode="markers",
                customdata=group["sw_batch_size"],
                marker=dict(
                    size=13,
                    color=t.series[index % 3],
                    symbol=[shapes.get(int(b), "circle") for b in group["sw_batch_size"]],
                    line=dict(color=t.surface, width=2),
                ),
                hovertemplate="Batch %{customdata}<br>%{x:,.1f} ms<br>Dice %{y:.6f}<extra></extra>",
            )
        )
    return _finish(fig, t, height=400, legend=True, x_title="Median sliding-window latency (ms)", y_title="Mean Dice", top=48)


def relation(frame: pd.DataFrame, x: str, y: str = "Dice") -> go.Figure:
    t = theme()
    fig = go.Figure(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="markers",
            text=frame["Case"],
            marker=dict(size=11, color=t.series[0], line=dict(color=t.surface, width=2)),
            hovertemplate="%{text}<br>" + x + " %{x:,.2f}<br>" + y + " %{y:.4f}<extra></extra>",
        )
    )
    for row in (frame.loc[frame[y].idxmin()], frame.loc[frame[y].idxmax()]):
        fig.add_annotation(x=row[x], y=row[y], text=str(row["Case"]), showarrow=False, yshift=14, font=dict(color=t.ink2, size=12))
    return _finish(fig, t, height=320, x_title=x, y_title=y)


def scaling_lines(frame: pd.DataFrame) -> go.Figure:
    """Plot custom CUDA C++ kernel speedups over PyTorch CUDA across volume sizes."""
    t = theme()
    fig = go.Figure()
    series_cols = [
        ("Fused Post Speedup", "Fused Postprocessing (Softmax + Trilinear Invert + Count)"),
        ("Trilinear Resample Speedup", "3D Trilinear Resampling"),
        ("Nearest u8 Speedup", "3D Nearest-Neighbor Label Resampling"),
        ("HU Vec4 Speedup", "HU Windowing + Normalization (float4)"),
    ]
    for idx, (col, label_text) in enumerate(series_cols):
        if col not in frame.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame["Volume Shape"],
                y=frame[col],
                name=label_text,
                mode="lines+markers",
                line=dict(color=t.series[idx % len(t.series)], width=2),
                marker=dict(size=9, color=t.series[idx % len(t.series)], line=dict(color=t.surface, width=2)),
                hovertemplate=label_text + "<br>Shape %{x}: %{y:,.2f}x speedup<extra></extra>",
            )
        )
    return _finish(fig, t, height=380, legend=True, x_title="3D Volume Shape (voxels)", y_title="Speedup over PyTorch GPU (x)", top=64)
