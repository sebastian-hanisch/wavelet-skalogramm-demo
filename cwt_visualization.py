"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cwt_constants as C
import cwt_morlet as M
from cwt_evaluation import Analysis


def _fixed(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def power_db(power: np.ndarray) -> np.ndarray:
    """Leistung in dB gegen das Maximum, nach unten bei DB_FLOOR abgeschnitten."""
    top = float(power.max())
    if top <= 0.0:
        return np.full(power.shape, C.DB_FLOOR)
    return np.maximum(10.0 * np.log10(np.maximum(power, 1e-300) / top), C.DB_FLOOR)


def build_signal(a: Analysis) -> go.Figure:
    """Messsignal; bei Rauschen zusätzlich das saubere Signal; senkrechte Linie beim Zeitpunkt des Frequenzprofils."""
    t = np.arange(len(a.x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.x, mode="lines", line=dict(color=C.COLOR_SIGNAL, width=1), name="Messsignal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    if a.settings.has_noise:
        fig.add_trace(go.Scatter(x=t, y=a.signal.x, mode="lines", line=dict(color=C.COLOR_TRUE, width=1, dash="dot"), name="sauberes Signal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_vline(x=a.t_index, line=dict(color=C.COLOR_AUX, width=1.5))
    fig.update_xaxes(title_text="Zeit (Abtastwerte); orange: Zeitpunkt des Frequenzprofils", range=[0, len(a.x) - 1])
    fig.update_yaxes(title_text="Signal")
    return _fixed(fig, height=240)


def build_scalogram(a: Analysis) -> go.Figure:
    """Skalogramm in dB gegen das Maximum mit logarithmischer Frequenzachse; Einflusskegel (gestrichelt weiß), wahre Frequenz (rot gestrichelt), Kamm (grün), Schnittlinien (orange)."""
    s = a.settings
    n = s.n
    fig = go.Figure(go.Heatmap(x=np.arange(n), y=a.freqs, z=power_db(a.power), colorscale="Viridis", zmin=C.DB_FLOOR, zmax=0.0, colorbar=dict(title="dB", thickness=12),
                               hovertemplate="t = %{x}, f = %{y:.3f}: %{z:.1f} dB<extra></extra>"))
    edge = M.coi_edge(M.scale_of(a.freqs, s.w0))
    fig.add_trace(go.Scatter(x=edge, y=a.freqs, mode="lines", line=dict(color="white", width=1, dash="dash"), name="Einflusskegel (√2 s)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=n - 1 - edge, y=a.freqs, mode="lines", line=dict(color="white", width=1, dash="dash"), showlegend=False, hoverinfo="skip"))
    tt = np.linspace(0, n - 1, 400)
    for i, track in enumerate(a.signal.tracks):
        fig.add_trace(go.Scatter(x=tt, y=track(tt), mode="lines", line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), name="wahre Frequenz" if i == 0 else "wahre Frequenz (zweite)", showlegend=i == 0))
    if a.ridge is not None:
        fig.add_trace(go.Scatter(x=np.arange(n), y=a.ridge, mode="lines", line=dict(color=C.COLOR_RIDGE, width=1.4), name="Kamm"))
    for e in a.signal.events:
        fig.add_vline(x=e, line=dict(color=C.COLOR_TRUE, width=1, dash="dash"))
    fig.add_vline(x=a.t_index, line=dict(color=C.COLOR_AUX, width=1.5))
    fig.add_hline(y=float(a.freqs[a.f_index]), line=dict(color=C.COLOR_AUX, width=1.2, dash="dot"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", range=[0, n - 1])
    fig.update_yaxes(title_text="Frequenz (Zyklen je Abtastwert, logarithmisch)", type="log", range=[math.log10(a.freqs[0]), math.log10(a.freqs[-1])])
    return _fixed(fig, height=440)


def build_profile(a: Analysis) -> go.Figure:
    """Skalenprofil zum Zeitpunkt des orangen Schnitts (dB gegen das Maximum des ganzen Skalogramms) mit den wahren Frequenzen."""
    col = a.power[:, a.t_index]
    top = float(a.power.max())
    db = np.maximum(10.0 * np.log10(np.maximum(col, 1e-300) / top), C.DB_FLOOR) if top > 0 else np.full(len(col), C.DB_FLOOR)
    fig = go.Figure(go.Scatter(x=a.freqs, y=db, mode="lines", line=dict(color=C.COLOR_EST, width=1.4), name="Profil", hovertemplate="f = %{x:.3f}: %{y:.1f} dB<extra></extra>"))
    for track in a.signal.tracks:
        fig.add_vline(x=float(track(np.array([a.t_index]))[0]), line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Frequenz (Zyklen je Abtastwert, logarithmisch); rot gestrichelt: wahr", type="log", range=[math.log10(a.freqs[0]), math.log10(a.freqs[-1])])
    fig.update_yaxes(title_text="dB gegen Maximum des Skalogramms", range=[C.DB_FLOOR, 3])
    return _fixed(fig, height=280)


def build_timeline(a: Analysis) -> go.Figure:
    """Zeitverlauf der Leistung bei der Frequenz des orangen Schnitts."""
    row = a.power[a.f_index]
    fig = go.Figure(go.Scatter(x=np.arange(len(row)), y=row, mode="lines", line=dict(color=C.COLOR_EST, width=1.4), name="Leistung", hovertemplate="t = %{x}: %{y:.3g}<extra></extra>"))
    for e in a.signal.events:
        fig.add_vline(x=e, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text=f"Zeit (Abtastwerte); rot gestrichelt: Impulse bzw. Burst", range=[0, len(row) - 1])
    fig.update_yaxes(title_text=f"Leistung bei f = {a.freqs[a.f_index]:.3f}".replace(".", ","), rangemode="tozero")
    return _fixed(fig, height=260)


def build_resolution_freq(res) -> go.Figure:
    rows = res["rows"]
    fs = [r["f"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fs, y=[r["ratio"] for r in rows], mode="lines+markers", name=f"Wavelet (ω₀ = {res['w0']})", hovertemplate="f = %{x}: Verhältnis %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fs, y=[r["stft_ratio"] for r in rows], mode="lines+markers", name=f"Spektrogramm (Hann, L = {res['length']})", hovertemplate="f = %{x}: Verhältnis %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Frequenz", type="log")
    fig.update_yaxes(title_text="Frequenzverhältnis, ab dem zwei Töne getrennt sind", rangemode="tozero")
    return _fixed(fig, height=320)


def build_resolution_time(res) -> go.Figure:
    rows = res["rows"]
    fs = [r["f"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fs, y=[r["robust"] for r in rows], mode="lines+markers", name=f"Wavelet (ω₀ = {res['w0']})", hovertemplate="f = %{x}: %{y} Abtastwerte<extra></extra>"))
    fig.add_trace(go.Scatter(x=fs, y=[res["stft_distance"]] * len(fs), mode="lines+markers", name=f"Spektrogramm (Hann, L = {res['length']})", hovertemplate="f = %{x}: %{y} Abtastwerte<extra></extra>"))
    fig.update_xaxes(title_text="Frequenz", type="log")
    fig.update_yaxes(title_text="Abstand, ab dem zwei Impulse sicher getrennt sind", type="log")
    return _fixed(fig, height=320)


def build_omega(rows) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    w0s = [r["w0"] for r in rows]
    fig.add_trace(go.Scatter(x=w0s, y=[r["ratio"] for r in rows], mode="lines+markers", name="Frequenzverhältnis (Töne)", hovertemplate="ω₀ = %{x}: %{y:.2f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=w0s, y=[r["robust"] for r in rows], mode="lines+markers", name="Impulsabstand (Abtastwerte)", hovertemplate="ω₀ = %{x}: %{y}<extra></extra>"), secondary_y=True)
    fig.update_xaxes(title_text="ω₀", type="log")
    fig.update_yaxes(title_text="Frequenzverhältnis", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Impulsabstand", secondary_y=True, rangemode="tozero")
    return _fixed(fig, height=340)


def build_cone(res) -> go.Figure:
    d = np.linspace(0.0, 3.0, 61)
    fig = go.Figure(go.Scatter(x=d, y=M.cone_ratio(d), mode="lines", line=dict(color="black", width=1.4), name="Theorie Φ(d / s)"))
    for r in res["rows"]:
        fig.add_trace(go.Scatter(x=[0, 1, math.sqrt(2), 2], y=[r["r0"], r["r1"], r["r_coi"], r["r2"]], mode="markers", name=f"ω₀ = {r['w0']}, f = {r['f']:g}".replace(".", ","),
                                 hovertemplate="d/s = %{x:.2f}: %{y:.3f}<extra></extra>"))
    fig.add_vline(x=math.sqrt(2), line=dict(color="gray", width=1, dash="dot"), annotation_text="√2 s", annotation_position="bottom right")
    fig.update_xaxes(title_text="Abstand zum Rand in Vielfachen von s")
    fig.update_yaxes(title_text="Amplitude / Amplitude im Inneren", range=[0.4, 1.02])
    return _fixed(fig, height=340)


def build_chirp(rows) -> go.Figure:
    fig = go.Figure()
    for snr in sorted({r["snr"] for r in rows}, reverse=True):
        sub = [r for r in rows if r["snr"] == snr]
        fig.add_trace(go.Scatter(x=[r["w0"] for r in sub], y=[max(r["rmse"], 1e-5) for r in sub], mode="lines+markers", name=f"{snr} dB", hovertemplate=f"{snr} dB, ω₀ = %{{x}}: %{{y:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text="ω₀", type="log")
    fig.update_yaxes(title_text="RMSE des Kamms (bis 1e-5 abgeschnitten)", type="log")
    return _fixed(fig, height=340)


def build_scaling(res) -> go.Figure:
    tones = res["tones"]
    labels = [f"f = {r['f']:g}".replace(".", ",") for r in tones]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["rel_l2"] for r in tones], name="|W|² (L2)", hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["rel_rect"] for r in tones], name="|W|² / s (gleichgewichtet)", hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_yaxes(title_text=f"Spitze relativ zu f = {tones[0]['f']:g}".replace(".", ","), rangemode="tozero")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=320)


def build_correlation(res) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Zeitabstand", "Frequenzunterschied"))
    t = res["time_corr"]
    f = res["freq_corr"]
    fig.add_trace(go.Scatter(x=[r["lag_s"] for r in t], y=[r["corr"] for r in t], mode="lines+markers", name="gemessen", hovertemplate="%{x} s: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["lag_s"] for r in t], y=[r["theory"] for r in t], mode="lines", line=dict(dash="dot"), name="Theorie", hovertemplate="%{x} s: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["rel"] for r in f], y=[r["corr"] for r in f], mode="lines+markers", name="gemessen", showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[r["rel"] for r in f], y=[r["theory"] for r in f], mode="lines", line=dict(dash="dot"), name="Theorie", showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Abstand in Vielfachen von s", row=1, col=1)
    fig.update_xaxes(title_text="relativer Frequenzunterschied", row=1, col=2)
    fig.update_yaxes(title_text="Korrelation der Zellleistungen", range=[0, 1])
    return _fixed(fig, height=320)
