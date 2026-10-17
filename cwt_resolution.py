"""Auflösung: zwei Töne in der Frequenz, zwei Impulse in der Zeit, Vergleich mit dem Spektrogramm, Korrelation. Reine Funktionen; Zufall nur über einen übergebenen SplitMix64.

Zwei Töne werden im Inneren (ohne Randeffekt) ausgewertet: Für cos(2 pi f t + phi) ist W(s) = (1/2) e^{i phi'} pi^(-1/4) sqrt(2 pi s) exp(-(s w - w0)^2 / 2) mit w = 2 pi f
(die Formel gleicht der FFT-Transformation; die Tests prüfen das). Zwei Impulse laufen durch die FFT-Transformation selbst.
"""

import math

import numpy as np

import cwt_morlet as M

TWO_PI = 2.0 * math.pi


def resolved(profile, i1: int, i2: int, depth: float = 0.5) -> bool:
    """Zwei Gipfel getrennt: je ein Maximum im Fenster ± halber Abstand um i1 und i2, und das Minimum dazwischen liegt unter depth mal dem kleineren Maximum."""
    lo, hi = sorted((i1, i2))
    half = max(1, int(round((hi - lo) / 2.0)))
    profile = np.asarray(profile, dtype=float)
    a = profile[max(lo - half, 0):lo + half + 1]
    b = profile[max(hi - half, 0):hi + half + 1]
    j1 = max(lo - half, 0) + int(np.argmax(a))
    j2 = max(hi - half, 0) + int(np.argmax(b))
    if j1 >= j2:
        return False
    return bool(profile[j1:j2 + 1].min() < depth * min(profile[j1], profile[j2]))


def tone_response(freqs, f: float, phase: float, w0: float) -> np.ndarray:
    """Komplexe Antwort W(s) eines Kosinus der Amplitude 1 bei der Frequenz f (Phase im Zeitpunkt der Auswertung), für die Analysefrequenzen freqs."""
    s = M.scale_of(freqs, w0)
    return 0.5 * np.exp(1j * phase) * M.PI14 * np.sqrt(TWO_PI * s) * np.exp(-0.5 * (s * TWO_PI * f - w0) ** 2)


def two_tone_profile(freqs, f1: float, f2: float, ph1: float, ph2: float, w0: float, rectified: bool = True) -> np.ndarray:
    """Skalenprofil |W|^2 (/ s) zweier Töne gleicher Amplitude im Inneren."""
    w = tone_response(freqs, f1, ph1, w0) + tone_response(freqs, f2, ph2, w0)
    p = np.abs(w) ** 2
    return p / M.scale_of(freqs, w0) if rectified else p


def two_tone_resolved_share(f1: float, ratio: float, w0: float, rng, runs: int, voices: int = 96) -> float:
    """Anteil der Läufe (zufällige Phasen), in denen zwei gleich starke Töne bei f1 und f1 * ratio im Skalenprofil (|W|^2 / s) getrennt sind (Einbruch unter 50 %)."""
    f2 = f1 * ratio
    freqs = M.freq_grid(f1 * 0.6, min(f2 * 1.5, 0.49), voices)
    i1, i2 = int(np.argmin(np.abs(freqs - f1))), int(np.argmin(np.abs(freqs - f2)))
    ok = 0
    for _ in range(runs):
        ph = TWO_PI * rng.uniform(2)
        ok += resolved(two_tone_profile(freqs, f1, f2, ph[0], ph[1], w0), i1, i2)
    return ok / runs


def min_ratio(f1: float, w0: float, make_rng, runs: int = 60, ratios=None):
    """Kleinstes Frequenzverhältnis (Raster 0,05 ab 1,05), bei dem zwei Töne in allen Läufen getrennt sind (Anteil >= 0,995); None, wenn f1 * Verhältnis über 0,45 liegt, bevor das eintritt."""
    for r in (np.arange(1.05, 3.0, 0.05) if ratios is None else ratios):
        if f1 * r > 0.45:
            return None
        if two_tone_resolved_share(f1, float(r), w0, make_rng(), runs) >= 0.995:
            return round(float(r), 2)
    return None


def impulses_resolved(f: float, d: int, w0: float, n: int = 1024) -> bool:
    """Zwei Einzelimpulse im Abstand d: im Zeitverlauf von |W|^2 bei der Analysefrequenz f getrennt (Einbruch unter 50 %)."""
    x = np.zeros(n)
    t1 = n // 2
    x[t1] = 1.0
    x[t1 + d] = 1.0
    prof = np.abs(M.cwt(x, [f], w0)[0]) ** 2
    return resolved(prof, t1, t1 + d)


def min_distance(f: float, w0: float, n: int = 1024):
    """(erster getrennter Abstand, sicherer Abstand): sicher heißt, dass alle größeren Abstände bis 6 s ebenfalls getrennt sind; der erste kann wegen der Schwebung (Phasenfaktor
    e^{i w0 d / s}) ein Scheineinbruch sein."""
    s = float(M.scale_of(f, w0))
    ds = list(range(1, int(6 * s) + 2))
    res = [impulses_resolved(f, d, w0, n) for d in ds]
    first = next((d for d, r in zip(ds, res) if r), None)
    robust = 1
    for i in range(len(ds) - 1, -1, -1):
        if not res[i]:
            robust = ds[i] + 1
            break
    return first, robust


# --- Spektrogramm zum Vergleich (Hann, wie in der Demo zum Spektrogramm) ---------------------------------------------------------------------------


def hann(length: int) -> np.ndarray:
    """Periodisches Hann-Fenster."""
    return 0.5 - 0.5 * np.cos(TWO_PI * np.arange(length) / length)


def stft_two_tone_share(f1: float, ratio: float, length: int, rng, runs: int) -> float:
    """Anteil der Läufe, in denen zwei gleich starke Töne im Spektrum eines Hann-Rahmens der Länge L (Nullen bis 8 L) getrennt sind."""
    t = np.arange(length)
    f2 = f1 * ratio
    w = hann(length)
    nfft = 8 * length
    i1, i2 = int(round(f1 * nfft)), int(round(f2 * nfft))
    ok = 0
    for _ in range(runs):
        ph = TWO_PI * rng.uniform(2)
        x = np.cos(TWO_PI * f1 * t + ph[0]) + np.cos(TWO_PI * f2 * t + ph[1])
        ok += resolved(np.abs(np.fft.rfft(x * w, nfft)) ** 2, i1, i2)
    return ok / runs


def stft_min_ratio(f1: float, length: int, make_rng, runs: int = 60):
    """Kleinstes Frequenzverhältnis (Raster 0,05 ab 1,05, bis 6), bei dem das Hann-Spektrum zwei Töne in allen Läufen trennt; None, wenn f1 * Verhältnis 0,5 erreicht."""
    for r in np.arange(1.05, 6.0, 0.05):
        if f1 * r >= 0.5:
            return None
        if stft_two_tone_share(f1, float(r), length, make_rng(), runs) >= 0.995:
            return round(float(r), 2)
    return None


def stft_min_impulse_distance(length: int) -> int | None:
    """Kleinster Abstand zweier Impulse, bei dem die Kurzzeitenergie (Hann, Schrittweite 1) beide trennt."""
    n = 4 * length
    w2 = hann(length) ** 2
    for dt in range(1, 2 * length):
        x = np.zeros(n)
        t1 = n // 2
        x[t1] = 1.0
        x[t1 + dt] = 1.0
        energy = np.sum(np.lib.stride_tricks.sliding_window_view(x, length) ** 2 * w2, axis=-1)
        if resolved(energy, t1 - length // 2, t1 + dt - length // 2):
            return dt
    return None


def frequency_correlation(w0: float, rel: float) -> float:
    """Erwartete Korrelation der Zellleistungen weißen Rauschens bei den Frequenzen f und f (1 + rel): 2 / (q + 1/q) mal exp(-w0^2 rel^2 / (q^2 + 1)) mit q = 1 + rel."""
    q = 1.0 + rel
    return 2.0 / (q + 1.0 / q) * math.exp(-(w0 * rel) ** 2 / (q * q + 1.0))


def correlation(a, b) -> float:
    """Korrelationskoeffizient zweier gleich langer Felder (flach)."""
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
