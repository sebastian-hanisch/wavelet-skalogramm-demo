"""Kern: kontinuierliche Wavelet-Transformation mit dem analytischen Morlet-Wavelet, Skala und Frequenz, Gültigkeitsgrenze, Kamm, Einflusskegel.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Abtastrate 1, Frequenzen in Zyklen je Abtastwert.
Wavelet psi(t) = pi^(-1/4) exp(i w0 t) exp(-t^2 / 2) (ohne Korrekturterm), L2-Normierung:
W(s, n) = sum_m x[m] conj(psi((m - n) / s)) / sqrt(s). Die Frequenz einer Skala ist f = w0 / (2 pi s).
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi
PI14 = math.pi ** -0.25
NYQUIST_MARGIN = 3.7            # Gaußkurve bei der Nyquist-Frequenz unter exp(-3,7^2 / 2) = 1e-3 der Spitze


def morlet(t, w0: float) -> np.ndarray:
    """Analytisches Morlet-Wavelet psi(t) = pi^(-1/4) exp(i w0 t) exp(-t^2/2)."""
    t = np.asarray(t, dtype=float)
    return PI14 * np.exp(1j * w0 * t) * np.exp(-0.5 * t * t)


def scale_of(f, w0: float):
    """Skala s = w0 / (2 pi f)."""
    return w0 / (TWO_PI * np.asarray(f, dtype=float))


def frequency_of(s, w0: float):
    """Frequenz f = w0 / (2 pi s)."""
    return w0 / (TWO_PI * np.asarray(s, dtype=float))


def f_max(w0: float) -> float:
    """Höchste Frequenz, bei der das abgetastete Wavelet noch unverfälscht ist: w0 / (2 (w0 + 3,7)), gedeckelt bei 0,45."""
    return min(0.45, w0 / (2.0 * (w0 + NYQUIST_MARGIN)))


def freq_grid(fmin: float, fmax: float, voices: int) -> np.ndarray:
    """Logarithmisches Frequenzgitter, aufsteigend: fmax * 2^(-k / voices) für k = 0 .. K (K so, dass fmin unterschritten oder erreicht wird)."""
    if not 0.0 < fmin < fmax or voices < 1:
        raise ValueError("Frequenzgitter braucht 0 < fmin < fmax und voices >= 1")
    k = int(math.ceil(voices * math.log2(fmax / fmin)))
    return (fmax * 2.0 ** (-np.arange(k + 1) / voices))[::-1]


def cwt(x, freqs, w0: float = 6.0) -> np.ndarray:
    """Transformation per FFT (nur positive Frequenzen, Nullen bis zur Zweierpotenz >= 2n angehängt): Form (..., Frequenzen, Zeit), komplex."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    size = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.fft(x, size, axis=-1)
    w = TWO_PI * np.fft.fftfreq(size)
    out = np.empty(x.shape[:-1] + (len(freqs), n), dtype=complex)
    for i, f in enumerate(freqs):
        s = float(scale_of(f, w0))
        psi = np.where(w > 0, PI14 * math.sqrt(TWO_PI * s) * np.exp(-0.5 * (s * w - w0) ** 2), 0.0)
        out[..., i, :] = np.fft.ifft(spec * psi, axis=-1)[..., :n]
    return out


def cwt_direct(x, f: float, w0: float = 6.0) -> np.ndarray:
    """Definition als Summe (Gegenprobe), eine Frequenz, eine Reihe."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = float(scale_of(f, w0))
    m = np.arange(n)
    out = np.empty(n, dtype=complex)
    for k in range(n):
        out[k] = np.sum(x * np.conj(morlet((m - k) / s, w0))) / math.sqrt(s)
    return out


def power(W, freqs, w0: float, rectified: bool = True) -> np.ndarray:
    """Leistung |W|^2 / s (gleichgewichtet: jede Linie der Amplitude 1 gibt dieselbe Spitze) oder |W|^2 (L2); Form wie W."""
    p = np.abs(W) ** 2
    if rectified:
        p = p / scale_of(freqs, w0)[:, None]
    return p


def refine_log_index(profile, i: int) -> float:
    """Verfeinerter Indexort des Gipfels bei i: Scheitel der Parabel durch den Logarithmus der drei Werte (am Rand: i)."""
    if i < 1 or i > len(profile) - 2:
        return float(i)
    a, b, c = (math.log(max(float(profile[j]), 1e-300)) for j in (i - 1, i, i + 1))
    den = a - 2.0 * b + c
    return i + (0.5 * (a - c) / den if den != 0 else 0.0)


def ridge_frequencies(p, freqs) -> np.ndarray:
    """Kamm: je Zeitpunkt die Frequenz des höchsten Gipfels (Parabelverfeinerung im Logarithmus der Frequenz); p hat die Form (..., Frequenzen, Zeit), freqs ist logarithmisch gleichmäßig.
    Am Rand des Frequenzgitters bleibt es beim Gitterpunkt."""
    p = np.asarray(p, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    count = p.shape[-2]
    ratio = freqs[1] / freqs[0]
    idx = np.argmax(p, axis=-2)
    inner = np.clip(idx, 1, count - 2)[..., None, :]
    a, b, c = (np.log(np.maximum(np.take_along_axis(p, inner + d, axis=-2)[..., 0, :], 1e-300)) for d in (-1, 0, 1))
    den = a - 2.0 * b + c
    off = np.where((idx >= 1) & (idx <= count - 2) & (den != 0), 0.5 * (a - c) / np.where(den != 0, den, 1.0), 0.0)
    return freqs[idx] * ratio ** off


def local_maxima(v) -> list[int]:
    """Indizes innerer Punkte, die höher als der linke und mindestens so hoch wie der rechte Nachbar sind."""
    v = np.asarray(v, dtype=float)
    return [i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]]


def peak_frequencies(profile, freqs, count: int) -> list[float]:
    """Frequenzen der `count` höchsten lokalen Maxima eines Skalenprofils (Parabelverfeinerung), aufsteigend; weniger, wenn es nicht genug Maxima gibt."""
    freqs = np.asarray(freqs, dtype=float)
    ratio = freqs[1] / freqs[0]
    idx = sorted(local_maxima(profile), key=lambda i: -profile[i])[:count]
    return sorted(float(freqs[i] * ratio ** (refine_log_index(profile, i) - i)) for i in idx)


def cone_ratio(d_over_s):
    """Erwartetes Verhältnis der Amplitude |W| im Abstand d vom Rand zur Amplitude im Inneren (Nullen angehängt): Phi(d / s), die Verteilungsfunktion der Normalverteilung."""
    z = np.asarray(d_over_s, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def cone_distance(s, level: float = 0.99):
    """Abstand zum Rand (in Abtastwerten), ab dem die Amplitude mindestens `level` des Innenwerts erreicht: s mal Phi^-1(level) (0,99: 2,326 s; 0,90: 1,282 s)."""
    level = float(level)
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < level:
            lo = mid
        else:
            hi = mid
    return np.asarray(s, dtype=float) * (0.5 * (lo + hi))


def coi_edge(s, e_fold: float = math.sqrt(2.0)):
    """Breite des Einflusskegels an jedem Rand (Abtastwerte): e-Falte sqrt(2) s nach Torrence und Compo."""
    return np.asarray(s, dtype=float) * e_fold
