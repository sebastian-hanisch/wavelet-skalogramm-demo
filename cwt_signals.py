"""Testsignale mit exakter Wahrheit: Chirp, zwei Töne, Frequenzsprung, zwei Impulse, tiefer Ton plus Burst.

Reine Funktionen, kein Zufall. Zeit t = 0 .. n - 1, Abtastrate 1. `tracks` ist die wahre Frequenz je Komponente als Funktion der Zeit (picklebare Objekte statt Lambdas),
`events` die Zeitpunkte von Impulsen, Sprüngen und der Burstmitte.
"""

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi
BURST_SIGMA_PERIODS = 1.5       # Standardabweichung der Gaußhülle des Bursts in Perioden


@dataclass(frozen=True)
class TestSignal:
    x: np.ndarray
    tracks: tuple
    events: tuple


@dataclass(frozen=True)
class LinearTrack:
    """Wahre Frequenz f0 + slope t."""
    f0: float
    slope: float

    def __call__(self, tt):
        return self.f0 + self.slope * np.asarray(tt, dtype=float)


@dataclass(frozen=True)
class StepTrack:
    """Wahre Frequenz f1 vor t0, f2 ab t0."""
    f1: float
    f2: float
    t0: int

    def __call__(self, tt):
        return np.where(np.asarray(tt, dtype=float) < self.t0, self.f1, self.f2)


@dataclass(frozen=True)
class ConstTrack:
    f: float

    def __call__(self, tt):
        return np.full(np.shape(tt), float(self.f))


def linear_chirp(n: int, f0: float, f1: float, phase: float = 0.0) -> TestSignal:
    """cos(2 pi (f0 t + (f1 - f0) t^2 / (2 (n - 1))) + phase): Frequenz f0 + (f1 - f0) t / (n - 1)."""
    t = np.arange(n)
    ph = TWO_PI * (f0 * t + (f1 - f0) * t * t / (2.0 * (n - 1)))
    return TestSignal(np.cos(ph + phase), (LinearTrack(f0, (f1 - f0) / (n - 1)),), ())


def two_tones(n: int, f1: float, ratio: float, phase1: float = 0.0, phase2: float = 0.0) -> TestSignal:
    """cos(2 pi f1 t + phase1) + cos(2 pi f1 ratio t + phase2), beide mit Amplitude 1."""
    t = np.arange(n)
    f2 = f1 * ratio
    return TestSignal(np.cos(TWO_PI * f1 * t + phase1) + np.cos(TWO_PI * f2 * t + phase2), (ConstTrack(f1), ConstTrack(f2)), ())


def frequency_step(n: int, f1: float, f2: float, t0: int, phase: float = 0.0) -> TestSignal:
    """cos(2 pi f1 t) für t < t0 und cos(2 pi f2 t + phase) ab t0."""
    t = np.arange(n)
    x = np.where(t < t0, np.cos(TWO_PI * f1 * t), np.cos(TWO_PI * f2 * t + phase))
    return TestSignal(x, (StepTrack(f1, f2, t0),), (float(t0),))


def two_impulses(n: int, t1: int, dt: int) -> TestSignal:
    """Zwei Einzelimpulse der Höhe 1 bei t1 und t1 + dt (keine Frequenzspur)."""
    x = np.zeros(n)
    x[t1] = 1.0
    x[t1 + dt] = 1.0
    return TestSignal(x, (), (float(t1), float(t1 + dt)))


def tone_plus_burst(n: int, f_low: float, f_high: float, t_burst: int, phase: float = 0.0) -> TestSignal:
    """Tiefer Ton cos(2 pi f_low t) plus ein kurzer Burst exp(-(t - t_burst)^2 / (2 sigma^2)) cos(2 pi f_high t + phase), sigma = 1,5 Perioden des Bursts. Spuren: tiefer Ton, hohe Frequenz."""
    t = np.arange(n)
    sigma = BURST_SIGMA_PERIODS / f_high
    burst = np.exp(-0.5 * ((t - t_burst) / sigma) ** 2) * np.cos(TWO_PI * f_high * t + phase)
    return TestSignal(np.cos(TWO_PI * f_low * t) + burst, (ConstTrack(f_low), ConstTrack(f_high)), (float(t_burst),))
