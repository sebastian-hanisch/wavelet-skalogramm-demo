"""Auflösung: Töne und Impulse beim Wavelet, Vergleich mit dem Spektrogramm, Korrelation. Handrechnung und feste Werte."""

import math

import numpy as np
import pytest

import cwt_morlet as M
import cwt_resolution as R
import signal_generator as G


def test_resolved_by_hand():
    assert R.resolved([0, 5, 1, 5, 0], 1, 3) and not R.resolved([0, 5, 4, 5, 0], 1, 3) and not R.resolved([0, 5, 0, 0, 0], 1, 2)
    assert R.resolved([0, 5, 4, 5, 0], 1, 3, depth=0.9) and R.resolved([0, 5, 1, 5, 0], 3, 1) and R.resolved([5, 1, 5], 0, 2)
    assert R.resolved([0, 5, 2.5 - 1e-9, 5, 0], 1, 3) and not R.resolved([0, 5, 2.5, 5, 0], 1, 3)


def test_tone_response_equals_the_fft_transform_inside():
    n, f0, phase = 2048, 0.1, 0.7
    x = np.cos(2 * math.pi * f0 * np.arange(n) + phase)
    fs = np.array([0.08, 0.1, 0.12])
    W = M.cwt(x, fs, 6.0)[:, n // 2]
    cf = R.tone_response(fs, f0, phase + 2 * math.pi * f0 * (n // 2), 6.0)
    assert W == pytest.approx(cf, abs=1e-6) and abs(cf[1]) == pytest.approx(2.909097, abs=1e-6)


def test_two_tone_profile_equals_the_fft_profile_for_both_measures():
    n = 2048
    t = np.arange(n)
    f1, f2, ph1, ph2 = 0.1, 0.15, 0.4, 1.3
    x = np.cos(2 * math.pi * f1 * t + ph1) + np.cos(2 * math.pi * f2 * t + ph2)
    fg = M.freq_grid(0.06, 0.22, 24)
    W = M.cwt(x, fg, 6.0)[:, n // 2]
    for rect in (True, False):
        mine = R.two_tone_profile(fg, f1, f2, ph1 + 2 * math.pi * f1 * (n // 2), ph2 + 2 * math.pi * f2 * (n // 2), 6.0, rect)
        ref = np.abs(W) ** 2 / (M.scale_of(fg, 6.0) if rect else 1.0)
        assert mine == pytest.approx(ref, rel=1e-6)


def test_two_tone_share_is_deterministic_and_extremes_are_exact():
    a = R.two_tone_resolved_share(0.1, 1.2, 6.0, G.SplitMix64(500), 30)
    assert a == R.two_tone_resolved_share(0.1, 1.2, 6.0, G.SplitMix64(500), 30) and 0.0 <= a < 0.5
    assert R.two_tone_resolved_share(0.1, 2.0, 6.0, G.SplitMix64(500), 30) == 1.0 and R.two_tone_resolved_share(0.1, 1.05, 6.0, G.SplitMix64(500), 30) == 0.0


def test_two_tone_resolution_equals_the_fft_transform_run_by_run():
    rng = G.SplitMix64(11)
    n, f1, ratio = 1024, 0.05, 1.5
    f2 = f1 * ratio
    fg = M.freq_grid(f1 * 0.6, f2 * 1.5, 96)
    i1, i2 = int(np.argmin(np.abs(fg - f1))), int(np.argmin(np.abs(fg - f2)))
    t = np.arange(n)
    for _ in range(8):
        ph = 2 * math.pi * rng.uniform(2)
        x = np.cos(2 * math.pi * f1 * t + ph[0]) + np.cos(2 * math.pi * f2 * t + ph[1])
        fft_profile = np.abs(M.cwt(x, fg, 6.0)[:, n // 2]) ** 2 / M.scale_of(fg, 6.0)
        closed = R.two_tone_profile(fg, f1, f2, ph[0] + 2 * math.pi * f1 * (n // 2), ph[1] + 2 * math.pi * f2 * (n // 2), 6.0)
        assert closed == pytest.approx(fft_profile, rel=1e-6)
        assert R.resolved(fft_profile, i1, i2) == R.resolved(closed, i1, i2)


def test_min_ratio_golden_and_constant_over_frequency():
    make = lambda: G.SplitMix64(500)
    assert [R.min_ratio(f, 6.0, make, 30) for f in (0.025, 0.05, 0.1, 0.2)] == [1.65] * 4
    assert R.min_ratio(0.1, 12.0, make, 30) == 1.3 and R.min_ratio(0.1, 3.0, make, 30) == 2.6
    assert R.min_ratio(0.4, 6.0, make, 30) is None                      # 0,4 mal Verhältnis überschreitet 0,45 vor der Trennung
    assert R.min_ratio(0.1, 6.0, make, 30, ratios=[1.1, 1.2]) is None


def test_impulses_resolved_and_distances_golden():
    assert not R.impulses_resolved(0.1, 12, 6.0) and R.impulses_resolved(0.1, 23, 6.0) and not R.impulses_resolved(0.1, 22, 6.0)
    assert R.min_distance(0.1, 6.0) == (5, 23) and R.min_distance(0.2, 6.0) == (7, 12) and R.min_distance(0.1, 12.0)[1] == 52
    first, robust = R.min_distance(0.05, 6.0)
    assert (first, robust) == (10, 45) and robust / float(M.scale_of(0.05, 6.0)) == pytest.approx(2.36, abs=0.01)


def test_stft_comparison_values():
    make = lambda: G.SplitMix64(500)
    assert [R.stft_min_ratio(f, 64, make, 30) for f in (0.025, 0.05, 0.1, 0.2)] == [2.55, 1.8, 1.4, 1.2]
    assert R.stft_min_impulse_distance(64) == 33 and R.stft_min_impulse_distance(16) == 9
    assert R.stft_min_ratio(0.45, 64, make, 30) is None
    assert R.hann(4).tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5])
    assert R.stft_two_tone_share(0.2, 2.0, 64, G.SplitMix64(1), 10) == 1.0 and R.stft_two_tone_share(0.2, 1.02, 64, G.SplitMix64(1), 10) == 0.0


def test_correlation_by_hand_and_the_frequency_formula():
    a = np.array([1.0, 2.0, 3.0])
    assert R.correlation(a, 2 * a + 1) == pytest.approx(1.0) and R.correlation(a, -a) == pytest.approx(-1.0) and R.correlation(a, np.array([1.0, 3.0, 2.0])) == pytest.approx(0.5)
    assert R.correlation(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 5.0]])) == pytest.approx(np.corrcoef([1, 2, 3, 4], [1, 2, 3, 5])[0, 1])
    assert R.frequency_correlation(6.0, 0.0) == 1.0 and R.frequency_correlation(6.0, 0.1) == pytest.approx(0.8458, abs=1e-3) and R.frequency_correlation(6.0, 0.3) == pytest.approx(0.2898, abs=1e-3)


def test_frequency_correlation_formula_equals_the_simulation():
    x = G.SplitMix64(7).normal(300 * 512).reshape(300, 512)
    p1 = np.abs(M.cwt(x, [0.1], 6.0)[:, 0, 100:412]) ** 2
    for rel in (0.1, 0.3):
        p2 = np.abs(M.cwt(x, [0.1 * (1 + rel)], 6.0)[:, 0, 100:412]) ** 2
        assert R.correlation(p1, p2) == pytest.approx(R.frequency_correlation(6.0, rel), abs=0.04)


def test_stft_impulse_distance_is_half_the_window_plus_one_for_other_lengths():
    assert R.stft_min_impulse_distance(32) == 17 and R.stft_min_impulse_distance(128) == 65
