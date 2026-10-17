"""Kern der Wavelet-Transformation gegen Handrechnung (Mini-Instanz), die Summendefinition und geschlossene Formeln."""

import math

import numpy as np
import pytest

import cwt_morlet as M
import signal_generator as G

W0 = 6.0


def test_mini_instance_by_hand(mini_x):
    f = W0 / (2 * math.pi * 2.0)                                  # s = 2
    w = np.abs(M.cwt_direct(mini_x, f, W0))
    hand = M.PI14 / math.sqrt(2) * np.exp(-((np.arange(9) - 4) ** 2) / 8.0)
    assert w == pytest.approx(hand, abs=1e-12)
    assert w[4] == pytest.approx(0.531126, abs=1e-6) and w[3] == pytest.approx(0.468717, abs=1e-6) and w[0] == pytest.approx(0.07188, abs=1e-5)


def test_morlet_by_hand():
    assert abs(M.morlet(0.0, 6.0)) == pytest.approx(math.pi ** -0.25)
    assert abs(M.morlet(1.0, 6.0)) == pytest.approx(math.pi ** -0.25 * math.exp(-0.5))
    assert M.morlet(0.5, 6.0) == pytest.approx(math.pi ** -0.25 * complex(math.cos(3.0), math.sin(3.0)) * math.exp(-0.125))
    assert M.morlet(np.array([0.0, 1.0]), 3.0).shape == (2,)


def test_scale_and_frequency_are_inverse():
    assert float(M.scale_of(0.1, 6.0)) == pytest.approx(6.0 / (2 * math.pi * 0.1)) and float(M.frequency_of(2.0, 6.0)) == pytest.approx(6.0 / (4 * math.pi))
    f = np.array([0.02, 0.1, 0.4])
    assert M.frequency_of(M.scale_of(f, 12.0), 12.0) == pytest.approx(f)


def test_f_max_by_hand_and_cap():
    assert M.f_max(6.0) == pytest.approx(6.0 / (2 * 9.7)) and M.f_max(3.0) == pytest.approx(3.0 / 13.4) and M.f_max(12.0) == pytest.approx(12.0 / 31.4)
    assert M.f_max(64.0) == 0.45 and M.f_max(1000.0) == 0.45 and M.NYQUIST_MARGIN == 3.7


def test_freq_grid_is_ascending_logarithmic_and_ends_at_fmax():
    g = M.freq_grid(0.05, 0.4, 8)
    assert g[-1] == pytest.approx(0.4) and g[0] <= 0.05 < g[1] * 1.0001 or g[0] <= 0.05
    assert (np.diff(g) > 0).all() and g[1:] / g[:-1] == pytest.approx(2 ** (1 / 8))
    assert len(g) == math.ceil(8 * math.log2(0.4 / 0.05)) + 1 == 25
    assert len(M.freq_grid(0.1, 0.2, 1)) == 2 and M.freq_grid(0.1, 0.2, 1).tolist() == pytest.approx([0.1, 0.2])
    for bad in ((0.2, 0.1, 4), (0.0, 0.1, 4), (0.1, 0.2, 0)):
        with pytest.raises(ValueError):
            M.freq_grid(*bad)


@pytest.mark.parametrize("w0,f", [(6.0, 0.05), (6.0, 0.2), (3.0, 0.1), (12.0, 0.1), (12.0, 0.3), (24.0, 0.4)])
def test_fft_transform_equals_the_direct_sum_inside_the_valid_range(w0, f):
    assert f <= M.f_max(w0) + 0.01
    y = np.random.default_rng(1).normal(size=300)
    d = M.cwt_direct(y, f, w0)
    assert np.abs(M.cwt(y, [f], w0)[0] - d).max() / np.abs(d).max() < 1e-2
    if w0 >= 6.0:
        assert np.abs(M.cwt(y, [f], w0)[0] - d).max() / np.abs(d).max() < 1e-4


def test_fft_transform_deviates_from_the_sum_beyond_the_valid_range():
    y = np.random.default_rng(1).normal(size=300)
    d = M.cwt_direct(y, 0.4, 6.0)
    assert np.abs(M.cwt(y, [0.4], 6.0)[0] - d).max() / np.abs(d).max() > 0.03 and M.f_max(6.0) < 0.4
    d3 = M.cwt_direct(y, 0.3, 3.0)
    assert np.abs(M.cwt(y, [0.3], 3.0)[0] - d3).max() / np.abs(d3).max() > 0.01 and M.f_max(3.0) < 0.3


def test_cwt_shape_and_rows_are_independent():
    x = np.random.default_rng(2).normal(size=(3, 200))
    fs = np.array([0.05, 0.1, 0.2])
    W = M.cwt(x, fs, 6.0)
    assert W.shape == (3, 3, 200)
    assert W[1] == pytest.approx(M.cwt(x[1], fs, 6.0)) and W[2, 1] == pytest.approx(M.cwt(x[2], [0.1], 6.0)[0])


def test_tone_response_inside_equals_the_closed_form():
    n, f0, phase = 2048, 0.1, 0.7
    x = np.cos(2 * math.pi * f0 * np.arange(n) + phase)
    fs = np.array([0.08, 0.1, 0.12])
    W = np.abs(M.cwt(x, fs, 6.0)[:, n // 2])
    s = M.scale_of(fs, 6.0)
    th = 0.5 * math.sqrt(2 * math.pi) * np.sqrt(s) * M.PI14 * np.exp(-0.5 * (s * 2 * math.pi * f0 - 6.0) ** 2)
    assert W == pytest.approx(th, rel=1e-6) and W[1] == pytest.approx(2.909097, abs=1e-6)


def test_power_rectification_and_the_s_scaling_of_equal_tones():
    n = 2048
    peaks, rect = [], []
    for f0 in (0.2, 0.1, 0.05):
        x = np.cos(2 * math.pi * f0 * np.arange(n) + 0.4)
        fg = M.freq_grid(f0 * 0.8, f0 * 1.25, 64)
        W = M.cwt(x, fg, 6.0)[:, n // 2 - 40:n // 2 + 40]
        peaks.append(M.power(W, fg, 6.0, rectified=False).max())
        rect.append(M.power(W, fg, 6.0, rectified=True).max())
    assert peaks[1] / peaks[0] == pytest.approx(2.0, rel=1e-3) and peaks[2] / peaks[0] == pytest.approx(4.0, rel=1e-3)
    assert rect[1] == pytest.approx(rect[0], rel=1e-3) and rect[2] == pytest.approx(rect[0], rel=1e-3) and rect[0] == pytest.approx(0.8856, abs=1e-3)
    fs = np.array([0.1, 0.2])
    Wm = np.ones((2, 3), dtype=complex)
    assert M.power(Wm, fs, 6.0, False).tolist() == [[1.0] * 3] * 2
    assert M.power(Wm, fs, 6.0, True)[:, 0] == pytest.approx(1.0 / M.scale_of(fs, 6.0))


def test_white_noise_power_is_flat_and_exponential():
    x = G.SplitMix64(31).normal(200 * 512).reshape(200, 512)
    for f in (0.3, 0.1, 0.03):
        s = float(M.scale_of(f, 6.0))
        core = np.abs(M.cwt(x, [f], 6.0)[:, 0, int(4 * s):512 - int(4 * s)]) ** 2
        assert core.mean() == pytest.approx(1.0, abs=0.04) and core.std() / core.mean() == pytest.approx(1.0, abs=0.04)


def test_refine_log_index_is_exact_for_a_log_parabola_and_safe_at_edges():
    p = np.exp(-((np.arange(10) - 3.3) ** 2))
    assert M.refine_log_index(p, 3) == pytest.approx(3.3)
    assert M.refine_log_index(p, 0) == 0.0 and M.refine_log_index(p, 9) == 9.0 and M.refine_log_index(np.ones(5), 2) == 2.0
    assert M.refine_log_index(np.exp(-((np.arange(6) - 1.3) ** 2)), 1) == pytest.approx(1.3)


def test_ridge_vectorised_equals_the_scalar_rule_and_follows_a_tone():
    fg = M.freq_grid(0.05, 0.4, 32)
    x = np.cos(2 * math.pi * 0.1337 * np.arange(1024))
    P = M.power(M.cwt(x, fg, 12.0), fg, 12.0)
    r = M.ridge_frequencies(P, fg)
    ratio = fg[1] / fg[0]
    ref = np.array([fg[int(np.argmax(P[:, k]))] * ratio ** (M.refine_log_index(P[:, k], int(np.argmax(P[:, k]))) - int(np.argmax(P[:, k]))) for k in range(1024)])
    assert r == pytest.approx(ref, abs=1e-12) and r[512] == pytest.approx(0.1337, abs=2e-4)
    both = M.ridge_frequencies(np.stack([P, P]), fg)
    assert both.shape == (2, 1024) and both[1] == pytest.approx(r)
    edge = np.zeros((4, 3))
    edge[3, :] = 1.0
    assert M.ridge_frequencies(edge, np.array([1.0, 2.0, 4.0, 8.0])).tolist() == [8.0, 8.0, 8.0]      # Gipfel am Rand: kein Verfeinern
    edge[:] = 0.0
    edge[0, :] = 1.0
    assert M.ridge_frequencies(edge, np.array([1.0, 2.0, 4.0, 8.0])).tolist() == [1.0, 1.0, 1.0]


def test_local_maxima_and_peak_frequencies():
    assert M.local_maxima([0, 2, 1, 3, 3, 0]) == [1, 3] and M.local_maxima([0, 1, 3, 0]) == [2] and M.local_maxima([0, 3, 1, 0]) == [1] and M.local_maxima([5, 1, 5]) == []
    fg = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    prof = np.exp(-((np.arange(6) - 1.0) ** 2) * 4) * 5 + np.exp(-((np.arange(6) - 4.0) ** 2) * 4) * 3
    p = M.peak_frequencies(prof, fg, 2)
    assert p == pytest.approx([2.0, 16.0], rel=1e-3) and p == sorted(p)
    assert len(M.peak_frequencies(prof, fg, 5)) == 2 and M.peak_frequencies(np.ones(6), fg, 2) == []


def test_cone_ratio_and_distances_by_hand():
    assert M.cone_ratio(0.0) == pytest.approx(0.5) and M.cone_ratio(1.0) == pytest.approx(0.841345, abs=1e-6) and M.cone_ratio(2.0) == pytest.approx(0.977250, abs=1e-6)
    assert M.cone_ratio(np.array([0.0, math.sqrt(2)])).tolist() == pytest.approx([0.5, 0.921350], abs=1e-6)
    assert float(M.cone_distance(1.0, 0.99)) == pytest.approx(2.326348, abs=1e-5) and float(M.cone_distance(1.0, 0.9)) == pytest.approx(1.281552, abs=1e-5)
    assert float(M.cone_distance(10.0, 0.99)) == pytest.approx(23.26348, abs=1e-4) and float(M.cone_distance(1.0)) == pytest.approx(2.326348, abs=1e-5)
    assert float(M.coi_edge(10.0)) == pytest.approx(10 * math.sqrt(2)) and float(M.coi_edge(10.0, 3.0)) == 30.0


@pytest.mark.parametrize("w0,f", [(6.0, 0.1), (12.0, 0.1), (3.0, 0.1)])
def test_measured_edge_ratio_follows_phi(w0, f):
    n = 1024
    w = np.abs(M.cwt(np.cos(2 * math.pi * f * np.arange(n) + 0.3), [f], w0)[0])
    s = float(M.scale_of(f, w0))
    ratio = w / w[n // 2 - 50:n // 2 + 50].mean()
    for k in (0.0, 1.0, math.sqrt(2), 2.0):
        assert ratio[int(round(k * s))] == pytest.approx(float(M.cone_ratio(k)), abs=0.06)
    assert ratio[int(round(3 * s))] > 0.99


def test_freq_grid_rejects_equal_bounds():
    with pytest.raises(ValueError):
        M.freq_grid(0.1, 0.1, 4)


def test_peak_frequencies_with_count_one_returns_the_highest_peak():
    fg = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    prof = np.exp(-((np.arange(6) - 1.0) ** 2) * 4) * 5 + np.exp(-((np.arange(6) - 4.0) ** 2) * 4) * 3
    assert M.peak_frequencies(prof, fg, 1) == pytest.approx([2.0], rel=1e-3)
    assert M.peak_frequencies(prof[::-1].copy(), fg, 1) == pytest.approx([16.0], rel=1e-3)


def test_refine_log_index_refines_the_last_inner_point_too():
    p = np.exp(-((np.arange(6) - 4.3) ** 2))
    assert M.refine_log_index(p, 4) == pytest.approx(4.3) and M.refine_log_index(p, 5) == 5.0


def test_ridge_refines_a_peak_at_the_second_and_the_second_to_last_grid_point():
    fg = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    for centre in (1.3, 3.3):
        col = np.exp(-((np.arange(5) - centre) ** 2))
        r = M.ridge_frequencies(np.stack([col, col], axis=1), fg)
        assert r == pytest.approx([2.0 ** centre] * 2, rel=1e-9)
