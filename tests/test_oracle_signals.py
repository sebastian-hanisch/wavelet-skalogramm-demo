"""Testsignale: Handrechnung, Ableitung der Phase, Spuren, Picklebarkeit (st.cache_data braucht das)."""

import math
import pickle

import numpy as np
import pytest

import cwt_signals as S

TWO_PI = 2 * math.pi


def test_linear_chirp_by_hand():
    s = S.linear_chirp(5, 0.0, 0.5)                      # Phase 2 pi t^2 / 16
    t = np.arange(5)
    assert s.x == pytest.approx(np.cos(TWO_PI * t ** 2 / 16))
    assert s.x[0] == 1.0 and s.x[2] == pytest.approx(0.0, abs=1e-12) and s.x[4] == pytest.approx(1.0)
    assert s.tracks[0](np.array([0, 4])).tolist() == [0.0, 0.5] and s.events == ()
    assert S.linear_chirp(5, 0.0, 0.5, 1.0).x[0] == pytest.approx(math.cos(1.0))


def test_chirp_track_is_the_derivative_of_the_phase_at_half_times():
    n, f0, f1 = 300, 0.1, 0.3
    s = S.linear_chirp(n, f0, f1, 0.7)
    t = np.arange(n)
    phase = TWO_PI * (f0 * t + (f1 - f0) * t * t / (2 * (n - 1))) + 0.7
    assert np.diff(phase) / TWO_PI == pytest.approx(s.tracks[0](t[:-1] + 0.5), abs=1e-9)
    assert s.x == pytest.approx(np.cos(phase))


def test_two_tones_by_hand():
    s = S.two_tones(8, 0.125, 2.0, 0.0, 0.0)
    t = np.arange(8)
    assert s.x == pytest.approx(np.cos(TWO_PI * 0.125 * t) + np.cos(TWO_PI * 0.25 * t))
    assert s.x[0] == 2.0 and s.tracks[0](np.arange(3)).tolist() == [0.125] * 3 and s.tracks[1](np.arange(3)).tolist() == [0.25] * 3 and s.events == ()
    assert S.two_tones(8, 0.125, 2.0, 1.0, 2.0).x[0] == pytest.approx(math.cos(1.0) + math.cos(2.0))


def test_frequency_step_by_hand():
    s = S.frequency_step(10, 0.25, 0.1, 4, 0.5)
    t = np.arange(10)
    assert s.x[:4] == pytest.approx(np.cos(TWO_PI * 0.25 * t[:4])) and s.x[4:] == pytest.approx(np.cos(TWO_PI * 0.1 * t[4:] + 0.5))
    assert s.tracks[0](t).tolist() == [0.25] * 4 + [0.1] * 6 and s.events == (4.0,)
    assert float(s.tracks[0](3.9)) == 0.25 and float(s.tracks[0](4.0)) == 0.1


def test_two_impulses_by_hand():
    s = S.two_impulses(20, 5, 7)
    assert np.flatnonzero(s.x).tolist() == [5, 12] and s.x.sum() == 2.0 and s.events == (5.0, 12.0) and s.tracks == ()


def test_tone_plus_burst_by_hand():
    s = S.tone_plus_burst(400, 0.05, 0.25, 200, 0.0)
    t = np.arange(400)
    sigma = 1.5 / 0.25
    assert S.BURST_SIGMA_PERIODS == 1.5 and sigma == 6.0
    expected = np.cos(TWO_PI * 0.05 * t) + np.exp(-0.5 * ((t - 200) / 6.0) ** 2) * np.cos(TWO_PI * 0.25 * t)
    assert s.x == pytest.approx(expected)
    assert s.x[200] == pytest.approx(2.0) and s.events == (200.0,) and s.tracks[0](np.arange(2)).tolist() == [0.05, 0.05] and s.tracks[1](np.arange(2)).tolist() == [0.25, 0.25]
    assert abs(s.x[0] - 1.0) < 1e-6 and S.tone_plus_burst(400, 0.05, 0.25, 200, 1.0).x[200] == pytest.approx(1.0 + math.cos(TWO_PI * 0.25 * 200 + 1.0))


def test_const_and_linear_and_step_tracks_accept_scalars_and_arrays():
    assert float(S.ConstTrack(0.2)(5.0)) == 0.2 and S.ConstTrack(0.2)(np.arange(3)).tolist() == [0.2] * 3
    assert S.LinearTrack(0.1, 0.01)(np.array([0, 10])).tolist() == pytest.approx([0.1, 0.2])
    assert S.StepTrack(1.0, 2.0, 3)(np.array([2, 3])).tolist() == [1.0, 2.0]


def test_every_signal_is_picklable():
    for s in (S.linear_chirp(64, 0.1, 0.3), S.two_tones(64, 0.1, 1.5), S.frequency_step(64, 0.2, 0.3, 30), S.two_impulses(64, 10, 5), S.tone_plus_burst(64, 0.04, 0.25, 30)):
        r = pickle.loads(pickle.dumps(s))
        assert r.x.tolist() == s.x.tolist() and r.events == s.events and len(r.tracks) == len(s.tracks)
        for a, b in zip(r.tracks, s.tracks):
            assert a(np.arange(10)).tolist() == b(np.arange(10)).tolist()


def test_phase_signs_are_checked_away_from_time_zero():
    t1 = S.two_tones(8, 0.125, 2.0, 1.0, 2.0).x[1]
    assert t1 == pytest.approx(math.cos(TWO_PI * 0.125 + 1.0) + math.cos(TWO_PI * 0.25 + 2.0))
    b = S.tone_plus_burst(400, 0.05, 0.25, 200, 1.0).x[201]
    assert b == pytest.approx(math.cos(TWO_PI * 0.05 * 201) + math.exp(-0.5 * (1 / 6.0) ** 2) * math.cos(TWO_PI * 0.25 * 201 + 1.0))
