"""Auswertung: Einstellungen, Messung, Analyse je Signaltyp und die Experimentfunktionen (kleine Instanzen von Hand, feste Werte)."""

import math
import pickle

import numpy as np
import pytest

import cwt_constants as C
import cwt_evaluation as ev
import cwt_morlet as M
import cwt_resolution as R
import cwt_signals as S
import signal_generator as G


def test_settings_defaults_and_flags():
    s = ev.Settings()
    assert s.has_noise is False and s.rectified is True and ev.Settings(noise="an").has_noise is True and ev.Settings(power="L2").rectified is False
    assert (s.n, s.signal, s.w0, s.voices, s.fmin, s.tcut, s.fcut) == (1024, "Chirp", 6, 16, 0.02, 0.5, 0.1)


def test_noise_sigma_by_hand_and_the_german_decimal_comma():
    assert ev.noise_sigma(0) == pytest.approx(math.sqrt(0.5)) and ev.noise_sigma(10) == pytest.approx(math.sqrt(0.05)) and ev.noise_sigma(20) == pytest.approx(math.sqrt(0.005))
    assert ev.de(0.12345, 3) == "0,123" and ev.de(2.0) == "2,0" and ev.de(-0.5, 2) == "-0,50"


def test_make_signal_for_each_type():
    n = 512
    assert ev.make_signal(ev.Settings(n=n, signal="Chirp", p1=0.1, p2=0.3)).x == pytest.approx(S.linear_chirp(n, 0.1, 0.3, 0.3).x)
    assert ev.make_signal(ev.Settings(n=n, signal="Zwei Töne", p1=0.1, p2=1.5)).x == pytest.approx(S.two_tones(n, 0.1, 1.5, 0.3, 1.1).x)
    st = ev.make_signal(ev.Settings(n=n, signal="Frequenzsprung", p1=0.2, p2=0.3, p3=0.5))
    assert st.events == (256.0,) and st.x == pytest.approx(S.frequency_step(n, 0.2, 0.3, 256, 0.3).x)
    im = ev.make_signal(ev.Settings(n=n, signal="Zwei Impulse", p1=64.0, p2=0.5))
    assert im.events == (256.0, 320.0) and np.flatnonzero(im.x).tolist() == [256, 320]
    assert ev.make_signal(ev.Settings(n=n, signal="Tiefer Ton plus Burst", p1=0.04, p2=0.25, p3=0.5)).x == pytest.approx(S.tone_plus_burst(n, 0.04, 0.25, 256, 0.3).x)
    with pytest.raises(ValueError):
        ev.make_signal(ev.Settings(signal="Rechteck"))


def test_second_impulse_is_moved_inside_the_record():
    im = ev.make_signal(ev.Settings(n=512, signal="Zwei Impulse", p1=400.0, p2=0.8))
    assert im.events == (410.0, 511.0) and len(im.x) == 512 and im.x.sum() == 2.0
    assert ev.make_signal(ev.Settings(n=256, signal="Zwei Impulse", p1=2.0, p2=0.1)).events == (26.0, 28.0)


def test_measurement_adds_the_scaled_splitmix_noise_and_nothing_without_noise():
    s = ev.Settings(noise="an", snr=10, seed=5)
    sig, x = ev.build_measurement(s)
    assert x - sig.x == pytest.approx(ev.noise_sigma(10) * G.SplitMix64(5).normal(1024))
    assert (x - sig.x).std() == pytest.approx(ev.noise_sigma(10), rel=0.1)
    assert ev.build_measurement(ev.Settings(seed=5))[1].tolist() == sig.x.tolist()
    assert ev.build_measurement(ev.Settings(noise="an", snr=10, seed=6))[1].tolist() != x.tolist()


def test_frequency_grid_follows_w0_and_fmin():
    g = ev.frequency_grid(ev.Settings(w0=6, fmin=0.02, voices=16))
    assert g[-1] == pytest.approx(M.f_max(6.0)) and g[0] <= 0.02 and g[0] > 0.02 / 2 ** (1 / 16) * 0.999 and len(g) == 65
    low = ev.frequency_grid(ev.Settings(w0=3, fmin=0.05))
    assert low[-1] == pytest.approx(M.f_max(3.0)) and 0.04 < low[0] <= 0.05
    high_min = ev.frequency_grid(ev.Settings(w0=3, fmin=0.3))
    assert high_min[0] <= M.f_max(3.0) / 2 and high_min[-1] == pytest.approx(M.f_max(3.0))      # kleinste Frequenz höchstens halbe Obergrenze


def test_analyse_chirp_golden_and_shapes():
    a = ev.analyse(ev.Settings())
    assert a.power.shape == (65, 1024) and a.cells == 66560 and len(a.freqs) == 65 and a.t_index == 512 and a.freqs[a.f_index] == pytest.approx(0.1003, abs=1e-4)
    assert a.rmse == pytest.approx(0.00015947, abs=1e-7) and a.interior_share == pytest.approx(0.962890625) and a.truth.shape == (1, 1024) and a.ridge.shape == (1024,)
    assert a.ratio_limit == pytest.approx(1.65) and a.time_limit == pytest.approx(22.918, abs=1e-3) and a.fmax == pytest.approx(M.f_max(6.0))
    assert a.resolved is None and a.both_found is None and a.step_error is None and a.burst_peaks is None
    b = ev.analyse(ev.Settings(w0=24, fmin=0.05))
    assert b.rmse == pytest.approx(0.00014468, abs=1e-7) and b.interior_share == pytest.approx(0.8662109375) and b.ratio_limit == pytest.approx(1 + 3.9 / 24)


def test_analyse_power_measure_changes_the_ridge_bias():
    assert ev.analyse(ev.Settings(power="L2")).rmse == pytest.approx(0.0026839, abs=1e-6) and ev.analyse(ev.Settings(power="L2")).rmse > 10 * ev.analyse(ev.Settings()).rmse


def test_analyse_chirp_without_interior_points():
    a = ev.analyse(ev.Settings(n=256, w0=64))
    assert a.rmse is None and a.interior_share == 0.0 and a.ridge is not None


def test_interior_mask_by_hand():
    m = ev.interior_mask(np.full(10, 0.1), 10, 6.0)          # s = 9,55, 3 s = 28,6 > 10: kein Zeitpunkt
    assert not m.any()
    big = ev.interior_mask(np.full(200, 0.4), 200, 6.0)      # s = 2,39, 3 s = 7,17: Zeitpunkte 8 .. 191
    assert big.tolist().index(True) == 8 and np.flatnonzero(big)[-1] == 191 and big.sum() == 184


def test_analyse_two_tones_resolved_unresolved_and_outside_the_grid():
    assert ev.analyse(ev.Settings(signal="Zwei Töne", p1=0.1, p2=1.5)).resolved is True and ev.analyse(ev.Settings(signal="Zwei Töne", p1=0.1, p2=2.0)).resolved is True
    assert ev.analyse(ev.Settings(signal="Zwei Töne", p1=0.1, p2=1.2)).resolved is False
    assert ev.analyse(ev.Settings(signal="Zwei Töne", p1=0.2, p2=2.0, w0=3)).resolved is None        # 0,4 liegt über f_max(3) = 0,224
    a = ev.analyse(ev.Settings(signal="Zwei Töne", p1=0.1, p2=1.5))
    assert a.rmse is None and a.ridge is None and a.truth.shape == (2, 1024) and a.step_error is None


def test_nearest_grid_index_and_the_tolerance_of_one_step():
    g = np.array([1.0, 2.0, 4.0])
    assert ev._nearest(g, 2.1) == 1 and ev._nearest(g, 0.6) == 0 and ev._nearest(g, 0.4) is None and ev._nearest(g, 8.0) == 2 and ev._nearest(g, 9.0) is None


def test_analyse_frequency_step_and_step_error_by_hand():
    up = ev.analyse(ev.Settings(signal="Frequenzsprung", p1=0.1, p2=0.2, p3=0.5))
    down = ev.analyse(ev.Settings(signal="Frequenzsprung", p1=0.2, p2=0.1, p3=0.5))
    assert up.step_error == -2.0 and down.step_error == 0.0 and up.rmse is None and up.ridge.shape == (1024,)
    s = ev.Settings(n=100, signal="Frequenzsprung", p1=0.2, p2=0.3, p3=0.5)
    t = np.array([10.0, 30.0, 50.0, 70.0])
    assert ev._step_error(np.array([0.2, 0.2, 0.3, 0.3]), t, s) == 0.0 and ev._step_error(np.array([0.2, 0.2, 0.2, 0.3]), t, s) == 20.0 and ev._step_error(np.array([0.2, 0.3, 0.3, 0.3]), t, s) == -20.0
    assert ev._step_error(np.array([0.2, 0.2, 0.2, 0.2]), t, s) is None and ev._step_error(np.array([0.2, 0.25, 0.3, 0.3]), t, s) == 0.0          # genau die Mitte zählt noch nicht
    d = ev.Settings(n=100, signal="Frequenzsprung", p1=0.3, p2=0.2, p3=0.5)
    assert ev._step_error(np.array([0.3, 0.3, 0.2, 0.2]), t, d) == 0.0 and ev._step_error(np.array([0.3, 0.25, 0.2, 0.2]), t, d) == 0.0 and ev._step_error(np.array([0.3, 0.3, 0.3, 0.3]), t, d) is None
    e = ev.Settings(n=100, signal="Frequenzsprung", p1=0.2, p2=0.2, p3=0.5)
    assert ev._step_error(np.array([0.19, 0.21, 0.21, 0.21]), t, e) == -40.0


def test_analyse_two_impulses_depends_on_the_frequency_of_the_timeline():
    def run(fcut, **kw):
        return ev.analyse(ev.Settings(signal="Zwei Impulse", p1=20.0, p2=0.5, fcut=fcut, **kw)).resolved
    assert run(0.1) is False and run(0.3) is True and run(0.2) is True and run(0.02) is False
    assert run(0.45) is None and run(0.45, w0=3) is None
    a = ev.analyse(ev.Settings(signal="Zwei Impulse", p1=20.0, p2=0.5, fcut=0.45))
    assert a.f_index == len(a.freqs) - 1 and ev.analyse(ev.Settings(signal="Zwei Impulse", p1=20.0, p2=0.5, fcut=0.02)).f_index == 1 and a.rmse is None and a.both_found is None


def test_analyse_burst_found_only_with_enough_periods_and_inside_the_grid():
    def run(w0, f_burst=0.25):
        return ev.analyse(ev.Settings(signal="Tiefer Ton plus Burst", p1=0.04, p2=f_burst, p3=0.5, w0=w0))
    assert run(6).both_found is True and run(6).burst_peaks == pytest.approx((0.04003298, 0.25287007), abs=1e-6)
    assert run(3).both_found is False and len(run(3).burst_peaks) == 1 and run(12).both_found is True
    assert run(6, 0.3).both_found is False        # 0,3 liegt über f_max(6) = 0,309 minus Gitterschritt: kein innerer Gipfel
    a = run(6)
    assert a.rmse is None and a.resolved is None and a.step_error is None


def test_analysis_is_picklable():
    a = ev.analyse(ev.Settings(noise="an", snr=10, seed=C.DEFAULT_SEED))
    b = pickle.loads(pickle.dumps(a))
    assert b.rmse == a.rmse and b.settings == a.settings and (b.power == a.power).all()


def test_sweep_resolution_small_instance():
    r = ev.sweep_resolution(fs=(0.05, 0.1), w0=6, runs=20, length=64)
    assert [x["f"] for x in r["rows"]] == [0.05, 0.1] and [x["ratio"] for x in r["rows"]] == [1.65, 1.65] and [x["robust"] for x in r["rows"]] == [45, 23] and [x["first"] for x in r["rows"]] == [10, 5]
    assert [x["stft_ratio"] for x in r["rows"]] == [1.8, 1.4] and r["stft_distance"] == 33 and r["w0"] == 6 and r["length"] == 64 and r["stft_width"] == pytest.approx(2.5 / 64)
    assert r["rows"][0]["s"] == pytest.approx(float(M.scale_of(0.05, 6)))


def test_sweep_omega_small_instance():
    rows = ev.sweep_omega(w0s=(3, 12), f=0.1, runs=20)
    assert [(r["w0"], r["ratio"], r["robust"]) for r in rows] == [(3, 2.6, 13), (12, 1.3, 52)] and rows[1]["s"] == pytest.approx(float(M.scale_of(0.1, 12)))


def test_sweep_cone_small_instance():
    r = ev.sweep_cone(combos=((6, 0.1),), n=1024)
    row = r["rows"][0]
    assert row["w0"] == 6 and row["f"] == 0.1 and row["r0"] == pytest.approx(0.523, abs=1e-3) and row["r1"] == pytest.approx(0.8636, abs=1e-3) and row["d99"] == 22 and row["d90"] == 12
    assert r["theory"]["r1"] == pytest.approx(0.841345, abs=1e-6) and r["theory"]["d99"] == pytest.approx(2.326348, abs=1e-5) and r["theory"]["r0"] == 0.5


def test_sweep_chirp_small_instance_has_one_row_per_pair_and_the_right_error():
    rows = ev.sweep_chirp(w0s=(6, 24), snrs=(60, 0), runs=3, n=1024, voices=24)
    assert [(r["w0"], r["snr"], r["runs"]) for r in rows] == [(6, 60, 3), (6, 0, 3), (24, 60, 3), (24, 0, 3)]
    by = {(r["w0"], r["snr"]): r["rmse"] for r in rows}
    assert by[(6, 0)] > 10 * by[(24, 0)] and by[(6, 0)] > 100 * by[(6, 60)] and by[(24, 60)] < 5e-4


def test_sweep_scaling_small_instance():
    r = ev.sweep_scaling(fs=(0.2, 0.05))
    assert [t["f"] for t in r["tones"]] == [0.2, 0.05] and r["tones"][1]["rel_l2"] == pytest.approx(4.0, rel=1e-3) and r["tones"][1]["rel_rect"] == pytest.approx(1.0, rel=1e-3)
    assert all(abs(n["mean"] - 1.0) < 0.06 and abs(n["cv"] - 1.0) < 0.08 for n in r["noise"]) and r["w0"] == 6.0
    assert [c["lag"] for c in r["time_corr"]] == [5, 10, 19, 29] and all(abs(c["corr"] - c["theory"]) < 0.03 for c in r["time_corr"])
    assert all(abs(c["corr"] - c["theory"]) < 0.04 for c in r["freq_corr"]) and r["alarm"]["runs_share"] == 1.0 and 0.04 < r["alarm"]["cells_share"] < 0.06 and r["alarm"]["freqs"] == 28


def test_nearest_accepts_exactly_one_step_outside_the_grid():
    g = np.array([1.0, 2.0, 4.0])
    assert ev._nearest(g, 0.5) == 0 and ev._nearest(g, 0.4999) is None and ev._nearest(g, 8.0) == 2


def test_interior_mask_includes_the_points_exactly_three_s_from_both_edges():
    f = 0.09549296585513721                       # s = 10 exakt bei w0 = 6
    assert float(M.scale_of(f, 6.0)) == 10.0
    m = ev.interior_mask(np.full(100, f), 100, 6.0)
    assert m.sum() == 40 and np.flatnonzero(m)[0] == 30 and np.flatnonzero(m)[-1] == 69


def test_sweep_scaling_pinned_values_cover_both_noise_windows():
    r = ev.sweep_scaling(fs=(0.2, 0.05, 0.01))
    flat = lambda rows, keys: np.array([[row[k] for k in keys] for row in rows])
    assert flat(r["tones"], ("f", "peak_l2", "peak_rect")) == pytest.approx(np.array([[0.2, 4.260139, 0.885636], [0.05, 17.040555, 0.885636], [0.01, 85.202777, 0.885636]]), abs=1e-5)
    assert flat(r["noise"], ("f", "mean", "cv")) == pytest.approx(np.array([[0.2, 0.996539, 1.003206], [0.05, 0.999212, 0.998205], [0.01, 1.01832, 0.968425]]), abs=1e-5)
    assert flat(r["time_corr"], ("lag", "corr")) == pytest.approx(np.array([[5, 0.874332], [10, 0.584816], [19, 0.143656], [29, 0.010464]]), abs=1e-5)
    assert flat(r["freq_corr"], ("rel", "corr")) == pytest.approx(np.array([[0.05, 0.959372], [0.1, 0.851276], [0.2, 0.548442], [0.3, 0.289623]]), abs=1e-5)
    assert r["alarm"]["cells_share"] == pytest.approx(0.0487095, abs=1e-6)
