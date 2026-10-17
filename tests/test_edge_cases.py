"""Randfälle: alle Signaltypen, Längen, ω₀, Gitter, Parametergrenzen, Rauschen, Abbildungen und PDF."""

import numpy as np
import pytest

import cwt_constants as C
import cwt_evaluation as ev
import cwt_visualization as V
from cwt_pdf_export import build_pdf
from sweeps_cache import get


def settings(kind, params, **kw):
    p = list(params) + [0.0] * (3 - len(params))
    return ev.Settings(signal=kind, p1=p[0], p2=p[1], p3=p[2], **kw)


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_type_runs_for_the_shortest_and_the_longest_record_and_every_w0(kind):
    defaults = [spec[4] for spec in C.PARAMS[kind]]
    for n, w0s in ((min(C.N_CHOICES), (C.W0_CHOICES[0], C.W0_CHOICES[3], C.W0_CHOICES[-1])), (max(C.N_CHOICES), (C.W0_CHOICES[3],))):
        for w0 in w0s:
            a = ev.analyse(settings(kind, defaults, n=n, w0=w0, noise="an", snr=5))
            assert np.isfinite(a.power).all() and a.power.shape == (len(a.freqs), n) and 0 <= a.t_index < n and 0 <= a.f_index < len(a.freqs)


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_parameter_at_its_limits_runs_with_every_grid_and_power(kind):
    specs = C.PARAMS[kind]
    for pick in ("lo", "hi"):
        params = [spec[1] if pick == "lo" else spec[2] for spec in specs]
        for fmin in (C.FMIN_CHOICES[0], C.FMIN_CHOICES[-1]):
            for voices in (C.VOICES_CHOICES[0], C.VOICES_CHOICES[-1]):
                for power in C.POWER_MODES:
                    a = ev.analyse(settings(kind, params, n=256, w0=6, fmin=fmin, voices=voices, power=power, fcut=C.FCUT_MIN, tcut=C.CUT_MAX))
                    assert np.isfinite(a.power).all()
    verdict = ev.verdict_main(ev.analyse(settings(kind, [spec[2] for spec in specs], n=256, w0=64, fcut=C.FCUT_MAX)))
    assert verdict[0] in ("success", "warning", "info") and verdict[1]


def test_extreme_noise_levels_and_seed_limits_run():
    for snr in (C.SNR_MIN, C.SNR_MAX):
        for seed in (0, C.SEED_MAX):
            a = ev.analyse(settings("Chirp", (0.1, 0.3), noise="an", snr=snr, seed=seed, w0=24))
            assert np.isfinite(a.rmse)


def test_power_db_is_bounded_and_survives_an_all_zero_spectrum():
    p = np.array([[1.0, 0.1, 1e-9, 0.0]])
    db = V.power_db(p)
    assert db.max() == 0.0 and db.min() == C.DB_FLOOR and db[0, 1] == pytest.approx(-10.0)
    assert V.power_db(np.zeros((2, 3))).tolist() == [[C.DB_FLOOR] * 3] * 2


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_figures_and_pdf_build_for_every_type(kind):
    a = ev.analyse(settings(kind, [spec[4] for spec in C.PARAMS[kind]], noise="an", snr=10))
    for fig in (V.build_signal(a), V.build_scalogram(a), V.build_profile(a), V.build_timeline(a)):
        assert len(fig.data) >= 1
    pdf = build_pdf(a)
    assert pdf[:4] == b"%PDF" and len(pdf) > 800


def test_scalogram_layers_depend_on_the_signal():
    chirp = V.build_scalogram(ev.analyse(settings("Chirp", (0.1, 0.3))))
    imp = V.build_scalogram(ev.analyse(settings("Zwei Impulse", (20.0, 0.5))))
    assert len(chirp.data) == 1 + 2 + 1 + 1 and chirp.data[1].name == "Einflusskegel (√2 s)"
    assert len(imp.data) == 1 + 2 and len(V.build_scalogram(ev.analyse(settings("Zwei Töne", (0.1, 1.5)))).data) == 1 + 2 + 2


def test_figures_for_the_experiments():
    assert len(V.build_resolution_freq(get("resolution")).data) == 2 and len(V.build_resolution_time(get("resolution")).data) == 2 and len(V.build_omega(get("omega")).data) == 2
    assert len(V.build_cone(get("cone")).data) == 1 + 4 and len(V.build_chirp(get("chirp")).data) == 3 and len(V.build_scaling(get("scaling")).data) == 2 and len(V.build_correlation(get("scaling")).data) == 4


def test_pdf_for_every_kennzahl_branch():
    for kind, params, extra in (("Chirp", (0.1, 0.3), {}), ("Chirp", (0.1, 0.3), {"n": 256, "w0": 64}), ("Zwei Töne", (0.1, 1.5), {}), ("Frequenzsprung", (0.1, 0.2, 0.5), {}),
                                ("Zwei Impulse", (20.0, 0.5), {}), ("Tiefer Ton plus Burst", (0.04, 0.25, 0.5), {})):
        assert build_pdf(ev.analyse(settings(kind, params, **extra)))[:4] == b"%PDF"
