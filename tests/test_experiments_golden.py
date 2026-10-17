"""Festwerte der Experimente (feste Zufallsströme): Das Ergebnis ist bei jedem Lauf dasselbe; Abweichungen bedeuten eine geänderte Rechnung."""

import pytest

from sweeps_cache import get

RES = {0.025: (1.65, 19, 90, 2.55), 0.05: (1.65, 10, 45, 1.8), 0.1: (1.65, 5, 23, 1.4), 0.2: (1.65, 7, 12, 1.2)}
RES_STFT_DISTANCE = 33
OMEGA = {3: (2.6, 5, 13), 4: (2.05, 5, 14), 5: (1.8, 5, 22), 6: (1.65, 5, 23), 8: (1.45, 5, 33), 10: (1.35, 5, 42), 12: (1.3, 5, 52), 16: (1.2, 5, 72)}
CONE = {(6, 0.1): (0.522984, 0.863556, 0.934783, 0.97896, 12, 22), (6, 0.2): (0.56957, 0.889012, 0.949463, 0.988313, 6, 11), (3, 0.1): (0.547503, 0.871128, 0.93823, 0.984524, 6, 11), (12, 0.1): (0.511241, 0.84638, 0.924947, 0.978004, 24, 44)}
CHIRP = {(6, 60): 2.5e-05, (6, 10): 0.002085, (6, 0): 0.049499, (8, 60): 3e-05, (8, 10): 0.00138, (8, 0): 0.036816, (12, 60): 4.6e-05, (12, 10): 0.000729, (12, 0): 0.015526, (16, 60): 6.7e-05, (16, 10): 0.000526, (16, 0): 0.00233, (24, 60): 0.000121, (24, 10): 0.000333, (24, 0): 0.001005, (32, 60): 0.000186, (32, 10): 0.000397, (32, 0): 0.001078, (48, 60): 0.000361, (48, 10): 0.000706, (48, 0): 0.002013, (64, 60): 0.000574, (64, 10): 0.001311, (64, 0): 0.003112}
TONES = {0.3: (2.840093, 0.885636), 0.2: (4.260139, 0.885636), 0.1: (8.520278, 0.885636), 0.05: (17.040555, 0.885636), 0.025: (34.081111, 0.885636)}
NOISE = {0.3: (1.008203, 1.003466), 0.2: (0.996539, 1.003206), 0.1: (0.998095, 1.018358), 0.05: (0.999212, 0.998205), 0.025: (1.017075, 0.986731)}
TIME_CORR = {0.5: (5, 0.874332, 0.871902), 1.0: (10, 0.584816, 0.577925), 2.0: (19, 0.143656, 0.138152), 3.0: (29, 0.010464, 0.009939)}
FREQ_CORR = {0.05: (0.959372, 0.956958), 0.1: (0.851276, 0.845835), 0.2: (0.548442, 0.545151), 0.3: (0.289623, 0.289821)}
ALARM = (28, 0.04871, 1.0, 200)


def test_resolution_golden():
    res = get("resolution")
    assert [r["f"] for r in res["rows"]] == list(RES) and res["stft_distance"] == RES_STFT_DISTANCE
    for r in res["rows"]:
        ratio, first, robust, stft = RES[r["f"]]
        assert r["ratio"] == pytest.approx(ratio, abs=1e-6) and r["first"] == first and r["robust"] == robust and r["stft_ratio"] == pytest.approx(stft, abs=1e-6)


def test_omega_golden():
    rows = get("omega")
    assert [r["w0"] for r in rows] == list(OMEGA)
    for r in rows:
        ratio, first, robust = OMEGA[r["w0"]]
        assert r["ratio"] == pytest.approx(ratio, abs=1e-6) and r["first"] == first and r["robust"] == robust


def test_cone_golden():
    rows = get("cone")["rows"]
    assert [(r["w0"], r["f"]) for r in rows] == list(CONE)
    for r in rows:
        ref = CONE[(r["w0"], r["f"])]
        assert (r["r0"], r["r1"], r["r_coi"], r["r2"]) == pytest.approx(ref[:4], abs=1e-6) and (r["d90"], r["d99"]) == ref[4:]


def test_chirp_golden():
    rows = get("chirp")
    assert len(rows) == len(CHIRP) == 24
    for r in rows:
        assert r["rmse"] == pytest.approx(CHIRP[(r["w0"], r["snr"])], abs=1e-6) and r["runs"] == 10


def test_scaling_golden():
    res = get("scaling")
    assert [t["f"] for t in res["tones"]] == list(TONES) and [n["f"] for n in res["noise"]] == list(NOISE)
    for t in res["tones"]:
        assert (t["peak_l2"], t["peak_rect"]) == pytest.approx(TONES[t["f"]], abs=1e-6)
    for n in res["noise"]:
        assert (n["mean"], n["cv"]) == pytest.approx(NOISE[n["f"]], abs=1e-6)
    for c in res["time_corr"]:
        assert (c["lag"], c["corr"], c["theory"]) == pytest.approx(TIME_CORR[c["lag_s"]], abs=1e-6)
    for c in res["freq_corr"]:
        assert (c["corr"], c["theory"]) == pytest.approx(FREQ_CORR[c["rel"]], abs=1e-6)
    al = res["alarm"]
    assert (al["freqs"], al["cells_share"], al["runs_share"], al["runs"]) == pytest.approx(ALARM, abs=1e-6)
