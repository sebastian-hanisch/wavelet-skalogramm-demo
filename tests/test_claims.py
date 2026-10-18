"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

import math
from pathlib import Path

import numpy as np
import pytest

import cwt_constants as C
import cwt_evaluation as ev
import cwt_morlet as M
from sweeps_cache import get

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def preset_settings(name, **over):
    p = C.PRESETS[name]
    base = dict(n=p["n"], signal=p["signal"], p1=p["params"][0], p2=p["params"][1], p3=p["params"][2], w0=p["w0"], fmin=p["fmin"], voices=p["voices"], power=p["power"], tcut=p["tcut"], fcut=p["fcut"],
                noise=p["noise"], snr=p["snr"], seed=p["seed"])
    return ev.Settings(**{**base, **over})


def A(name, **over):
    return ev.analyse(preset_settings(name, **over))


def de(x, k=4):
    return f"{x:.{k}f}".replace(".", ",")


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_fitting_w0_follows_the_chirp_to_about_one_thousandth():
    name = "Chirp mit passendem ω₀"
    a = A(name)
    assert a.settings.w0 == 24 and a.settings.snr == 0 and de(a.rmse, 3) == "0,001" and "etwa 0,001" in C.PRESET_HELP[name]


def test_preset_small_w0_is_over_fifteen_times_worse_and_large_w0_about_twice():
    good = A("Chirp mit passendem ω₀")
    small = A("Zu kleines ω₀: die Frequenz verschwimmt")
    large = A("Zu großes ω₀: der Chirp verschmiert")
    assert small.settings.w0 == 6 and small.rmse / good.rmse > 15 and "Fünfzehnfache" in C.PRESET_HELP["Zu kleines ω₀: die Frequenz verschwimmt"]
    assert large.settings.w0 == 64 and 1.9 < large.rmse / good.rmse < 2.1 and "auf das Doppelte" in C.PRESET_HELP["Zu großes ω₀: der Chirp verschmiert"]
    assert round(float(M.scale_of(0.1, 64))) == 102 and "s = 102" in C.PRESET_HELP["Zu großes ω₀: der Chirp verschmiert"]
    assert round(100 * large.interior_share) == 68 and "68 %" in C.PRESET_HELP["Zu großes ω₀: der Chirp verschmiert"]


def test_preset_burst_shows_both_peaks_near_forty_thousandths_and_a_quarter():
    name = "Tiefer Ton plus Burst: Auflösung folgt der Frequenz"
    a = A(name)
    assert a.both_found is True and de(a.burst_peaks[0], 3) == "0,040" and de(a.burst_peaks[1], 3) == "0,253"
    assert "Burst bei 0,25" in C.PRESET_HELP[name] and "ton bei 0,04".lower() in C.PRESET_HELP[name].lower() and "0,040 und 0,253" in C.PRESET_HELP[name]


def test_preset_two_impulses_blur_at_low_frequency_and_separate_at_high_frequency():
    name = "Zwei Impulse: tief gröber, hoch feiner"
    a = A(name)
    assert a.settings.p1 == 20.0 and a.resolved is False and a.settings.fcut == 0.1 and A(name, fcut=0.3).resolved is True
    assert round(float(M.scale_of(0.1, 6)), 1) == 9.5 and "Abstand 20" in C.PRESET_HELP[name] and "s = 9,5" in C.PRESET_HELP[name] and "0,3" in C.PRESET_HELP[name]


# --- Auflösung --------------------------------------------------------------------------------------------------------------------------------


def test_resolution_numbers_in_the_readme_and_the_app():
    res = get("resolution")
    rows = {r["f"]: r for r in res["rows"]}
    assert [rows[f]["ratio"] for f in (0.025, 0.05, 0.1, 0.2)] == [1.65] * 4 and "ab dem Frequenzverhältnis 1,65" in README
    assert [rows[f]["stft_ratio"] for f in (0.025, 0.05, 0.1, 0.2)] == [2.55, 1.8, 1.4, 1.2] and "2,55 / 1,8 / 1,4 / 1,2" in README and de(res["stft_width"], 3) == "0,039" and "0,039" in README
    assert [rows[f]["robust"] for f in (0.025, 0.05, 0.1, 0.2)] == [90, 45, 23, 12] and "90 / 45 / 23 / 12" in README and res["stft_distance"] == 33 and "ab 33 bei jeder Frequenz" in README
    ratios = [rows[f]["robust"] / rows[f]["s"] for f in rows]
    assert 2.3 < min(ratios) and max(ratios) < 2.6 and "das 2,4-Fache von s" in README and C.TIME_RULE_K == 2.4
    assert rows[0.025]["first"] == 19 and "19 Abtastwerte bei f = 0,025 gegen 90 sicher" in README
    cross = res["stft_width"] / (rows[0.1]["ratio"] - 1)
    assert round(cross, 2) == 0.06 and "f ≈ 0,06" in README and "etwa 0,06" in APP
    cross_time = C.TIME_RULE_K * 6 / (2 * math.pi * res["stft_distance"])        # Wavelet 2,4 s gleich 33 Abtastwerte des Spektrogramms
    assert 0.065 < cross_time < 0.07 and "f ≈ 0,06 bis 0,07 (ω₀ = 6, L = 64)" in README
    for f in (0.1, 0.2):
        assert (rows[f]["stft_ratio"] - 1) * f == pytest.approx(res["stft_width"], abs=0.003)


def test_omega_numbers_and_the_rules_of_thumb():
    rows = {r["w0"]: r for r in get("omega")}
    assert [rows[w]["ratio"] for w in (3, 4, 5, 6, 8, 10, 12, 16)] == [2.6, 2.05, 1.8, 1.65, 1.45, 1.35, 1.3, 1.2] and "2,6 / 2,05 / 1,8 / 1,65 / 1,45 / 1,35 / 1,3 / 1,2" in README
    assert [rows[w]["robust"] for w in rows] == [13, 14, 22, 23, 33, 42, 52, 72] and "13 / 14 / 22 / 23 / 33 / 42 / 52 / 72" in README
    k = [(rows[w]["ratio"] - 1) * w for w in rows]
    assert round(min(k), 1) == 3.2 and round(max(k), 1) == 4.8 and "zwischen 3,2 und 4,8" in README and min(k) <= C.FREQ_RULE_K <= max(k) and C.FREQ_RULE_K == 3.9
    t = [rows[w]["robust"] / rows[w]["s"] for w in rows]
    assert round(min(t), 1) == 2.2 and round(max(t), 1) == 2.8 and "zwischen 2,2 und 2,8 s" in README and "1 + 3,9" in README and "1 + 3{,}9" in APP
    assert 2.5 < sum(t) / len(t) < 2.7 and "im Mittel etwa 2,5 s" in README                                 # Mittel über die ω₀-Reihe 2,61, über beide Reihen 2,55
    assert all(rows[w]["robust"] > C.TIME_RULE_K * rows[w]["s"] for w in (8, 10, 12, 16)) and "für ω₀ ab 8 liegt der gemessene sichere Impulsabstand über 2,4 s" in README


def test_cone_numbers_in_the_readme_and_the_app():
    cone = get("cone")
    th = cone["theory"]
    assert [de(th[k], 2) for k in ("r0", "r1", "r_coi", "r2")] == ["0,50", "0,84", "0,92", "0,98"] and "(0,50 / 0,84 / 0,92 / 0,98)" in README
    first = cone["rows"][0]
    assert [de(first[k], 2) for k in ("r0", "r1", "r_coi", "r2")] == ["0,52", "0,86", "0,93", "0,98"] and "0,52 des Innenwerts, bei s 0,86, bei √2 s (dem üblichen Einflusskegel) 0,93 und bei 2 s 0,98" in README
    for r in cone["rows"]:
        assert abs(r["r0"] - th["r0"]) <= 0.07 and abs(r["r1"] - th["r1"]) < 0.05 and abs(r["r_coi"] - th["r_coi"]) < 0.03 and abs(r["r2"] - th["r2"]) < 0.015
        assert de(r["d90"] / r["s"], 2) in ("1,26", "1,25") and de(r["d99"] / r["s"], 2) in ("2,30", "2,31")
    assert de(th["d90"], 2) == "1,28" and de(th["d99"], 2) == "2,33" and "ab 1,26 s, 99 % ab 2,30 s" in README and "2,3 s" in APP
    assert round(100 * (1 - th["r_coi"])) == 8 and "8 % der Amplitude" in APP


# --- Chirp ------------------------------------------------------------------------------------------------------------------------------------


def test_chirp_best_w0_is_24_at_noise_and_the_table_numbers():
    rows = {(r["w0"], r["snr"]): r["rmse"] for r in get("chirp")}
    for snr in (0, 10):
        assert min(C.EXP_CHIRP_W0, key=lambda w: rows[(w, snr)]) == 24
    assert sorted(C.EXP_CHIRP_W0, key=lambda w: rows[(w, 0)])[:2] == [24, 32] and "bei 24 bis 32 (bei 10 dB eindeutig 24)" in README
    assert min(C.EXP_CHIRP_W0, key=lambda w: rows[(w, 60)]) == 6 and rows[(64, 60)] > rows[(6, 60)] and "bei 60 dB steigt der Fehler" in README
    for snr, line in ((0, "0,0495 / 0,0368 / 0,0155 / 0,0023 / 0,0010 / 0,0011 / 0,0020 / 0,0031"), (10, "0,0021 / 0,0014 / 0,0007 / 0,0005 / 0,0003 / 0,0004 / 0,0007 / 0,0013")):
        assert " / ".join(de(rows[(w, snr)]) for w in C.EXP_CHIRP_W0) == line and line in README
    assert de(rows[(64, 60)], 5) == "0,00057" and "0,00057 bei ω₀ = 64" in README and max(rows[(w, 60)] for w in (6, 8, 12, 16)) < 1e-4
    assert (C.EXP_CHIRP_F0, C.EXP_CHIRP_F1, C.EXP_CHIRP_N, C.EXP_CHIRP_RUNS, C.EDGE_S) == (0.1, 0.3, 1024, 10, 3.0) and "mindestens 3 s vom Rand" in README and "10 Läufe je Zeile beim Chirp" in README


# --- Skalierung, Korrelation, Falschalarm -------------------------------------------------------------------------------------------------------


def test_scaling_numbers_in_the_readme():
    res = get("scaling")
    tones = {t["f"]: t for t in res["tones"]}
    assert " / ".join(de(tones[f]["peak_l2"], 2) for f in (0.3, 0.2, 0.1, 0.05, 0.025)) == "2,84 / 4,26 / 8,52 / 17,04 / 34,08" and "2,84 / 4,26 / 8,52 / 17,04 / 34,08" in README
    assert all(de(t["peak_rect"], 3) == "0,886" for t in tones.values()) and "überall 0,886" in README and round(tones[0.025]["rel_l2"]) == 12 and "das 12-Fache" in README
    means = [n["mean"] for n in res["noise"]]
    assert de(min(means), 2) == "1,00" and de(max(means), 2) == "1,02" and "E\\|W\\|² 1,00 bis 1,02" in README
    cvs = [n["cv"] for n in res["noise"]]
    assert 0.98 < min(cvs) and max(cvs) < 1.02 and "Variationskoeffizienten 1,0" in README


def test_correlation_numbers_in_the_readme_and_the_app():
    res = get("scaling")
    tc = {c["lag_s"]: c for c in res["time_corr"]}
    assert " / ".join(de(tc[k]["corr"], 3) for k in (0.5, 1.0, 2.0, 3.0)) == "0,874 / 0,585 / 0,144 / 0,010" and "0,874 / 0,585 / 0,144 / 0,010" in README
    assert " / ".join(de(tc[k]["theory"], 3) for k in (0.5, 1.0, 2.0, 3.0)) == "0,872 / 0,578 / 0,138 / 0,010" and "0,872 / 0,578 / 0,138 / 0,010" in README
    fc = {c["rel"]: c for c in res["freq_corr"]}
    assert " / ".join(de(fc[r]["corr"], 3) for r in (0.05, 0.1, 0.2, 0.3)) == "0,959 / 0,851 / 0,548 / 0,290" and "0,959 / 0,851 / 0,548 / 0,290" in README
    assert " / ".join(de(fc[r]["theory"], 3) for r in (0.05, 0.1, 0.2, 0.3)) == "0,957 / 0,846 / 0,545 / 0,290" and "0,957 / 0,846 / 0,545 / 0,290" in README
    assert de(tc[1.0]["corr"], 2) == "0,58" and de(fc[0.1]["corr"], 2) == "0,85" and "0,58 bei Abstand s, 0,85 bei 10 % Frequenzunterschied" in APP


def test_false_alarm_numbers_in_the_readme():
    al = get("scaling")["alarm"]
    assert de(100 * al["cells_share"], 1) == "4,9" and al["runs_share"] == 1.0 and al["runs"] == 200 and al["freqs"] == 28 and C.EXP_SCALE_NOISE_N == 512
    assert "Sie trifft 4,9 % der Zellen, aber in 100 % der 200 Läufe mindestens eine Zelle (28 Frequenzen, n = 512)" in README and "400 bei der Korrelation, 200 beim Falschalarm" in README and C.EXP_SCALE_RUNS == 400


# --- Konstanten, Gültigkeit, Mini-Instanz --------------------------------------------------------------------------------------------------------


def test_method_numbers_in_the_readme_match_the_constants():
    assert (min(C.N_CHOICES), max(C.N_CHOICES), min(C.W0_CHOICES), max(C.W0_CHOICES)) == (256, 2048, 3, 64) and "n = 256 bis 2048, ω₀ = 3 bis 64" in README
    assert C.EXP_SEED == 950000 and "Seed 950000" in README and (min(C.VOICES_CHOICES), max(C.VOICES_CHOICES)) == (8, 64) and "8 bis 64 Stimmen je Oktave" in README
    assert C.EXP_RES_RUNS == 60 and "in 60 Läufen" in README and C.NYQUIST_MARGIN == 3.7 if hasattr(C, "NYQUIST_MARGIN") else M.NYQUIST_MARGIN == 3.7
    assert "ω₀/(2(ω₀ + 3,7))" in README and M.f_max(6.0) == pytest.approx(0.3093, abs=1e-4) and "höchstens 0,45" in README and M.f_max(100.0) == 0.45


def test_deviation_beyond_f_max_is_over_five_percent_at_04():
    y = np.random.default_rng(1).normal(size=300)
    d = M.cwt_direct(y, 0.4, 6.0)
    dev = np.abs(M.cwt(y, [0.4], 6.0)[0] - d).max() / np.abs(d).max()
    short = np.random.default_rng(1).normal(size=100)
    dev_short = np.abs(M.cwt(short, [0.4], 6.0)[0] - M.cwt_direct(short, 0.4, 6.0)).max() / np.abs(M.cwt_direct(short, 0.4, 6.0)).max()
    assert dev > 0.05 and dev_short > 0.05 and "bei ω₀ = 6 und f = 0,4 um mehr als 5 %" in README and "bei f = 0,4 und ω₀ = 6 um mehr als 5 %" in APP and 0.4 > M.f_max(6.0)
    assert "(bei ω₀ = 6: 0,31)" in APP and de(M.f_max(6.0), 2) == "0,31"


def test_hand_set_mini_instance_numbers_in_the_readme():
    assert "0,0719 / 0,1724 / 0,3221 / 0,4687 / 0,5311" in README
    x = np.zeros(9)
    x[4] = 1.0
    w = np.abs(M.cwt_direct(x, 6.0 / (2 * math.pi * 2.0), 6.0))
    assert " / ".join(de(v) for v in w[:5]) == "0,0719 / 0,1724 / 0,3221 / 0,4687 / 0,5311" and "2,909097" in README


def test_the_thresholds_used_in_the_texts():
    assert C.OK_RMSE == 0.005 and C.BURST_TOLERANCE == 0.1 and C.FALSE_ALARM_LEVEL == 0.05 and C.FREQ_RULE_K == 3.9 and C.TIME_RULE_K == 2.4
    assert "innerhalb von 10 %" in ev.verdict_main(ev.analyse(ev.Settings(signal="Tiefer Ton plus Burst", p1=0.04, p2=0.25, p3=0.5, w0=3)))[1]
