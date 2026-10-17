"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import cwt_evaluation as ev
import cwt_signals as S

LIMITS = " Faustregel: zwei Töne trennt es ab dem Frequenzverhältnis 1,65, zwei Impulse bei f = 0,100 ab etwa 23 Abtastwerten. Gültig bis f = 0,31."


def make_analysis(signal="Chirp", p=(0.1, 0.3, 0.0), w0=6, fcut=0.1, rmse=None, share=None, resolved=None, both=None, peaks=None, step=None, events=()):
    z = np.zeros(1)
    s = ev.Settings(signal=signal, p1=p[0], p2=p[1], p3=p[2], w0=w0, fcut=fcut)
    return ev.Analysis(s, S.TestSignal(z, (), tuple(events)), z, np.array([0.1, 0.2]), np.zeros((2, 1)), None, None, rmse, share, resolved, both, peaks, step, 0, 0, 1.0 + 3.9 / w0,
                       2.4 * w0 / (2 * 3.141592653589793 * fcut), min(0.45, w0 / (2 * (w0 + 3.7))), 100)


def test_main_verdict_chirp_threshold_and_text():
    kind, text = ev.verdict_main(make_analysis(rmse=0.0003, share=0.963))
    assert kind == "success" and text == "ω₀ = 6: Der Kamm weicht außerhalb der Einflusskegel (mindestens 3 s vom Rand, 96 % der Zeit) im Mittel um 0,0003 von der wahren Frequenz ab." + LIMITS
    assert ev.verdict_main(make_analysis(rmse=0.00499, share=0.9))[0] == "success" and ev.verdict_main(make_analysis(rmse=0.005, share=0.9))[0] == "warning"
    kind, text = ev.verdict_main(make_analysis(rmse=None, share=0.0))
    assert kind == "info" and text.startswith("ω₀ = 6: Für den Kammfehler bleibt kein Zeitpunkt übrig, der mehr als 3 s vom Rand entfernt ist (Einflusskegel); wählen Sie ein kleineres ω₀")


def test_main_verdict_two_tones():
    base = dict(signal="Zwei Töne", p=(0.1, 1.5, 0.0))
    kind, text = ev.verdict_main(make_analysis(resolved=True, **base))
    assert kind == "success" and text == "ω₀ = 6: Die beiden Töne (Verhältnis 1,50) sind im Skalenprofil getrennt." + LIMITS
    kind, text = ev.verdict_main(make_analysis(resolved=False, **base))
    assert kind == "warning" and text == "ω₀ = 6: Die beiden Töne (Verhältnis 1,50) verschwimmen zu einem Gipfel." + LIMITS
    kind, text = ev.verdict_main(make_analysis(resolved=None, **base))
    assert kind == "info" and text.startswith("ω₀ = 6: Mindestens einer der beiden Töne liegt außerhalb des Frequenzgitters")


def test_main_verdict_frequency_step():
    base = dict(signal="Frequenzsprung", p=(0.1, 0.2, 0.5))
    kind, text = ev.verdict_main(make_analysis(step=0.0, **base))
    assert kind == "info" and text == "ω₀ = 6: Der Kamm wechselt am wahren Sprung zur zweiten Frequenz; die Kante ist bei der tieferen Frequenz etwa 23 Abtastwerte breit (2,4 s)." + LIMITS
    assert "wechselt 48 Abtastwerte nach dem wahren Sprung" in ev.verdict_main(make_analysis(step=48.0, **base))[1]
    assert "wechselt 2 Abtastwerte vor dem wahren Sprung" in ev.verdict_main(make_analysis(step=-2.0, **base))[1]
    kind, text = ev.verdict_main(make_analysis(step=None, **base))
    assert kind == "warning" and text.startswith("ω₀ = 6: Der Kamm erreicht die zweite Frequenz nie.")
    assert "etwa 46 Abtastwerte breit" in ev.verdict_main(make_analysis(step=0.0, signal="Frequenzsprung", p=(0.05, 0.2, 0.5)))[1]


def test_main_verdict_two_impulses():
    base = dict(signal="Zwei Impulse", p=(20.0, 0.5, 0.0), events=(512.0, 532.0))
    kind, text = ev.verdict_main(make_analysis(resolved=True, **base))
    assert kind == "success" and text == "ω₀ = 6: Die beiden Impulse im Abstand 20 sind im Zeitverlauf bei f = 0,100 getrennt." + LIMITS
    kind, text = ev.verdict_main(make_analysis(resolved=False, **base))
    assert kind == "warning" and text == "ω₀ = 6: Die beiden Impulse im Abstand 20 verschwimmen bei f = 0,100 zu einem Gipfel; bei höherer Frequenz wird das Wavelet kürzer." + LIMITS
    kind, text = ev.verdict_main(make_analysis(resolved=None, **base))
    assert kind == "info" and text.startswith("ω₀ = 6: Die Frequenz des Zeitverlaufs liegt außerhalb des Frequenzgitters.")


def test_main_verdict_burst():
    base = dict(signal="Tiefer Ton plus Burst", p=(0.04, 0.25, 0.5))
    kind, text = ev.verdict_main(make_analysis(both=True, peaks=(0.04003, 0.25287), **base))
    assert kind == "success" and text == "ω₀ = 6: Zur Burstzeit zeigt das Skalenprofil beide Gipfel: 0,040 (Ton, wahr 0,040) und 0,253 (Burst, wahr 0,250)." + LIMITS
    kind, text = ev.verdict_main(make_analysis(both=False, peaks=(0.04003,), **base))
    assert kind == "warning" and text == ("ω₀ = 6: Zur Burstzeit sind nicht beide Gipfel innerhalb von 10 % der wahren Frequenz zu finden; ein größeres oder kleineres ω₀ verschiebt die Auflösung." + LIMITS)


def test_limits_show_the_rule_of_thumb_for_other_w0():
    text = ev.verdict_main(make_analysis(rmse=0.0003, share=0.9, w0=24, fcut=0.2))[1]
    assert "Frequenzverhältnis 1,16" in text and "bei f = 0,200 ab etwa 46 Abtastwerten" in text and "Gültig bis f = 0,43." in text


def test_verdict_resolution_text():
    rows = [{"f": 0.025, "s": 38.2, "ratio": 1.65, "first": 19, "robust": 90, "stft_ratio": 2.55}, {"f": 0.2, "s": 4.77, "ratio": 1.65, "first": 7, "robust": 12, "stft_ratio": 1.2}]
    kind, text = ev.verdict_resolution({"rows": rows, "w0": 6, "length": 64, "stft_distance": 33, "stft_width": 0.039})
    assert kind == "info"
    assert text == ("Zwei gleich starke Töne trennt das Wavelet (ω₀ = 6) bei jeder Frequenz ab dem Verhältnis 1,65, das Hann-Spektrum (L = 64) ab dem Verhältnis 2,55 bei f = 0,025 und 1,20 bei f = 0,200. "
                    "Zwei Impulse trennt das Wavelet sicher ab 90 Abtastwerten bei f = 0,025 und ab 12 bei f = 0,200, das Spektrogramm immer ab 33. "
                    "**Das Wavelet verteilt die Unschärfe anders: tief fein in der Frequenz und grob in der Zeit, hoch umgekehrt.**")
    rows[1]["ratio"] = 1.7
    assert "ab dem Verhältnis 1,65 bis 1,70, das Hann-Spektrum" in ev.verdict_resolution({"rows": rows, "w0": 6, "length": 64, "stft_distance": 33, "stft_width": 0.039})[1]


def test_verdict_omega_text():
    rows = [{"w0": 3, "s": 4.8, "ratio": 2.6, "first": 5, "robust": 13}, {"w0": 16, "s": 25.5, "ratio": 1.2, "first": 5, "robust": 72}]
    kind, text = ev.verdict_omega(rows)
    assert kind == "info"
    assert text == ("Bei f = 0,1 trennt ω₀ = 3 zwei Töne ab dem Verhältnis 2,60 und zwei Impulse sicher ab 13 Abtastwerten; ω₀ = 16 ab dem Verhältnis 1,20 und ab 72 Abtastwerten. "
                    "**Mehr Perioden im Wavelet: schärfer in der Frequenz, gröber in der Zeit.**")


def test_verdict_cone_text():
    res = {"rows": [{"w0": 6, "f": 0.1, "s": 9.5, "r0": 0.523, "r1": 0.8636, "r_coi": 0.9348, "r2": 0.979, "d90": 12, "d99": 22}],
           "theory": {"r0": 0.5, "r1": 0.8413, "r_coi": 0.9214, "r2": 0.9772, "d90": 1.2816, "d99": 2.3263}}
    kind, text = ev.verdict_cone(res)
    assert kind == "info"
    assert text == ("Am Rand erreicht die Amplitude nur 0,52 des Innenwerts, bei s 0,86, bei √2 s (dem üblichen Einflusskegel) 0,93 und bei 2 s 0,98 (Theorie 0,50 / 0,84 / 0,92 / 0,98, "
                    "in Vielfachen von s für alle ω₀ und f gleich). **Der Kegel ist kein scharfer Rand: 99 % erreicht die Amplitude erst bei 2,33 s.**")


def test_verdict_chirp_names_the_best_omega_at_the_lowest_snr():
    rows = [{"w0": 6, "snr": 10, "rmse": 0.0021, "runs": 1}, {"w0": 24, "snr": 10, "rmse": 0.0003, "runs": 1}, {"w0": 6, "snr": 0, "rmse": 0.0495, "runs": 1}, {"w0": 24, "snr": 0, "rmse": 0.001, "runs": 1},
            {"w0": 12, "snr": 0, "rmse": 0.0155, "runs": 1}, {"w0": 64, "snr": 0, "rmse": 0.0031, "runs": 1}]
    kind, text = ev.verdict_chirp(rows)
    assert kind == "info"
    assert text == ("Bei 0 dB liegt der Kammfehler bei ω₀ = 6 bei 0,0495, bei ω₀ = 24 bei 0,0010 und bei ω₀ = 64 bei 0,0031. **Es gibt ein bestes ω₀ (hier 24), und es liegt weit über dem üblichen 6: "
                    "Das Wavelet muss so lang sein, wie es das Rauschen verlangt, und so kurz, wie es die Steigung erlaubt.**")


def test_verdict_scaling_text():
    res = {"tones": [{"f": 0.3, "rel_l2": 1.0}, {"f": 0.025, "rel_l2": 12.0}], "alarm": {"cells_share": 0.0487, "runs_share": 1.0}}
    kind, text = ev.verdict_scaling(res)
    assert kind == "warning"
    assert text == ("Gleich starke Töne zeigen |W|² im Verhältnis 12 : 1 bei f = 0,025 gegen f = 0,30, mit |W|²/s überall dasselbe; weißes Rauschen ist flach. "
                    "Eine punktweise 5-%-Schwelle trifft 4,9 % der Zellen, aber in 100 % der Läufe mindestens eine. **Die Leistung ist kein lineares Maß, und viele Zellen sind keine vielen Messungen.**")
