"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import cwt_constants as C
import cwt_morlet as M
import cwt_resolution as R
import cwt_signals as S
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def noise_sigma(snr_db: float) -> float:
    """Standardabweichung des Rauschens gegen eine Linie der Amplitude 1 (Leistung 1/2): sqrt(0,5 / 10^(SNR/10))."""
    return math.sqrt(0.5 / 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    signal: str = C.DEFAULT_SIGNAL
    p1: float = C.PARAMS[C.DEFAULT_SIGNAL][0][4]
    p2: float = C.PARAMS[C.DEFAULT_SIGNAL][1][4]
    p3: float = 0.0
    w0: int = C.DEFAULT_W0
    fmin: float = C.DEFAULT_FMIN
    voices: int = C.DEFAULT_VOICES
    power: str = C.DEFAULT_POWER
    tcut: float = C.DEFAULT_TCUT
    fcut: float = C.DEFAULT_FCUT
    noise: str = C.DEFAULT_NOISE
    snr: int = C.DEFAULT_SNR
    seed: int = C.DEFAULT_SEED

    @property
    def has_noise(self) -> bool:
        return self.noise == "an"

    @property
    def rectified(self) -> bool:
        return self.power == "gleichgewichtet"


def make_signal(s: Settings) -> S.TestSignal:
    """Das saubere Signal des gewählten Typs (Anfangsphasen 0,3 und 1,1)."""
    if s.signal == "Chirp":
        return S.linear_chirp(s.n, s.p1, s.p2, 0.3)
    if s.signal == "Zwei Töne":
        return S.two_tones(s.n, s.p1, s.p2, *C.TONE_PHASES)
    if s.signal == "Frequenzsprung":
        return S.frequency_step(s.n, s.p1, s.p2, int(round(s.p3 * s.n)), 0.3)
    if s.signal == "Zwei Impulse":
        t1 = int(round(s.p2 * s.n))
        return S.two_impulses(s.n, t1, max(1, min(int(round(s.p1)), s.n - 1 - t1)))
    if s.signal == "Tiefer Ton plus Burst":
        return S.tone_plus_burst(s.n, s.p1, s.p2, int(round(s.p3 * s.n)), 0.3)
    raise ValueError(f"unbekanntes Signal: {s.signal}")


def build_measurement(s: Settings) -> tuple[S.TestSignal, np.ndarray]:
    sig = make_signal(s)
    x = sig.x.copy()
    if s.has_noise:
        x = x + noise_sigma(s.snr) * G.SplitMix64(s.seed).normal(s.n)
    return sig, x


def frequency_grid(s: Settings) -> np.ndarray:
    """Analysefrequenzen: von der kleinsten Frequenz (höchstens die Hälfte der Obergrenze) bis min(0,45; f_max(w0)), logarithmisch mit `voices` Stimmen je Oktave."""
    top = M.f_max(float(s.w0))
    return M.freq_grid(min(s.fmin, top / 2.0), top, s.voices)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    signal: S.TestSignal
    x: np.ndarray
    freqs: np.ndarray
    power: np.ndarray               # Frequenzen x Zeit
    ridge: np.ndarray | None        # Kamm je Zeitpunkt (Chirp und Frequenzsprung)
    truth: np.ndarray | None        # wahre Frequenzen je Zeitpunkt (Spuren x Zeit)
    rmse: float | None
    interior_share: float | None    # Anteil der Zeitpunkte außerhalb der Einflusskegel, die in den Kammfehler eingehen
    resolved: bool | None           # zwei Töne (Skalenprofil) bzw. zwei Impulse (Zeitverlauf)
    both_found: bool | None         # Burst und tiefer Ton im Skalenprofil zur Burstzeit
    burst_peaks: tuple | None
    step_error: float | None
    t_index: int
    f_index: int
    ratio_limit: float              # Faustregel 1 + K / w0
    time_limit: float               # Faustregel K mal s bei der Frequenz des Zeitverlaufs
    fmax: float
    cells: int


def _step_error(ridge, times, s: Settings) -> float | None:
    """Zeitpunkt, an dem der Kamm die Mitte zwischen beiden Frequenzen zur zweiten Frequenz hin überschreitet, minus der wahre Sprungzeitpunkt (None, wenn nie)."""
    mid = 0.5 * (s.p1 + s.p2)
    crossed = ridge > mid if s.p2 > s.p1 else ridge < mid
    if not crossed.any():
        return None
    return float(times[int(np.argmax(crossed))] - int(round(s.p3 * s.n)))


def _nearest(freqs, f: float) -> int | None:
    """Index der Analysefrequenz nächst f, None, wenn f um mehr als einen Gitterschritt außerhalb des Gitters liegt."""
    ratio = freqs[1] / freqs[0]
    if f < freqs[0] / ratio or f > freqs[-1] * ratio:
        return None
    return int(np.argmin(np.abs(np.log(freqs / f))))


def interior_mask(truth_freq, n: int, w0: float) -> np.ndarray:
    """Zeitpunkte, die um mindestens EDGE_S mal s(wahre Frequenz) vom Rand entfernt sind."""
    s = M.scale_of(truth_freq, w0)
    t = np.arange(n)
    return (t >= C.EDGE_S * s) & (t <= n - 1 - C.EDGE_S * s)


def analyse(s: Settings) -> Analysis:
    sig, x = build_measurement(s)
    freqs = frequency_grid(s)
    w0 = float(s.w0)
    W = M.cwt(x, freqs, w0)
    p = M.power(W, freqs, w0, s.rectified)
    times = np.arange(s.n)
    ridge = truth = rmse = share = resolved = both = burst_peaks = step = None
    t_index = int(round(s.tcut * (s.n - 1)))
    f_near = _nearest(freqs, s.fcut)
    f_index = int(np.argmin(np.abs(np.log(freqs / s.fcut))))
    if sig.tracks:
        truth = np.array([tr(times) for tr in sig.tracks])
    if s.signal in ("Chirp", "Frequenzsprung"):
        ridge = M.ridge_frequencies(p, freqs)
        if s.signal == "Chirp":
            mask = interior_mask(truth[0], s.n, w0) & (truth[0] >= freqs[0]) & (truth[0] <= freqs[-1])
            share = float(mask.mean())
            rmse = float(math.sqrt(np.mean((ridge[mask] - truth[0][mask]) ** 2))) if mask.any() else None
        else:
            step = _step_error(ridge, times, s)
    elif s.signal == "Zwei Töne":
        i1, i2 = _nearest(freqs, s.p1), _nearest(freqs, s.p1 * s.p2)
        if i1 is not None and i2 is not None:
            resolved = R.resolved(p[:, t_index], i1, i2)
    elif s.signal == "Zwei Impulse":
        if f_near is not None:
            t1, t2 = (int(round(e)) for e in sig.events)
            resolved = R.resolved(p[f_near], t1, t2)
    else:
        k = int(round(sig.events[0]))
        burst_peaks = tuple(M.peak_frequencies(p[:, k], freqs, 2))
        low, high = s.p1, s.p2
        both = len(burst_peaks) == 2 and abs(burst_peaks[0] - low) < C.BURST_TOLERANCE * low and abs(burst_peaks[1] - high) < C.BURST_TOLERANCE * high
    return Analysis(s, sig, x, freqs, p, ridge, truth, rmse, share, resolved, both, burst_peaks, step, t_index, f_index, 1.0 + C.FREQ_RULE_K / w0,
                    C.TIME_RULE_K * float(M.scale_of(s.fcut, w0)), M.f_max(w0), p.shape[0] * p.shape[1])


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _rng():
    return G.SplitMix64(C.EXP_RES_SEED)


def sweep_resolution(fs=C.EXP_RES_F, w0: float = C.EXP_RES_W0, runs: int = C.EXP_RES_RUNS, length: int = C.EXP_RES_STFT_L) -> dict:
    """Auflösung gegen Frequenz: Frequenzverhältnis, ab dem zwei gleich starke Töne getrennt sind, und Abstand, ab dem zwei Impulse sicher getrennt sind (Wavelet, ω₀ fest);
    zum Vergleich dieselben Größen beim Spektrogramm (Hann, Länge L)."""
    rows = []
    for f in fs:
        first, robust = R.min_distance(f, w0)
        rows.append({"f": f, "s": float(M.scale_of(f, w0)), "ratio": R.min_ratio(f, w0, _rng, runs), "first": first, "robust": robust, "stft_ratio": R.stft_min_ratio(f, length, _rng, runs)})
    return {"rows": rows, "w0": w0, "length": length, "stft_distance": R.stft_min_impulse_distance(length), "stft_width": 2.5 / length}


def sweep_omega(w0s=C.EXP_OMEGA_W0, f: float = C.EXP_OMEGA_F, runs: int = C.EXP_RES_RUNS) -> list[dict]:
    """Morlet-ω₀ bei fester Frequenz: Frequenzverhältnis und sicherer Impulsabstand."""
    rows = []
    for w0 in w0s:
        first, robust = R.min_distance(f, float(w0))
        rows.append({"w0": w0, "s": float(M.scale_of(f, w0)), "ratio": R.min_ratio(f, float(w0), _rng, runs), "first": first, "robust": robust})
    return rows


def sweep_cone(combos=C.EXP_CONE_COMBOS, n: int = C.EXP_CONE_N, phase: float = C.EXP_CONE_PHASE) -> dict:
    """Einflusskegel: Verhältnis der Amplitude |W| im Abstand d vom Rand zur Amplitude im Inneren (Ton der Amplitude 1, Nullen angehängt) gegen die Theorie Phi(d / s)."""
    rows = []
    t = np.arange(n)
    for w0, f in combos:
        w = np.abs(M.cwt(np.cos(2 * math.pi * f * t + phase), [f], float(w0))[0])
        s = float(M.scale_of(f, w0))
        inner = w[n // 2 - 50:n // 2 + 50].mean()
        ratio = w / inner
        at = lambda d: float(ratio[int(round(d))])
        rows.append({"w0": w0, "f": f, "s": s, "r0": at(0), "r1": at(s), "r_coi": at(math.sqrt(2) * s), "r2": at(2 * s), "d90": int(np.argmax(ratio >= 0.9)), "d99": int(np.argmax(ratio >= 0.99))})
    theory = {"r0": 0.5, "r1": float(M.cone_ratio(1.0)), "r_coi": float(M.cone_ratio(math.sqrt(2))), "r2": float(M.cone_ratio(2.0)), "d90": float(M.cone_distance(1.0, 0.9)), "d99": float(M.cone_distance(1.0, 0.99))}
    return {"rows": rows, "theory": theory}


def sweep_chirp(w0s=C.EXP_CHIRP_W0, snrs=C.EXP_CHIRP_SNR, runs: int = C.EXP_CHIRP_RUNS, n: int = C.EXP_CHIRP_N, voices: int = C.EXP_CHIRP_VOICES) -> list[dict]:
    """Chirp in Rauschen: RMSE des Kamms (Maximum von |W|^2/s, Parabelverfeinerung) gegen die wahre Frequenz im Inneren (mindestens EDGE_S mal s vom Rand), je ω₀ und Rauschabstand."""
    sig = S.linear_chirp(n, C.EXP_CHIRP_F0, C.EXP_CHIRP_F1)
    truth = sig.tracks[0](np.arange(n))
    freqs = M.freq_grid(C.EXP_CHIRP_FMIN, C.EXP_CHIRP_FMAX, voices)
    rows = []
    for w0 in w0s:
        mask = interior_mask(truth, n, float(w0))
        for snr in snrs:
            rng = G.SplitMix64(C.EXP_SEED + 1000 * int(w0) + snr)
            x = sig.x[None, :] + noise_sigma(snr) * rng.normal(runs * n).reshape(runs, n)
            ridge = M.ridge_frequencies(M.power(M.cwt(x, freqs, float(w0)), freqs, float(w0)), freqs)
            rows.append({"w0": w0, "snr": snr, "rmse": float(math.sqrt(np.mean((ridge[:, mask] - truth[mask][None, :]) ** 2))), "runs": runs})
    return rows


def sweep_scaling(fs=C.EXP_SCALE_F, w0: float = 6.0) -> dict:
    """Skalierung und Zellen: gleich starke Töne (|W|^2 wächst mit s), weißes Rauschen (E|W|^2 flach, Variationskoeffizient 1), Korrelation der Zellleistung in Zeit und Frequenz
    gegen die Theorie, Falschalarm einer punktweisen 5-%-Schwelle."""
    n = C.EXP_SCALE_N
    t = np.arange(n)
    tones = []
    for f in fs:
        fg = M.freq_grid(f * 0.8, f * 1.25, 64)
        W = np.abs(M.cwt(np.cos(2 * math.pi * f * t + 0.4), fg, w0)[:, n // 2 - 40:n // 2 + 40]) ** 2
        s = M.scale_of(fg, w0)
        tones.append({"f": f, "s": float(M.scale_of(f, w0)), "peak_l2": float(W.max()), "peak_rect": float((W / s[:, None]).max())})
    base = tones[0]
    for r in tones:
        r["rel_l2"] = r["peak_l2"] / base["peak_l2"]
        r["rel_rect"] = r["peak_rect"] / base["peak_rect"]
    nn, runs = C.EXP_SCALE_NOISE_N, C.EXP_SCALE_RUNS
    x = G.SplitMix64(C.EXP_SEED + 2000).normal(runs * nn).reshape(runs, nn)
    noise = []
    for f in fs:
        s = float(M.scale_of(f, w0))
        core = np.abs(M.cwt(x, [f], w0)[:, 0, :]) ** 2
        core = core[:, int(4 * s):nn - int(4 * s)] if 4 * s < nn / 2 - 6 else core[:, nn // 2 - 6:nn // 2 + 6]
        noise.append({"f": f, "s": s, "mean": float(core.mean()), "cv": float(core.std() / core.mean())})
    f0 = C.EXP_SCALE_CORR_F
    s0 = float(M.scale_of(f0, w0))
    base_power = np.abs(M.cwt(x, [f0], w0)[:, 0, :]) ** 2
    time_corr = []
    for k in C.EXP_SCALE_LAGS:
        lag = int(round(k * s0))
        time_corr.append({"lag_s": k, "lag": lag, "corr": R.correlation(base_power[:, 100:nn - 100 - lag], base_power[:, 100 + lag:nn - 100]), "theory": math.exp(-((lag / s0) ** 2) / 2.0)})
    freq_corr = []
    for rel in C.EXP_SCALE_REL:
        other = np.abs(M.cwt(x, [f0 * (1 + rel)], w0)[:, 0, :]) ** 2
        freq_corr.append({"rel": rel, "corr": R.correlation(base_power[:, 100:nn - 100], other[:, 100:nn - 100]), "theory": R.frequency_correlation(w0, rel)})
    fg = M.freq_grid(C.EXP_SCALE_FA_FMIN, M.f_max(w0), C.EXP_SCALE_FA_VOICES)
    over = np.abs(M.cwt(x[:C.EXP_SCALE_FA_RUNS], fg, w0)) ** 2 > -math.log(C.FALSE_ALARM_LEVEL)
    alarm = {"freqs": len(fg), "cells_share": float(over.mean()), "runs_share": float(over.any(axis=(1, 2)).mean()), "runs": C.EXP_SCALE_FA_RUNS}
    return {"tones": tones, "noise": noise, "time_corr": time_corr, "freq_corr": freq_corr, "alarm": alarm, "w0": w0}


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def _limits(a: Analysis) -> str:
    s = a.settings
    return (f" Faustregel: zwei Töne trennt es ab dem Frequenzverhältnis {de(a.ratio_limit, 2)}, zwei Impulse bei f = {de(s.fcut, 3)} ab etwa {de(a.time_limit, 0)} Abtastwerten."
            f" Gültig bis f = {de(a.fmax, 2)}.")


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    head = f"ω₀ = {s.w0}: "
    if s.signal == "Chirp":
        if a.rmse is None:
            return "info", head + "Für den Kammfehler bleibt kein Zeitpunkt übrig, der mehr als 3 s vom Rand entfernt ist (Einflusskegel); wählen Sie ein kleineres ω₀ oder eine längere Aufzeichnung." + _limits(a)
        ok = a.rmse < C.OK_RMSE
        return ("success" if ok else "warning"), (head + f"Der Kamm weicht außerhalb der Einflusskegel (mindestens 3 s vom Rand, {de(100 * a.interior_share, 0)} % der Zeit) im Mittel um {de(a.rmse, 4)} von der wahren Frequenz ab."
                                                  + _limits(a))
    if s.signal == "Zwei Töne":
        if a.resolved is None:
            return "info", head + "Mindestens einer der beiden Töne liegt außerhalb des Frequenzgitters (kleinste Frequenz oder gültiger Bereich)." + _limits(a)
        if a.resolved:
            return "success", head + f"Die beiden Töne (Verhältnis {de(s.p2, 2)}) sind im Skalenprofil getrennt." + _limits(a)
        return "warning", head + f"Die beiden Töne (Verhältnis {de(s.p2, 2)}) verschwimmen zu einem Gipfel." + _limits(a)
    if s.signal == "Frequenzsprung":
        if a.step_error is None:
            return "warning", head + "Der Kamm erreicht die zweite Frequenz nie." + _limits(a)
        width = C.TIME_RULE_K * float(M.scale_of(min(s.p1, s.p2), s.w0))
        where = "am wahren Sprung" if a.step_error == 0 else f"{de(abs(a.step_error), 0)} Abtastwerte {'nach' if a.step_error > 0 else 'vor'} dem wahren Sprung"
        return "info", head + f"Der Kamm wechselt {where} zur zweiten Frequenz; die Kante ist bei der tieferen Frequenz etwa {de(width, 0)} Abtastwerte breit (2,4 s)." + _limits(a)
    if s.signal == "Zwei Impulse":
        dt = int(round(a.signal.events[1] - a.signal.events[0]))
        if a.resolved is None:
            return "info", head + "Die Frequenz des Zeitverlaufs liegt außerhalb des Frequenzgitters." + _limits(a)
        if a.resolved:
            return "success", head + f"Die beiden Impulse im Abstand {dt} sind im Zeitverlauf bei f = {de(s.fcut, 3)} getrennt." + _limits(a)
        return "warning", head + f"Die beiden Impulse im Abstand {dt} verschwimmen bei f = {de(s.fcut, 3)} zu einem Gipfel; bei höherer Frequenz wird das Wavelet kürzer." + _limits(a)
    if a.both_found:
        return "success", head + f"Zur Burstzeit zeigt das Skalenprofil beide Gipfel: {de(a.burst_peaks[0], 3)} (Ton, wahr {de(s.p1, 3)}) und {de(a.burst_peaks[1], 3)} (Burst, wahr {de(s.p2, 3)})." + _limits(a)
    return "warning", head + "Zur Burstzeit sind nicht beide Gipfel innerhalb von 10 % der wahren Frequenz zu finden; ein größeres oder kleineres ω₀ verschiebt die Auflösung." + _limits(a)


def verdict_resolution(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    first, last = rows[0], rows[-1]
    return "info", (f"Zwei gleich starke Töne trennt das Wavelet (ω₀ = {res['w0']}) bei jeder Frequenz ab dem Verhältnis {de(min(ratios), 2)}"
                    f"{'' if len(set(ratios)) == 1 else ' bis ' + de(max(ratios), 2)}, das Hann-Spektrum (L = {res['length']}) ab dem Verhältnis {de(first['stft_ratio'], 2)} bei f = {de(first['f'], 3)} und {de(last['stft_ratio'], 2)} "
                    f"bei f = {de(last['f'], 3)}. Zwei Impulse trennt das Wavelet sicher ab {first['robust']} Abtastwerten bei f = {de(first['f'], 3)} und ab {last['robust']} bei f = {de(last['f'], 3)}, das Spektrogramm immer ab "
                    f"{res['stft_distance']}. **Das Wavelet verteilt die Unschärfe anders: tief fein in der Frequenz und grob in der Zeit, hoch umgekehrt.**")


def verdict_omega(rows: list[dict]) -> tuple[str, str]:
    lo, hi = rows[0], rows[-1]
    return "info", (f"Bei f = {de(C.EXP_OMEGA_F, 1)} trennt ω₀ = {lo['w0']} zwei Töne ab dem Verhältnis {de(lo['ratio'], 2)} und zwei Impulse sicher ab {lo['robust']} Abtastwerten; ω₀ = {hi['w0']} ab dem Verhältnis "
                    f"{de(hi['ratio'], 2)} und ab {hi['robust']} Abtastwerten. **Mehr Perioden im Wavelet: schärfer in der Frequenz, gröber in der Zeit.**")


def verdict_cone(res: dict) -> tuple[str, str]:
    rows, th = res["rows"], res["theory"]
    r = rows[0]
    return "info", (f"Am Rand erreicht die Amplitude nur {de(r['r0'], 2)} des Innenwerts, bei s {de(r['r1'], 2)}, bei √2 s (dem üblichen Einflusskegel) {de(r['r_coi'], 2)} und bei 2 s {de(r['r2'], 2)} (Theorie "
                    f"{de(th['r0'], 2)} / {de(th['r1'], 2)} / {de(th['r_coi'], 2)} / {de(th['r2'], 2)}, in Vielfachen von s für alle ω₀ und f gleich). **Der Kegel ist kein scharfer Rand: 99 % erreicht die Amplitude erst bei {de(th['d99'], 2)} s.**")


def verdict_chirp(rows: list[dict]) -> tuple[str, str]:
    snr = min(r["snr"] for r in rows)
    sub = sorted((r for r in rows if r["snr"] == snr), key=lambda r: r["w0"])
    best = min(sub, key=lambda r: r["rmse"])
    first, last = sub[0], sub[-1]
    return "info", (f"Bei {snr} dB liegt der Kammfehler bei ω₀ = {first['w0']} bei {de(first['rmse'], 4)}, bei ω₀ = {best['w0']} bei {de(best['rmse'], 4)} und bei ω₀ = {last['w0']} bei {de(last['rmse'], 4)}. "
                    f"**Es gibt ein bestes ω₀ (hier {best['w0']}), und es liegt weit über dem üblichen 6: Das Wavelet muss so lang sein, wie es das Rauschen verlangt, und so kurz, wie es die Steigung erlaubt.**")


def verdict_scaling(res: dict) -> tuple[str, str]:
    tones, alarm = res["tones"], res["alarm"]
    last = tones[-1]
    return "warning", (f"Gleich starke Töne zeigen |W|² im Verhältnis {de(last['rel_l2'], 0)} : 1 bei f = {de(last['f'], 3)} gegen f = {de(tones[0]['f'], 2)}, mit |W|²/s überall dasselbe; weißes Rauschen ist flach. "
                       f"Eine punktweise 5-%-Schwelle trifft {de(100 * alarm['cells_share'], 1)} % der Zellen, aber in {de(100 * alarm['runs_share'], 0)} % der Läufe mindestens eine. **Die Leistung ist kein lineares Maß, "
                       f"und viele Zellen sind keine vielen Messungen.**")
