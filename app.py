"""Kontinuierliche Wavelets: gleiche Zahl von Perioden statt gleicher Fensterlänge – ω₀ entscheidet, was man sieht - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 11 der Reihe „Spektral- und Wavelet-Analyse“. Fünf Testsignale mit exakt bekannter Wahrheit (Chirp, zwei Töne, Frequenzsprung, zwei Impulse, tiefer Ton plus Burst) werden mit dem analytischen
Morlet-Wavelet in ein Skalogramm verwandelt; gezeigt werden die Auflösung gegen die Frequenz (gegen das Spektrogramm), die Rolle von ω₀, der Einflusskegel, das beste ω₀ für einen Chirp und die
Skalierung der Leistung mit der Korrelation der Zellen. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import cwt_constants as C
import cwt_evaluation as ev
from cwt_evaluation import Settings, analyse, de
from cwt_pdf_export import build_pdf
from cwt_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    param_key,
    randomize_seed,
    sync_query_params,
)
from cwt_visualization import (
    build_chirp,
    build_cone,
    build_correlation,
    build_omega,
    build_profile,
    build_resolution_freq,
    build_resolution_time,
    build_scalogram,
    build_scaling,
    build_signal,
    build_timeline,
)

st.set_page_config(page_title="Wavelets – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🌊 Kontinuierliche Wavelets: gleiche Zahl von Perioden statt gleicher Fensterlänge")
st.markdown(
    """
Das Spektrogramm schneidet das Signal in Fenster **fester Länge** $L$: Es trennt Frequenzen überall gleich fein und Zeitpunkte überall gleich fein. Die **Wavelet-Transformation** dehnt dagegen ein Wellenpaket
mit der Skala $s$: Das Morlet-Wavelet $\\psi(t) = \\pi^{-1/4} e^{i\\omega_0 t} e^{-t^2/2}$ hat immer dieselbe Zahl von Perioden ($\\omega_0$), seine Länge wächst mit $s = \\omega_0 / (2\\pi f)$.
Tiefe Frequenzen sehen daher ein langes, hohe ein kurzes Wavelet: Die Frequenzauflösung ist proportional zur Frequenz, die Zeitauflösung umgekehrt. Das Skalogramm ist die Leistung $|W(s, t)|^2$ über Zeit und Frequenz;
der Parameter $\\omega_0$ regelt, wie viel Frequenzschärfe man gegen Zeitschärfe tauscht.

Dies ist die elfte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf **Spektrogramm** und **Momentanfrequenz** aufbauend (das Morlet-Wavelet ist ein analytisches Signal je Skala). Alles ist **simuliert**: Chirp, Töne
und Impulse sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So entsteht das Skalogramm", expanded=True):
    st.markdown(
        """
1. **Wavelet dehnen:** Zu jeder Frequenz $f$ gehört die Skala $s = \\omega_0 / (2\\pi f)$; das Wavelet hat dort $\\omega_0$ Perioden in einer Gaußhülle der Breite $s$.
2. **Falten:** $W(s, n)$ ist die Faltung des Signals mit dem gedehnten Wavelet (hier über die FFT, Nullen am Rand).
3. **Leistung:** Dargestellt wird $|W|^2 / s$ (gleichgewichtet: jede Linie der Amplitude 1 gibt dieselbe Spitze) oder $|W|^2$; in dB gegen das Maximum, bei $-60$ dB abgeschnitten.
4. **Grenzen:** Zwei Töne trennt das Wavelet ab dem Frequenzverhältnis $1 + 3{,}9 / \\omega_0$, zwei Impulse sicher ab etwa $2{,}4\\,s$; am Rand zeigt der **Einflusskegel**, wo die Werte unsicher werden.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Zahl der Abtastwerte (Abtastrate 1, Frequenzen in Zyklen je Abtastwert, Nyquist-Frequenz 0,5).")
    kind = st.selectbox("Signaltyp", list(C.SIGNALS), key="signal_select",
                        help="Chirp: Frequenz steigt linear. Zwei Töne: gleich starke Kosinus. Frequenzsprung: Ton, der springt. Zwei Impulse: zwei Einzelwerte. Tiefer Ton plus Burst: ein dauernder Ton und ein kurzes Wellenpaket.")
    ti = C.SIGNALS.index(kind)
    values = []
    for slot, (label, lo, hi, step, default, help_text) in enumerate(C.PARAMS[kind], start=1):
        v = st.slider(label, lo, hi, float(st.session_state[param_key(ti, slot)]), step=step, format="%.3f" if step < 0.01 else "%.2f", help=help_text)
        st.session_state[param_key(ti, slot)] = float(v)
        values.append(float(v))
    while len(values) < 3:
        values.append(0.0)
    st.markdown("**Wavelet und Gitter**")
    w0 = st.select_slider("ω₀ (Perioden im Wavelet)", options=list(C.W0_CHOICES), key="w0_select",
                          help="Zahl der Perioden unter der Gaußhülle. Mehr: schärfer in der Frequenz, gröber in der Zeit. Üblich: 6. Die Frequenzachse endet bei f_max = ω₀ / (2 (ω₀ + 3,7)), höchstens 0,45.")
    fmin = st.select_slider("kleinste Frequenz", options=list(C.FMIN_CHOICES), key="fmin_select", help="Untere Grenze der Frequenzachse; tiefe Frequenzen brauchen lange Wavelets.")
    voices = st.select_slider("Stimmen je Oktave", options=list(C.VOICES_CHOICES), key="voices_select", help="Dichte des logarithmischen Frequenzgitters.")
    power = st.radio("Leistung", list(C.POWER_MODES), key="power_select", horizontal=True,
                     help="gleichgewichtet: |W|² / s, jede Linie der Amplitude 1 gibt dieselbe Spitze. L2: |W|², tiefe Frequenzen erscheinen um den Faktor s stärker.")
    st.markdown("**Schnitte**")
    tcut = st.slider("Zeitpunkt des Frequenzprofils", C.CUT_MIN, C.CUT_MAX, float(st.session_state["tcut_slider"]), step=C.CUT_STEP, key="tcut_slider", help="Zeit (Anteil der Länge), bei der das Skalenprofil gezeigt wird.")
    fcut = st.slider("Frequenz des Zeitverlaufs", C.FCUT_MIN, C.FCUT_MAX, float(st.session_state["fcut_slider"]), step=C.FCUT_STEP, format="%.3f", key="fcut_slider",
                     help="Frequenz, bei der der Zeitverlauf gezeigt wird; bei zwei Impulsen entscheidet sie, ob sie getrennt erscheinen.")
    st.markdown("**Messung**")
    noise = st.radio("Rauschen", list(C.NOISE_MODES), key="noise_select", horizontal=True, help="Weißes Gaußrauschen zum Signal addiert.")
    if noise == "an":
        snr = st.slider("Rauschabstand (dB)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]), help="Leistung einer Linie der Amplitude 1 (also 1/2) gegen die Rauschleistung.")
        st.session_state["snr_kept"] = int(snr)
    else:
        snr = int(st.session_state["snr_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "signal_select": kind, "w0_select": int(w0), "fmin_select": float(fmin), "voices_select": int(voices), "power_select": power, "tcut_slider": float(tcut),
                   "fcut_slider": float(fcut), "noise_select": noise, "snr_kept": int(snr), "seed_input": int(seed),
                   **{param_key(ti, slot): v for slot, v in enumerate(values[:len(C.PARAMS[kind])], start=1)}})

settings = Settings(int(n), kind, values[0], values[1], values[2], int(w0), float(fmin), int(voices), power, float(tcut), float(fcut), noise, int(snr), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
if a.rmse is not None:
    m1.metric("Fehler des Kamms", de(a.rmse, 4), delta=f"{de(100 * a.interior_share, 0)} % der Zeit außerhalb der Kegel", delta_color="off",
              help="Der Gipfel jedes Zeitpunkts (mit Parabelverfeinerung) gegen die wahre Frequenz, nur Zeitpunkte, die mindestens 3 s vom Rand entfernt sind.")
elif a.resolved is not None:
    what = "Töne getrennt" if kind == "Zwei Töne" else "Impulse getrennt"
    m1.metric(what, "ja" if a.resolved else "nein", delta="Einbruch unter 50 %", delta_color="off", help="Zwischen den Gipfeln liegt ein Einbruch unter die Hälfte des kleineren.")
elif a.both_found is not None:
    m1.metric("Beide Gipfel gefunden", "ja" if a.both_found else "nein", delta="innerhalb 10 % der wahren Frequenz", delta_color="off", help="Skalenprofil zur Burstzeit: die zwei höchsten Gipfel gegen die wahren Frequenzen.")
elif a.step_error is not None:
    m1.metric("Wechsel des Kamms", f"{de(a.step_error, 0)}", delta="Abtastwerte gegen den wahren Sprung", delta_color="off", help="Zeitpunkt, an dem der Kamm zur zweiten Frequenz hinüber ist, minus der wahre Sprungzeitpunkt.")
else:
    m1.metric("Kennzahl", "–", delta="nicht auswertbar", delta_color="off")
m2.metric("Trenngrenze Frequenz", f"Verhältnis {de(a.ratio_limit, 2)}", delta=f"1 + {de(C.FREQ_RULE_K, 1)} / ω₀ (Faustregel)", delta_color="off",
          help="Zwei gleich starke Töne werden ab diesem Verhältnis der Frequenzen getrennt, bei jeder Frequenz gleich (Experiment unten).")
m3.metric("Trenngrenze Zeit", f"{de(a.time_limit, 0)} Abtastwerte", delta=f"{de(C.TIME_RULE_K, 1)} s bei f = {de(settings.fcut, 3)}", delta_color="off",
          help="Zwei Impulse werden bei der Frequenz des Zeitverlaufs ab etwa 2,4 s sicher getrennt; s = ω₀ / (2π f).")
m4.metric("Zellen", f"{a.cells}", delta=f"gültig bis f = {de(a.fmax, 2)}", delta_color="off", help="Frequenzen mal Zeitpunkte; benachbarte Zellen sind stark korreliert. Das abgetastete Wavelet ist nur bis f_max unverfälscht.")
st.plotly_chart(build_signal(a), width="stretch", key="signal")
st.plotly_chart(build_scalogram(a), width="stretch", key="scalogram")
st.caption("Leistung in dB gegen das Maximum (bei −60 dB abgeschnitten), logarithmische Frequenzachse. Weiß gestrichelt: der Einflusskegel (√2 s vom Rand); rot gestrichelt: die wahre Frequenz bzw. die Impulse; grün: der Gipfel je "
           "Zeitpunkt; orange: die beiden Schnitte.")
st.plotly_chart(build_profile(a), width="stretch", key="profile")
st.caption("Skalenprofil beim orangen Zeitpunkt. Die Breite eines Gipfels ist proportional zu seiner Frequenz (etwa 1/ω₀ relativ): bei höheren Frequenzen sind die Gipfel absolut breiter.")
if kind in ("Zwei Impulse", "Tiefer Ton plus Burst"):
    st.plotly_chart(build_timeline(a), width="stretch", key="timeline")
    st.caption("Zeitverlauf der Leistung bei der orangen Frequenz. Je höher die Frequenz, desto kürzer das Wavelet und desto feiner trennt es zwei Impulse.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="skalogramm.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viele Perioden braucht das Wavelet?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Feste Zufallsströme, unabhängig von den Reglern; analytisches Morlet-Wavelet, Leistung |W|²/s. Abtastrate 1, Frequenzen in Zyklen je Abtastwert. Rauschen mit Varianz 0,5/10^(SNR/10), also gegen eine Linie "
           "der Amplitude 1.")

st.markdown("#### Auflösung gegen Frequenz: Wavelet und Spektrogramm")
res = _experiment("resolution")
st.caption(f"Zwei gleich starke Töne bei f und f·Verhältnis ({C.EXP_RES_RUNS} Läufe mit zufälligen Phasen, Raster 0,05; „getrennt“ heißt: Einbruch unter 50 %) und zwei Einzelimpulse bei der Analysefrequenz f: "
           "„sicher“ heißt, dass auch alle größeren Abstände bis 6 s getrennt sind (die erste getrennte Stelle ist wegen der Schwebung des Wavelets oft ein Scheineinbruch).")
st.dataframe(pd.DataFrame([{"f": de(r["f"], 3), "s": de(r["s"], 1), "Wavelet: Verhältnis": de(r["ratio"], 2), "Spektrogramm: Verhältnis": de(r["stft_ratio"], 2), "Wavelet: Impulse sicher": r["robust"],
                            "Wavelet: erste Stelle": r["first"], "Spektrogramm: Impulse": res["stft_distance"]} for r in res["rows"]]), hide_index=True, width="stretch")
st.plotly_chart(build_resolution_freq(res), width="stretch", key="exp_res_freq")
st.plotly_chart(build_resolution_time(res), width="stretch", key="exp_res_time")
kind_v, text = ev.verdict_resolution(res)
getattr(st, kind_v)(text)

st.markdown("#### Der Parameter ω₀")
omega = _experiment("omega")
st.caption(f"Bei f = {de(C.EXP_OMEGA_F, 1)}: Frequenzverhältnis, ab dem zwei Töne getrennt sind, und sicherer Impulsabstand für verschiedene ω₀ (dieselben Kriterien).")
st.dataframe(pd.DataFrame([{"ω₀": r["w0"], "s": de(r["s"], 1), "Verhältnis": de(r["ratio"], 2), "(Verhältnis − 1) · ω₀": de((r["ratio"] - 1) * r["w0"], 1), "Impulse sicher": r["robust"],
                            "Impulse / s": de(r["robust"] / r["s"], 2)} for r in omega]), hide_index=True, width="stretch")
st.plotly_chart(build_omega(omega), width="stretch", key="exp_omega")
kind_v, text = ev.verdict_omega(omega)
getattr(st, kind_v)(text)

st.markdown("#### Einflusskegel am Rand")
cone = _experiment("cone")
st.caption(f"Ton der Amplitude 1, Nullen am Rand angehängt, n = {C.EXP_CONE_N}: Amplitude |W| im Abstand d vom Rand gegen die Amplitude im Inneren, in Vielfachen von s, gegen die Theorie Φ(d/s) der Normalverteilung.")
st.dataframe(pd.DataFrame([{"ω₀": r["w0"], "f": de(r["f"], 2), "s": de(r["s"], 1), "d = 0": de(r["r0"], 3), "d = s": de(r["r1"], 3), "d = √2 s": de(r["r_coi"], 3), "d = 2 s": de(r["r2"], 3),
                            "0,90 ab (s)": de(r["d90"] / r["s"], 2), "0,99 ab (s)": de(r["d99"] / r["s"], 2)} for r in cone["rows"]]
                          + [{"ω₀": "Theorie", "f": "", "s": "", "d = 0": de(cone["theory"]["r0"], 3), "d = s": de(cone["theory"]["r1"], 3), "d = √2 s": de(cone["theory"]["r_coi"], 3), "d = 2 s": de(cone["theory"]["r2"], 3),
                              "0,90 ab (s)": de(cone["theory"]["d90"], 2), "0,99 ab (s)": de(cone["theory"]["d99"], 2)}]), hide_index=True, width="stretch")
st.plotly_chart(build_cone(cone), width="stretch", key="exp_cone")
kind_v, text = ev.verdict_cone(cone)
getattr(st, kind_v)(text)

st.markdown("#### Chirp in Rauschen: das beste ω₀")
chirp = _experiment("chirp")
st.caption(f"Linearer Chirp von {de(C.EXP_CHIRP_F0, 1)} bis {de(C.EXP_CHIRP_F1, 1)}, n = {C.EXP_CHIRP_N}, {C.EXP_CHIRP_RUNS} Läufe je Zeile, {C.EXP_CHIRP_VOICES} Stimmen je Oktave; RMSE des Kamms gegen die wahre Frequenz, nur "
           "Zeitpunkte mit mindestens 3 s Abstand zum Rand. ω₀ ≥ 6, damit der Chirp im gültigen Frequenzbereich liegt.")
st.dataframe(pd.DataFrame([{"ω₀": w, **{f"{snr_} dB": de(r["rmse"], 4) for snr_ in C.EXP_CHIRP_SNR for r in chirp if r["w0"] == w and r["snr"] == snr_}} for w in C.EXP_CHIRP_W0]),
             hide_index=True, width="stretch")
st.plotly_chart(build_chirp(chirp), width="stretch", key="exp_chirp")
kind_v, text = ev.verdict_chirp(chirp)
getattr(st, kind_v)(text)

st.markdown("#### Skalierung der Leistung und Korrelation der Zellen")
scaling = _experiment("scaling")
st.caption(f"Gleich starke Töne (Amplitude 1, ω₀ = {scaling['w0']:g}): Spitze von |W|² gegen die Spitze bei der höchsten Frequenz; weißes Rauschen (Varianz 1, {C.EXP_SCALE_RUNS} Läufe): mittleres |W|² und "
           "Variationskoeffizient; Korrelation der Zellleistungen bei f = 0,1 gegen die Theorie; punktweise 5-%-Schwelle (Exp(1)) bei weißem Rauschen.")
st.dataframe(pd.DataFrame([{"f": de(r["f"], 3), "s": de(r["s"], 1), "Spitze |W|²": de(r["peak_l2"], 2), "relativ": de(r["rel_l2"], 1), "Spitze |W|²/s": de(r["peak_rect"], 4),
                            "Rauschen: E|W|²": de(nr["mean"], 3), "Rauschen: CV": de(nr["cv"], 3)} for r, nr in zip(scaling["tones"], scaling["noise"])]), hide_index=True, width="stretch")
st.plotly_chart(build_scaling(scaling), width="stretch", key="exp_scaling")
st.dataframe(pd.DataFrame([{"Zeitabstand (s)": de(r["lag_s"], 1), "Korrelation": de(r["corr"], 3), "Theorie exp(−k²/2)": de(r["theory"], 3)} for r in scaling["time_corr"]]), hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"relativer Frequenzunterschied": de(r["rel"], 2), "Korrelation": de(r["corr"], 3), "Theorie": de(r["theory"], 3)} for r in scaling["freq_corr"]]), hide_index=True, width="stretch")
st.plotly_chart(build_correlation(scaling), width="stretch", key="exp_corr")
al = scaling["alarm"]
st.caption(f"Falschalarm der punktweisen 5-%-Schwelle: {de(100 * al['cells_share'], 1)} % der Zellen, {de(100 * al['runs_share'], 0)} % der {al['runs']} Läufe mit mindestens einer Zelle darüber ({al['freqs']} Frequenzen).")
kind_v, text = ev.verdict_scaling(scaling)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein ω₀ passt für alles** | ω₀ steuert die Auflösung für alle Frequenzen zugleich: Mehr Perioden schärfen die Frequenz und verschmieren die Zeit. Das beste ω₀ hängt von der Steigung und vom Rauschen ab. | Anpassung von ω₀ je Signal, Reassignment |
| **Hohe Frequenzen sind fein aufgelöst** | Die absolute Frequenzauflösung ist proportional zu f: Oberhalb von etwa 0,06 trennt ein Hann-Spektrum der Länge 64 Töne feiner als das Wavelet (ω₀ = 6). | Spektrogramm mit kurzem Fenster |
| **Die Werte am Rand stimmen** | Im Einflusskegel fehlt ein Teil des Wavelets; die Amplitude erreicht 99 % erst bei 2,3 s Abstand vom Rand. | Randwerte verwerfen, Spiegeln (nicht gebaut) |
| **Das abgetastete Wavelet ist unverfälscht** | Oberhalb von f_max (bei ω₀ = 6: 0,31) weicht die FFT-Transformation von der Summendefinition ab (bei f = 0,4 und ω₀ = 6 um mehr als 5 %); die Frequenzachse endet bei f_max. | größeres ω₀, Überabtastung |
| **Die Leistung ist ein Maß für die Stärke** | \\|W\\|² wächst mit s; erst \\|W\\|²/s macht gleich starke Linien gleich hoch. Das Skalogramm ist kein lineares Leistungsspektrum. | Gleichgewichtung |
| **Zellen sind unabhängig** | Benachbarte Zellen sind stark korreliert (0,58 bei Abstand s, 0,85 bei 10 % Frequenzunterschied); eine Schwelle pro Zelle trifft irgendwo fast immer. | Mehrfachtests, Wavelet-Kohärenz |
| **Nur ein Wavelet** | Hier nur das analytische Morlet-Wavelet; reelle Wavelets (Mexikanerhut) und andere Familien verhalten sich anders. | nicht gebaut |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Morlet-Wavelet.** $\psi(t) = \pi^{-1/4} e^{i\omega_0 t} e^{-t^2/2}$ (ohne Korrekturterm, für $\omega_0 \ge 5$ vernachlässigbar). **Transformation.** $W(s, n) = \frac{1}{\sqrt s} \sum_m x[m]\, \overline{\psi}\!\left(\frac{m - n}{s}\right)$,
umgesetzt über die FFT mit $\hat\psi(s\omega) = \pi^{-1/4} \sqrt{2\pi s}\, e^{-(s\omega - \omega_0)^2/2}$ für $\omega > 0$ (null sonst).

**Skala und Frequenz.** $f = \omega_0 / (2\pi s)$. Für einen Kosinus der Amplitude $A$ ist $|W| = \tfrac A2 \pi^{-1/4}\sqrt{2\pi s}\, e^{-(s\omega - \omega_0)^2/2}$, also $|W|^2 \propto s$ bei gleicher Amplitude; $|W|^2/s$ ist bei $s\omega = \omega_0$ exakt
in der Spitze.

**Auflösung.** Die Gaußkurve im Frequenzbereich hat die relative Breite $1/\omega_0$: Zwei Töne trennt das Wavelet bei einem festen Frequenzverhältnis (gemessen etwa $1 + 3{,}9/\omega_0$), unabhängig von $f$. Im Zeitbereich ist die Hülle $e^{-t^2/(2s^2)}$
breit wie $s$; zwei Impulse sind sicher ab etwa $2{,}4\,s$ getrennt.

**Einflusskegel.** Bei Nullen am Rand ist die Amplitude eines Tons im Abstand $d$ vom Rand $\Phi(d/s)$ mal der Innenwert ($\Phi$ die Verteilungsfunktion der Normalverteilung); die e-Falte $\sqrt 2\, s$ lässt 8 % der Amplitude fehlen.

**Rauschen.** Für weißes Rauschen der Varianz $\sigma^2$ ist $E|W|^2 = \sigma^2$ bei allen Skalen (L2-Normierung) und $|W|^2/\sigma^2 \sim \mathrm{Exp}(1)$. Die Korrelation der Leistung im Zeitabstand $\tau$ ist $e^{-\tau^2/(2 s^2)}$, bei den
Frequenzen $f$ und $f q$: $\frac{2}{q + 1/q}\, e^{-\omega_0^2 (q-1)^2/(q^2 + 1)}$.

**Quellen.** Lehrbuchwissen und Aufsätze (Morlet, Wavelet-Transformation, 1980er Jahre; Torrence und Compo 1998; Liu, Liang und Weisberg 2007). Formeln hier per Simulation, Handrechnung und gegen die Summendefinition geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
