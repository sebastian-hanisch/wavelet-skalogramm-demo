"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import cwt_constants as C
from cwt_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "β": "beta", "π": "pi", "≥": ">=",
            "ω₀": "w0", "ω": "w", "₀": "0", "Φ": "Phi"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Skalogramm: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    params = ", ".join(f"{label} {de(v, 3)}" for (label, *_rest), v in zip(C.PARAMS[s.signal], (s.p1, s.p2, s.p3)))
    noise = f"Rauschen {s.snr} dB gegen eine Linie der Amplitude 1" if s.has_noise else "ohne Rauschen"
    pdf.multi_cell(0, 5, _s(f"{s.signal} ({params}), n = {s.n}, Morlet-Wavelet mit ω₀ = {s.w0}, Frequenzen {de(a.freqs[0], 3)} bis {de(a.freqs[-1], 3)} ({s.voices} Stimmen je Oktave), Leistung {s.power}, {noise}, "
                      f"Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [f"Skalogramm: {len(a.freqs)} Frequenzen, {a.cells} Zellen; gültiger Bereich bis f = {de(a.fmax, 3)}.",
           f"Faustregeln: zwei Töne ab dem Frequenzverhältnis {de(a.ratio_limit, 2)} (1 + {de(C.FREQ_RULE_K, 1)} / ω₀), zwei Impulse bei f = {de(s.fcut, 3)} ab etwa {de(a.time_limit, 0)} Abtastwerten ({de(C.TIME_RULE_K, 1)} s)."]
    if a.rmse is not None:
        out.append(f"Kammfehler (RMSE gegen die wahre Frequenz, mindestens 3 s vom Rand, {de(100 * a.interior_share, 0)} % der Zeit): {de(a.rmse, 4)}.")
    if a.step_error is not None:
        out.append(f"Wechsel des Kamms zur zweiten Frequenz: {de(a.step_error, 0)} Abtastwerte gegen den wahren Sprung.")
    if s.signal == "Zwei Töne" and a.resolved is not None:
        out.append("Die beiden Töne sind im Skalenprofil " + ("getrennt." if a.resolved else "nicht getrennt."))
    if s.signal == "Zwei Impulse" and a.resolved is not None:
        out.append("Die beiden Impulse sind im Zeitverlauf " + ("getrennt." if a.resolved else "nicht getrennt."))
    if a.burst_peaks is not None:
        out.append("Gipfel im Skalenprofil zur Burstzeit: " + ", ".join(de(f, 3) for f in a.burst_peaks) + (" (beide gefunden)." if a.both_found else " (nicht beide gefunden)."))
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: wavelet-skalogramm-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
