# Kontinuierliche Wavelets: gleiche Zahl von Perioden statt gleicher Fensterlänge – ω₀ entscheidet, was man sieht – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-wavelet-skalogramm-demo.streamlit.app/)** -->

Elfte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos zum **Spektrogramm** und zur
**Momentanfrequenz**. Fünf Testsignale mit exakt bekannter Wahrheit (Chirp, zwei Töne, Frequenzsprung, zwei Impulse, tiefer Ton plus Burst) werden mit dem **analytischen Morlet-Wavelet** in ein **Skalogramm** verwandelt.
Gemessen werden die Auflösung in Frequenz und Zeit gegen die Frequenz (mit dem Spektrogramm als Vergleich), die Rolle von ω₀, der Einflusskegel am Rand, das beste ω₀ für einen Chirp in Rauschen sowie die Skalierung der
Leistung und die Korrelation der Zellen. Die Demo liefert das redundante Skalogramm, das die diskreten Wavelets danach ausdünnen.

## Warum dieses Problem

Das Spektrogramm schneidet das Signal in Fenster fester Länge: Es trennt Frequenzen überall gleich fein und Zeitpunkte überall gleich fein. Die Wavelet-Transformation dehnt dagegen ein Wellenpaket mit der Skala s; das
Morlet-Wavelet hat immer dieselbe Zahl von Perioden (ω₀), seine Länge wächst mit s = ω₀/(2π f). Damit verteilt es die Unschärfe anders, schärfer verschiebt es sie nicht. Die Demo misst, wo die Grenzen liegen, was
ω₀ daran ändert und wo das Skalogramm trügt (Rand, Leistungsmaß, korrelierte Zellen).

## Modell

Testsignale auf den Zeitpunkten t = 0 … n − 1 (Abtastrate 1, Frequenzen in Zyklen je Abtastwert, Nyquist-Frequenz 0,5): linearer Chirp, zwei gleich starke Kosinus, ein Ton, der von f1 auf f2 springt, zwei Einzelimpulse, ein
tiefer Ton plus ein kurzer Burst (Gaußhülle von 1,5 Perioden). Analytisches Morlet-Wavelet ψ(t) = π^(−1/4) e^{iω₀t} e^{−t²/2} ohne Korrekturterm, L2-Normierung, Transformation per FFT (nur positive Frequenzen, Nullen
am Rand angehängt), logarithmisches Frequenzgitter mit 8 bis 64 Stimmen je Oktave bis zur Obergrenze f_max = ω₀/(2(ω₀ + 3,7)) (höchstens 0,45), oberhalb der das abgetastete Wavelet verfälscht ist. Dargestellt wird
|W|²/s (gleichgewichtet) oder |W|² (L2) in dB. Alles selbst in numpy; scipy dient nur als Gegenprobe in den Tests (die Summendefinition ist die eigentliche Gegenprobe).

## Methodik

n = 256 bis 2048, ω₀ = 3 bis 64. Das Rauschen hat die Varianz 0,5/10^(SNR/10), also gegen eine Linie der Amplitude 1. „Getrennt“ heißt: Der Einbruch zwischen den Gipfeln liegt unter der Hälfte des kleineren; zwei Töne
werden bei zufälligen Phasen in 60 Läufen ausgewertet und gelten als getrennt, wenn es in allen Läufen so ist (Raster 0,05 beim Frequenzverhältnis); zwei Impulse sind „sicher“ getrennt, wenn auch alle größeren Abstände bis
6 s getrennt sind. Experimente über feste Zufallsströme (Seed 950000): 10 Läufe je Zeile beim Chirp, 400 bei der Korrelation, 200 beim Falschalarm; die Zwei-Töne-Messung nutzt die geschlossene Formel im Inneren (in den Tests
Lauf für Lauf gegen die FFT-Transformation geprüft).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie hängt die Auflösung von der Frequenz ab (ω₀ = 6, Spektrogramm Hann mit L = 64 zum Vergleich)? | Zwei gleich starke Töne trennt das Wavelet bei f = 0,025 / 0,05 / 0,1 / 0,2 immer ab dem Frequenzverhältnis 1,65, das Spektrogramm ab 2,55 / 1,8 / 1,4 / 1,2 (konstante absolute Grenze 0,039). Zwei Impulse trennt das Wavelet sicher ab 90 / 45 / 23 / 12 Abtastwerten (etwa das 2,4-Fache von s), das Spektrogramm ab 33 bei jeder Frequenz. **Unterhalb von f ≈ 0,06 ist das Wavelet in der Frequenz feiner und in der Zeit gröber, oberhalb umgekehrt.** |
| Was ändert ω₀ (f = 0,1)? | ω₀ = 3 / 4 / 5 / 6 / 8 / 10 / 12 / 16: Frequenzverhältnis 2,6 / 2,05 / 1,8 / 1,65 / 1,45 / 1,35 / 1,3 / 1,2, sicherer Impulsabstand 13 / 14 / 22 / 23 / 33 / 42 / 52 / 72 Abtastwerte. (Verhältnis − 1)·ω₀ liegt zwischen 3,2 und 4,8, der Impulsabstand zwischen 2,2 und 2,8 s: mehr Perioden schärfen die Frequenz und verschmieren die Zeit. |
| Wie wirkt der Rand (Ton der Amplitude 1, Nullen angehängt)? | Die Amplitude erreicht am Rand 0,52 des Innenwerts, bei s 0,86, bei √2 s (dem üblichen Einflusskegel) 0,93 und bei 2 s 0,98, in Vielfachen von s für ω₀ = 3 / 6 / 12 und f = 0,1 / 0,2 gleich und gleich der Verteilungsfunktion Φ(d/s) der Normalverteilung (0,50 / 0,84 / 0,92 / 0,98). 90 % erreicht sie ab 1,26 s, 99 % ab 2,30 s Abstand. |
| Welches ω₀ verfolgt einen Chirp (0,1 bis 0,3, n = 1024) am besten? | Kammfehler (RMSE, mindestens 3 s vom Rand) bei 0 dB für ω₀ = 6 / 8 / 12 / 16 / 24 / 32 / 48 / 64: 0,0495 / 0,0368 / 0,0155 / 0,0023 / 0,0010 / 0,0011 / 0,0020 / 0,0031; bei 10 dB: 0,0021 / 0,0014 / 0,0007 / 0,0005 / 0,0003 / 0,0004 / 0,0007 / 0,0013. Das beste ω₀ ist 24 bei 0 und 10 dB, weit über dem üblichen 6; bei 60 dB steigt der Fehler von ω₀ = 6 an, nur noch die Steigung stört (0,00057 bei ω₀ = 64). |
| Ist die Skalogramm-Leistung ein lineares Maß? | Nein: Gleich starke Töne (Amplitude 1) zeigen \|W\|² von 2,84 / 4,26 / 8,52 / 17,04 / 34,08 bei f = 0,3 / 0,2 / 0,1 / 0,05 / 0,025 (proportional zu s, das 12-Fache), mit \|W\|²/s überall 0,886. Weißes Rauschen (Varianz 1) ist flach (E\|W\|² 1,00 bis 1,02) und hat den Variationskoeffizienten 1,0 wie ein χ²₂. |
| Wie stark sind benachbarte Zellen korreliert (weißes Rauschen, f = 0,1)? | Im Zeitabstand 0,5 / 1 / 2 / 3 s: 0,874 / 0,585 / 0,144 / 0,010 (Theorie exp(−k²/2): 0,872 / 0,578 / 0,138 / 0,010); bei einem relativen Frequenzunterschied von 5 / 10 / 20 / 30 %: 0,959 / 0,851 / 0,548 / 0,290 (Theorie 0,957 / 0,846 / 0,545 / 0,290). |
| Was bringt eine punktweise 5-%-Schwelle? | Sie trifft 4,9 % der Zellen, aber in 100 % der 200 Läufe mindestens eine Zelle (28 Frequenzen, n = 512): Viele Zellen sind keine vielen Messungen. |

## Ehrliche Grenzen

- **Heimvorteil:** Alle Testsignale sind exakt bekannte Modelle; echte Signale haben unbekannte, oft mehrere Komponenten.
- **Kriterium und Raster:** Die Auflösungsgrenzen gelten für das Einbruchskriterium und die Schrittweite 0,05 des Frequenzverhältnisses; das Verhältnis 1,65 ist ein Raster, kein Wert auf drei Stellen. Die erste getrennte Stelle bei Impulsen
  ist wegen der Schwebung des Wavelets oft ein Scheineinbruch (19 Abtastwerte bei f = 0,025 gegen 90 sicher).
- **Faustregeln:** Die Metriken der App (1 + 3,9/ω₀ für Töne, 2,4 s für Impulse) sind Mittelwerte über die gemessenen ω₀ und Frequenzen, keine Formeln.
- **Gültiger Frequenzbereich:** Das abgetastete Wavelet ist nur bis f_max unverfälscht; darüber weicht die FFT-Transformation von der Summendefinition ab (bei ω₀ = 6 und f = 0,4 um mehr als 5 %). Die Chirp-Messung nimmt ω₀ ≥ 6, weil der Chirp sonst über f_max hinaus reicht.
- **Beste ω₀ gilt für Steigung und Rauschen des Beispiels** (Chirp 0,1 bis 0,3, n = 1024); keine allgemeine Formel. Der Kammfehler ist zudem nur außerhalb der Einflusskegel gemessen.
- **Nur Morlet:** Ein analytisches Wavelet ohne Korrekturterm (bei ω₀ < 5 nicht exakt zulässig); reelle Wavelets (Mexikanerhut), Paul und andere Familien sind nicht gebaut. Keine Signifikanzlinien (rotes Rauschen), keine Umkehrtransformation.
- **Quellen** (Morlet, Wavelet-Transformation, 1980er Jahre; Torrence und Compo 1998; Liu, Liang und Weisberg 2007): nicht gelesen; Formeln per Simulation, Handrechnung und gegen die Summendefinition geprüft.

## Verifikation

- **Kern gegen Handrechnung:** Impuls bei 4 (n = 9), s = 2, ω₀ = 6: |W| = π^(−1/4)/√2 · exp(−(n − 4)²/8) = 0,0719 / 0,1724 / 0,3221 / 0,4687 / 0,5311 / …; Ton: |W| im Inneren = (A/2) π^(−1/4) √(2πs) exp(−(sω − ω₀)²/2) (2,909097 bei f₀ = 0,1).
- **Orakel (`tests/test_oracle_*.py`):** FFT-Transformation gegen die Summendefinition (1e-8 relativ bei f ≤ 0,2 und ω₀ = 6, Abweichung oberhalb f_max nachgewiesen), geschlossene Formel der Zwei-Töne-Messung Lauf für Lauf
  gegen die FFT, Einflusskegel gegen Φ(d/s), Korrelation gegen exp(−k²/2) und die Frequenzformel, Rauschen E|W|² = σ², Parabelverfeinerung exakt für eine Log-Parabel.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Auswahlfelder, Permalink, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich; hier nur der Zufallsstrom SplitMix64) |
| `cwt_morlet.py` | Morlet-Wavelet, FFT-Transformation, Summendefinition, Gitter, Kamm, Gipfel, Einflusskegel |
| `cwt_signals.py` | die fünf Testsignale mit exakter Wahrheit (Spuren als picklebare Klassen) |
| `cwt_resolution.py` | Auflösung zweier Töne und zweier Impulse, Spektrogramm zum Vergleich, Korrelation |
| `cwt_constants.py` | Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Presets |
| `cwt_evaluation.py` | Analyse, Experimente, Verdikte |
| `cwt_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `cwt_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `cwt_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Signale, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Diskrete Wavelets, Mehrskalenanalyse, Schrumpfung:** Thema der Demos danach.
- **Reassignment und Synchrosqueezing:** schärfen das Skalogramm anhand der Phasenableitung; nicht enthalten.
- **Wavelet-Kohärenz und Signifikanzlinien:** nicht enthalten.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf wenige Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
