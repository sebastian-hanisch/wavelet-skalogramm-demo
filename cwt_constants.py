"""Konstanten: Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Farben und Presets (Kern: cwt_morlet.py, cwt_signals.py, cwt_resolution.py)."""

N_CHOICES = (256, 512, 1024, 2048)
DEFAULT_N = 1024
SIGNALS = ("Chirp", "Zwei Töne", "Frequenzsprung", "Zwei Impulse", "Tiefer Ton plus Burst")
DEFAULT_SIGNAL = "Chirp"
# Je Signaltyp bis zu drei Parameter: (Beschriftung, Minimum, Maximum, Schritt, Standardwert, Hilfetext)
PARAMS = {
    "Chirp": (("Startfrequenz", 0.02, 0.45, 0.005, 0.1, "Frequenz am Anfang (Zyklen je Abtastwert)."), ("Endfrequenz", 0.02, 0.45, 0.005, 0.3, "Frequenz am Ende; dazwischen steigt sie linear.")),
    "Zwei Töne": (("Frequenz f1", 0.02, 0.3, 0.005, 0.1, "Frequenz des tieferen Tons."), ("Verhältnis f2 / f1", 1.05, 3.0, 0.05, 1.5, "Frequenz des zweiten Tons als Vielfaches der ersten; beide haben Amplitude 1.")),
    "Frequenzsprung": (("Frequenz 1", 0.02, 0.45, 0.005, 0.1, "Frequenz vor dem Sprung."), ("Frequenz 2", 0.02, 0.45, 0.005, 0.2, "Frequenz nach dem Sprung."),
                       ("Sprungzeitpunkt (Anteil der Länge)", 0.1, 0.9, 0.05, 0.5, "Zeitpunkt des Sprungs als Anteil der Aufzeichnung.")),
    "Zwei Impulse": (("Abstand dt (Abtastwerte)", 2.0, 400.0, 1.0, 20.0, "Abstand der beiden Einzelimpulse."), ("Lage des ersten (Anteil der Länge)", 0.1, 0.8, 0.05, 0.5, "Zeitpunkt des ersten Impulses.")),
    "Tiefer Ton plus Burst": (("Frequenz des Tons", 0.01, 0.1, 0.005, 0.04, "Frequenz des dauernden tiefen Tons."), ("Frequenz des Bursts", 0.1, 0.45, 0.005, 0.25, "Frequenz des kurzen Bursts (Gaußhülle von 1,5 Perioden)."),
                              ("Zeitpunkt des Bursts (Anteil der Länge)", 0.1, 0.9, 0.05, 0.5, "Mitte des Bursts als Anteil der Aufzeichnung.")),
}
W0_CHOICES = (3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64)
DEFAULT_W0 = 6
FMIN_CHOICES = (0.01, 0.02, 0.03, 0.05)
DEFAULT_FMIN = 0.02
VOICES_CHOICES = (8, 16, 32, 64)
DEFAULT_VOICES = 16
POWER_MODES = ("gleichgewichtet", "L2")
DEFAULT_POWER = "gleichgewichtet"
CUT_MIN, CUT_MAX, CUT_STEP, DEFAULT_TCUT = 0.0, 1.0, 0.05, 0.5                  # Zeit des Frequenzprofils als Anteil der Länge
FCUT_MIN, FCUT_MAX, FCUT_STEP, DEFAULT_FCUT = 0.02, 0.45, 0.005, 0.1            # Frequenz des Zeitverlaufs
NOISE_MODES = ("aus", "an")
DEFAULT_NOISE = "aus"
SNR_MIN, SNR_MAX, DEFAULT_SNR = -10, 40, 10                                    # Linie der Amplitude 1 (Leistung 1/2) gegen die Rauschleistung, in dB
SEED_MAX, DEFAULT_SEED = 999999, 39
DB_FLOOR = -60.0                                                               # Darstellung des Skalogramms in dB gegen das Maximum
FREQ_RULE_K = 3.9                                                              # Faustregel: Frequenzverhältnis, ab dem zwei Töne getrennt sind, etwa 1 + K / w0 (Experiment: K zwischen 3,2 und 4,8)
TIME_RULE_K = 2.4                                                              # Faustregel: zwei Impulse sicher getrennt ab etwa K mal s (Experiment: 2,2 bis 2,8)
EDGE_S = 3.0                                                                   # Zeitpunkte näher als EDGE_S mal s am Rand zählen für den Kammfehler nicht
OK_RMSE = 0.005                                                                # Kammfehler, unter dem die Schätzung „genau“ heißt
BURST_TOLERANCE = 0.1                                                          # relative Abweichung, innerhalb der ein Gipfel als gefunden gilt
FALSE_ALARM_LEVEL = 0.05
TONE_PHASES = (0.3, 1.1)                                                       # Phasen der beiden Töne in der App

# Experimente (feste Ströme; Rauschvarianz 0,5 / 10^(SNR/10) gegen eine Linie der Amplitude 1)
EXP_SEED = 950000
EXP_RES_SEED = 500
EXP_RES_F = (0.025, 0.05, 0.1, 0.2)
EXP_RES_W0 = 6
EXP_RES_RUNS = 60
EXP_RES_STFT_L = 64
EXP_OMEGA_W0 = (3, 4, 5, 6, 8, 10, 12, 16)
EXP_OMEGA_F = 0.1
EXP_CONE_N = 1024
EXP_CONE_COMBOS = ((6, 0.1), (6, 0.2), (3, 0.1), (12, 0.1))
EXP_CONE_PHASE = 0.3
EXP_CHIRP_N, EXP_CHIRP_F0, EXP_CHIRP_F1, EXP_CHIRP_RUNS, EXP_CHIRP_VOICES = 1024, 0.1, 0.3, 10, 48
EXP_CHIRP_W0 = (6, 8, 12, 16, 24, 32, 48, 64)
EXP_CHIRP_SNR = (60, 10, 0)
EXP_CHIRP_FMIN, EXP_CHIRP_FMAX = 0.05, 0.45
EXP_SCALE_F = (0.3, 0.2, 0.1, 0.05, 0.025)
EXP_SCALE_N, EXP_SCALE_NOISE_N, EXP_SCALE_RUNS = 2048, 512, 400
EXP_SCALE_CORR_F = 0.1
EXP_SCALE_LAGS = (0.5, 1.0, 2.0, 3.0)
EXP_SCALE_REL = (0.05, 0.1, 0.2, 0.3)
EXP_SCALE_FA_RUNS, EXP_SCALE_FA_FMIN, EXP_SCALE_FA_VOICES = 200, 0.03, 8

COLOR_TRUE = "#B03A2E"
COLOR_RIDGE = "#4C9F70"
COLOR_SIGNAL = "rgba(120,120,120,0.9)"
COLOR_EST = "rgba(76,120,168,0.95)"
COLOR_AUX = "#E8833A"

PRESETS = {
    "Chirp mit passendem ω₀": {"n": 1024, "signal": "Chirp", "params": (0.1, 0.3, 0.0), "w0": 24, "fmin": 0.05, "voices": 16, "power": "gleichgewichtet", "tcut": 0.5, "fcut": 0.1, "noise": "an", "snr": 0,
                               "seed": DEFAULT_SEED},
    "Zu kleines ω₀: die Frequenz verschwimmt": {"n": 1024, "signal": "Chirp", "params": (0.1, 0.3, 0.0), "w0": 6, "fmin": 0.05, "voices": 16, "power": "gleichgewichtet", "tcut": 0.5, "fcut": 0.1, "noise": "an",
                                                "snr": 0, "seed": DEFAULT_SEED},
    "Zu großes ω₀: der Chirp verschmiert": {"n": 1024, "signal": "Chirp", "params": (0.1, 0.3, 0.0), "w0": 64, "fmin": 0.05, "voices": 16, "power": "gleichgewichtet", "tcut": 0.5, "fcut": 0.1, "noise": "an",
                                            "snr": 0, "seed": DEFAULT_SEED},
    "Tiefer Ton plus Burst: Auflösung folgt der Frequenz": {"n": 1024, "signal": "Tiefer Ton plus Burst", "params": (0.04, 0.25, 0.5), "w0": 6, "fmin": 0.02, "voices": 16, "power": "gleichgewichtet", "tcut": 0.5,
                                                               "fcut": 0.1, "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
    "Zwei Impulse: tief gröber, hoch feiner": {"n": 1024, "signal": "Zwei Impulse", "params": (20.0, 0.5, 0.0), "w0": 6, "fmin": 0.02, "voices": 16, "power": "gleichgewichtet", "tcut": 0.5, "fcut": 0.1,
                                               "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Chirp mit passendem ω₀": "Linearer Chirp von 0,1 bis 0,3 bei 0 dB Rauschabstand und ω₀ = 24: Der Kamm folgt der wahren Frequenz auf etwa 0,001.",
    "Zu kleines ω₀: die Frequenz verschwimmt": "Dieselbe Messung mit ω₀ = 6, dem üblichen Wert: Das Wavelet hat nur wenige Perioden, die Frequenzauflösung ist grob und das Rauschen stört; der Kammfehler liegt mehr als das Fünfzehnfache über dem des ersten Beispiels.",
    "Zu großes ω₀: der Chirp verschmiert": "Dieselbe Messung mit ω₀ = 64: Das Wavelet ist so lang (s = 102 bei f = 0,1), dass sich die Frequenz innerhalb seiner Länge mehrfach um die Bandbreite ändert; der Gipfel wird breit, der Kammfehler steigt wieder auf das Doppelte, und nur noch 68 % der Zeit liegen außerhalb der Einflusskegel.",
    "Tiefer Ton plus Burst: Auflösung folgt der Frequenz": "Ein tiefer Ton bei 0,04 und ein kurzer Burst bei 0,25: Das Wavelet ist bei hoher Frequenz kurz und bei tiefer lang (tiefer Ton fein in der Frequenz, Burst fein in der Zeit); zur Burstzeit zeigt das Profil beide Gipfel (bei 0,040 und 0,253).",
    "Zwei Impulse: tief gröber, hoch feiner": "Zwei Impulse im Abstand 20: Bei der Frequenz 0,1 (s = 9,5) verschwimmen sie zu einem Gipfel; der Zeitverlauf bei 0,3 trennt sie. Wählen Sie unter „Frequenz des Zeitverlaufs“ den Wert 0,3.",
}
