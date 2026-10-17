"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Die Parameter der Signaltypen stehen je Typ und Platz unter `p<Typ><Platz>_kept` (kein Frontend/Backend-Auseinanderlaufen: der Regler bekommt den Wert als Startwert); die Adresszeile
führt sie als `q<Typ><Platz>`. Der Rauschabstand erscheint nur bei „Rauschen an“, sein Wert steht unter `snr_kept`.
"""

import random
from dataclasses import dataclass

import streamlit as st

import cwt_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


def param_key(type_index: int, slot: int) -> str:
    return f"p{type_index}{slot}_kept"


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "signal_select": SettingSpec("s", str, C.DEFAULT_SIGNAL, choices=C.SIGNALS),
    "w0_select": SettingSpec("w", int, C.DEFAULT_W0, choices=C.W0_CHOICES),
    "fmin_select": SettingSpec("fm", float, C.DEFAULT_FMIN, choices=C.FMIN_CHOICES),
    "voices_select": SettingSpec("v", int, C.DEFAULT_VOICES, choices=C.VOICES_CHOICES),
    "power_select": SettingSpec("lm", str, C.DEFAULT_POWER, choices=C.POWER_MODES),
    "tcut_slider": SettingSpec("t", float, C.DEFAULT_TCUT, C.CUT_MIN, C.CUT_MAX),
    "fcut_slider": SettingSpec("fc", float, C.DEFAULT_FCUT, C.FCUT_MIN, C.FCUT_MAX),
    "noise_select": SettingSpec("g", str, C.DEFAULT_NOISE, choices=C.NOISE_MODES),
    "snr_kept": SettingSpec("snr", int, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
for _ti, _kind in enumerate(C.SIGNALS):
    for _slot, (_label, _lo, _hi, _step, _default, _help) in enumerate(C.PARAMS[_kind], start=1):
        SETTING_SPECS[param_key(_ti, _slot)] = SettingSpec(f"q{_ti}{_slot}", float, _default, _lo, _hi)
PRESET_KEYS = {"n": "n_select", "signal": "signal_select", "w0": "w0_select", "fmin": "fmin_select", "voices": "voices_select", "power": "power_select", "tcut": "tcut_slider", "fcut": "fcut_slider",
               "noise": "noise_select", "snr": "snr_kept", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        if key in preset:
            st.session_state[state_key] = preset[key]
    ti = C.SIGNALS.index(preset["signal"])
    for slot, value in enumerate(preset["params"][:len(C.PARAMS[preset["signal"]])], start=1):
        st.session_state[param_key(ti, slot)] = value


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
