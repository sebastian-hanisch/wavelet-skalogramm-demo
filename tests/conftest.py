"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """Impuls bei t = 4 in einer Reihe der Länge n = 9, Skala s = 2 (Frequenz f = w0 / (2 pi s) bei w0 = 6): Summendefinition W(2, n) = conj(psi((4 - n) / 2)) / sqrt(2),
    also |W(n)| = pi^(-1/4) / sqrt(2) * exp(-(n - 4)^2 / 8) = (0,07188; 0,17243; 0,32214; 0,46872; 0,53113; 0,46872; 0,32214; 0,17243; 0,07188)."""
    x = np.zeros(9)
    x[4] = 1.0
    return x
