"""
mep_cmap.detection.tkeo
~~~~~~~~~~~~~~~~~~~~~~~
Teager-Kaiser Energy Operator (TKEO) preconditioning for onset detection.

The TKEO estimates the instantaneous energy of a signal as a product of its
amplitude AND frequency content:

    psi[n] = x[n]^2 - x[n-1] * x[n+1]

For a pure sinusoid of amplitude A and normalised frequency w this evaluates
to A^2 * sin^2(w), so the operator amplifies components that are both large
and fast-changing. Baseline EMG noise is low-amplitude and broadband; the
rising edge of an MEP or CMAP is high-amplitude and fast. The TKEO therefore
increases the contrast of the *transition* rather than of the amplitude alone,
which is precisely the quantity an onset detector needs.

Empirically this improves threshold-based onset detection most at low
signal-to-noise ratio, where a plain amplitude threshold on the rectified
signal is least reliable (Solnik et al., Eur J Appl Physiol 2010;110:489-498).

IMPORTANT — units
-----------------
The output of the TKEO is in the SQUARE of the input units (mV^2 for a signal
in mV) and is not a physical amplitude. Any threshold applied to a TKEO signal
must be derived from the baseline statistics of the TKEO signal itself, never
from the baseline of the raw recording. Callers must not mix the two.

  * apply_tkeo   -- Teager-Kaiser energy operator
"""

import numpy as np


def apply_tkeo(x):
    """
    Teager-Kaiser Energy Operator.

    Parameters
    ----------
    x : 1-D array_like
        Input signal (raw, NOT rectified — the operator squares internally).

    Returns
    -------
    psi : np.ndarray
        Same length as ``x``. The first and last samples cannot be computed
        from the three-point kernel and are set equal to their nearest
        computable neighbour, which keeps the array length stable and avoids
        introducing two artificial zeros into the baseline statistics.

    Notes
    -----
    The operator is not sign-definite: ``psi`` can be negative where the
    signal is locally concave. Downstream envelope computation squares the
    values again, so no explicit rectification is applied here.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.zeros(n, dtype=float)

    psi = np.empty(n, dtype=float)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    # Edge replication rather than zero-fill: zeros would deflate the
    # baseline SD and bias every threshold derived from it downward.
    psi[0] = psi[1]
    psi[-1] = psi[-2]
    return psi
