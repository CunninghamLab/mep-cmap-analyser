"""
mep_cmap.detection.quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared signal quantification functions, the single source of truth for every
scalar trial metric. The pipeline calls these directly; anything computing the
same quantity a second way will eventually disagree with the trial CSVs, which
is exactly what happened while this module went unused.

  • compute_ptp          — peak-to-peak amplitude within a window
  • compute_rms          — RMS within a window (response or baseline)
  • compute_auc          — area under the rectified signal between two indices
  • compute_prestim_rms  — pre-stimulus RMS
  • compute_prestim_ptp  — pre-stimulus peak-to-peak
"""

import numpy as np

try:
    from ..compat import _np_trapz, _np_ptp
except ImportError:
    # Fallback for standalone use / testing. Resolved by feature detection
    # because NumPy 2 renamed trapz to trapezoid and removed the old name, so a
    # bare np.trapz raises AttributeError on import under NumPy 2 and makes the
    # module unimportable outside the package.
    _np_trapz = getattr(np, "trapezoid", None) or np.trapz
    _np_ptp   = np.ptp


def compute_ptp(segment, start_idx, end_idx):
    """
    Peak-to-peak amplitude of *segment* within [start_idx, end_idx).

    Parameters
    ----------
    segment   : 1-D np.ndarray  EMG trial segment
    start_idx : int             window start (samples)
    end_idx   : int             window end (samples, exclusive)

    Returns
    -------
    ptp : float  peak-to-peak amplitude (same units as segment)
    """
    window = segment[start_idx:end_idx]
    if len(window) == 0:
        return 0.0
    return float(_np_ptp(window))


def compute_rms(segment, start_idx, end_idx):
    """
    Root-mean-square of *segment* within [start_idx, end_idx).

    Complements compute_ptp over the same window. Peak-to-peak is set by two
    samples and is therefore sensitive to a single spike; RMS integrates the
    whole response, so a broad low-amplitude MEP and a narrow spiky one of the
    same peak-to-peak amplitude are distinguished. Neither is a substitute for
    the other, which is why both are reported.

    Windowed by index rather than taking a pre-sliced array, so callers use it
    exactly as they use compute_ptp and the two can never be computed over
    different windows by accident.

    Parameters
    ----------
    segment   : 1-D np.ndarray  EMG trial segment
    start_idx : int             window start (samples)
    end_idx   : int             window end (samples, exclusive)

    Returns
    -------
    rms : float  same units as segment; 0.0 for an empty window
    """
    window = segment[start_idx:end_idx]
    if len(window) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(window, dtype=float) ** 2)))


def compute_auc(segment, onset_idx, end_idx, fs):
    """
    Area under the rectified EMG signal from onset_idx to end_idx.

    Uses the trapezoidal rule on |segment|. The result is in mV·s when
    the segment is in mV and fs is in Hz.

    Parameters
    ----------
    segment   : 1-D np.ndarray  EMG trial segment
    onset_idx : int             onset sample index (inclusive)
    end_idx   : int             end sample index (exclusive)
    fs        : float           sampling frequency in Hz

    Returns
    -------
    auc : float, or None if window is empty or invalid
    """
    if end_idx <= onset_idx:
        return None
    window = np.abs(segment[onset_idx:end_idx])
    if len(window) == 0:
        return None
    return float(_np_trapz(window, dx=1.0 / fs))


def compute_prestim_rms(prestim_segment, demean=True, axis=None):
    """
    Root-mean-square of the pre-stimulus segment, DC offset removed by default.

    De-meaning is the default because any residual DC offset would otherwise
    enter the r.m.s. as between-trial variance unrelated to the state of the
    motoneurone pool. That matters wherever PreStimRMS is used as a regressor,
    notably the Carson (2026) excitability compensation, where the spurious
    variance attenuates the very association the method exists to remove.

    Before this parameter existed the function did not de-mean, while the
    pipeline did, so the two disagreed by roughly ten percent on a segment
    carrying a modest offset. `demean=True` is therefore the value that matches
    the PreStimRMS column in the trial CSVs; pass `demean=False` only when the
    raw, offset-inclusive r.m.s. is specifically what is wanted.

    Parameters
    ----------
    prestim_segment : np.ndarray  pre-stimulus EMG samples; 1-D for a single
                      window, or (trials, samples) with `axis=1`
    demean          : bool        remove the DC offset before squaring
    axis            : int | None  None for a single window (returns a float),
                      1 for a stack (returns one value per trial)

    Returns
    -------
    rms : float, or np.ndarray when `axis` is given
    """
    x = np.asarray(prestim_segment, dtype=float)
    if x.size == 0:
        return 0.0 if axis is None else np.zeros(0)
    if demean:
        x = x - np.mean(x, axis=axis, keepdims=(axis is not None))
    return (float(np.sqrt(np.mean(x ** 2))) if axis is None
            else np.sqrt(np.mean(x ** 2, axis=axis)))


def compute_prestim_ptp(prestim_segment):
    """
    Peak-to-peak amplitude of the pre-stimulus segment.

    Parameters
    ----------
    prestim_segment : 1-D np.ndarray  pre-stimulus EMG samples

    Returns
    -------
    ptp : float
    """
    if len(prestim_segment) == 0:
        return 0.0
    return float(_np_ptp(prestim_segment))
