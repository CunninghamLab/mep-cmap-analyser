"""
mep_cmap.detection.envelope_stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared statistics for envelope-based detection: RMS envelope construction,
baseline threshold estimation, and false-positive-calibrated run lengths.

Why the envelope and not the raw samples
----------------------------------------
``onset_bootstrap.py`` derives its threshold from the SD of the raw rectified
pre-stimulus samples. For zero-mean Gaussian noise the rectified signal has
sigma/mu ~= 0.755, so a ``mu + k*sigma`` threshold sits far above the noise
floor and single-sample excursions dominate the statistic. The RMS envelope
averages over the oscillation cycles of the interference pattern, so its SD is
several-fold smaller and a ``mu + k*sigma`` threshold on the envelope is both
tighter and more stable. ``csp_detection.py`` already relies on this reasoning
for silent-period detection; this module makes the same machinery available to
onset and offset detection.

Why a bootstrap run length and not a fixed minimum duration
-----------------------------------------------------------
Requiring the envelope to stay above threshold for N samples is what actually
controls the false-positive rate — the threshold alone does not, because noise
crosses any finite threshold eventually. Rather than hard-coding N, the chance
distribution of above-threshold run lengths is estimated by resampling the
pre-stimulus envelope. The requested percentile of that distribution is the
shortest run that is unlikely to have arisen from noise, which turns an
arbitrary tuning constant into a stated significance level.

  * compute_rms_envelope          -- moving-window RMS
  * compute_envelope_baseline     -- baseline mu / sd / threshold
  * bootstrap_runlength_criterion -- chance-calibrated minimum run length
  * find_sustained_run            -- first run satisfying threshold + duration
  * passes_width_guard            -- reject single-sample artefacts
"""

from collections import namedtuple

import numpy as np

# mu / sd / threshold of the pre-stimulus envelope.
EnvelopeBaseline = namedtuple("EnvelopeBaseline", "mu sd threshold")

_EPS = 1e-12


def compute_rms_envelope(x, fs, window_ms=5.0, causal=False):
    """
    Moving-window root-mean-square envelope, same length as the input.

    Parameters
    ----------
    x         : 1-D array_like  signal (raw — the RMS squares internally, so
                do NOT pre-rectify)
    fs        : float           sampling frequency in Hz
    window_ms : float           window width in ms (default 5.0)
    causal    : bool            if True the window spans [n-W+1, n] so the
                envelope never anticipates the signal; if False (default) the
                window is centred on n.

    Returns
    -------
    env : np.ndarray  same length as x

    Notes
    -----
    Edges are handled by replicating the boundary values rather than
    zero-padding. Zero-padding would produce an artificial roll-off at the
    start of the segment, which deflates the baseline SD and biases every
    threshold derived from it.

    A centred window smears the onset transition symmetrically, so the
    envelope crosses a fixed threshold roughly W/2 EARLY; a causal window
    crosses roughly W/2 LATE. Callers that need an unbiased latency should
    refine the coarse crossing on a much shorter window rather than applying
    a fixed W/2 correction — see ``onset_rms_envelope``.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=float)

    win = max(1, int(round(window_ms * fs / 1000.0)))
    if win == 1:
        return np.abs(x)
    win = min(win, x.size)

    sq = x ** 2
    if causal:
        pad_left, pad_right = win - 1, 0
    else:
        pad_left = (win - 1) // 2
        pad_right = win - 1 - pad_left

    # Reflection rather than edge replication. Replicating a single boundary
    # sample copies whatever that one noisy value happened to be, so the
    # envelope at the segment edge is driven by one sample; reflection
    # reproduces the local variance and keeps the baseline statistics honest.
    pad_mode = "reflect" if sq.size > max(pad_left, pad_right) else "edge"
    padded = np.pad(sq, (pad_left, pad_right), mode=pad_mode)
    kernel = np.ones(win, dtype=float) / float(win)
    mean_sq = np.convolve(padded, kernel, mode="valid")
    return np.sqrt(np.maximum(mean_sq, 0.0))


def compute_envelope_baseline(env_pre, criterion=2.5):
    """
    Baseline mean, SD and detection threshold from a pre-stimulus envelope.

    Deliberately applies NO clipping to the threshold. ``onset_bootstrap``
    clips to ``[1.5*mu, 5.0*mu]``, which on a quiet baseline forces the
    threshold up to 1.5*mu regardless of the SD and places onsets late. Here
    the threshold is exactly ``mu + criterion*sd``, and duration control is
    handled separately by ``bootstrap_runlength_criterion``.

    Parameters
    ----------
    env_pre   : 1-D array_like  pre-stimulus envelope samples
    criterion : float           SD multiplier (default 2.5)

    Returns
    -------
    EnvelopeBaseline(mu, sd, threshold), or None if the baseline is unusable
    (fewer than 5 samples, or effectively flat).
    """
    env_pre = np.asarray(env_pre, dtype=float)
    if env_pre.size < 5:
        return None

    mu = float(env_pre.mean())
    if not np.isfinite(mu) or abs(mu) < _EPS:
        return None

    sd = float(env_pre.std(ddof=1)) if env_pre.size > 1 else 0.0
    if not np.isfinite(sd):
        return None
    # A degenerate SD (constant baseline, e.g. a saturated or synthetic
    # channel) would give threshold == mu and fire on every sample. Fall back
    # to a small fraction of the mean so detection stays finite but strict.
    sd = max(sd, abs(mu) * 1e-3)

    return EnvelopeBaseline(mu=mu, sd=sd, threshold=mu + criterion * sd)


def _run_lengths(mask_2d):
    """
    Lengths of every contiguous True run in each row of a 2-D boolean array.

    Returns a flat 1-D array of run lengths pooled across rows.
    """
    pad = np.zeros((mask_2d.shape[0], 1), dtype=np.int8)
    padded = np.concatenate([pad, mask_2d.astype(np.int8), pad], axis=1)
    d = np.diff(padded, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if starts.shape[0] == 0 or starts.shape[0] != ends.shape[0]:
        return np.zeros(0, dtype=int)
    # Rows are traversed in order by argwhere, and each row contributes the
    # same number of starts as ends, so the two index arrays align pairwise.
    return (ends[:, 1] - starts[:, 1]).astype(int)


def bootstrap_runlength_criterion(env_pre, criterion=2.5, significance=0.99,
                                  n_boot=500, seed=42, tail="upper",
                                  min_samples=2, block_samples=1):
    """
    Shortest above-threshold run length unlikely to arise from noise alone.

    The pre-stimulus envelope is resampled ``n_boot`` times. Each resample gets
    its own mu and sd, a threshold is applied, and the lengths of all
    threshold-exceeding runs are pooled. The ``significance``-th percentile of
    that pooled distribution is returned: a run at least this long occurs by
    chance with probability ``1 - significance``.

    Why the resampling must be in BLOCKS
    ------------------------------------
    An i.i.d. resample destroys temporal autocorrelation. A moving-window RMS
    envelope is heavily autocorrelated by construction — adjacent samples share
    all but one input sample — so genuine chance excursions persist for a
    substantial fraction of the window width. Resampling sample-by-sample
    produces chance runs of length ~1 and therefore reports a criterion of one
    or two samples, which any noisy baseline satisfies constantly. In testing,
    an i.i.d. version of this function returned 2 samples for a 5 ms (25-sample)
    envelope and admitted pure-noise onsets.

    ``block_samples`` should therefore be set to the envelope window width in
    samples. A circular block bootstrap is used so every sample has equal
    inclusion probability regardless of position.

    Note that ``csp_detection.detect_csp_bootstrap`` contains an i.i.d. version
    of this calculation and has the same optimism; there it is masked by the
    ``min_silence_ms`` floor (25 ms by default), which dominates the returned
    criterion in practice.

    Parameters
    ----------
    env_pre       : 1-D array_like  pre-stimulus envelope samples
    criterion     : float  SD multiplier, must match the detection threshold
    significance  : float  percentile as a fraction (default 0.99)
    n_boot        : int    resamples (default 500)
    seed          : int    RNG seed — detection must be reproducible
    tail          : str    'upper' (excursions above baseline, onset),
                           'lower' (below, silent period),
                           'both'  (two-sided)
    min_samples   : int    hard floor on the returned value
    block_samples : int    block length; set to the envelope window width in
                           samples. 1 reproduces the i.i.d. behaviour and
                           should only be used on unsmoothed data.

    Returns
    -------
    criterion_samples : int
    """
    env_pre = np.asarray(env_pre, dtype=float)
    min_samples = max(1, int(min_samples))
    if env_pre.size < 5:
        return min_samples

    rng = np.random.default_rng(seed)
    n_pre = env_pre.size
    n_boot = int(n_boot)
    block = max(1, min(int(block_samples), n_pre))

    if block == 1:
        idx = rng.integers(0, n_pre, size=(n_boot, n_pre))
    else:
        # Circular block bootstrap: draw ceil(n_pre/block) block starts per
        # resample and lay the blocks end to end, wrapping at the array edge.
        n_blocks = int(np.ceil(n_pre / float(block)))
        starts = rng.integers(0, n_pre, size=(n_boot, n_blocks, 1))
        within = np.arange(block).reshape(1, 1, block)
        idx = ((starts + within) % n_pre).reshape(n_boot, n_blocks * block)
        idx = idx[:, :n_pre]

    resamp = env_pre[idx]

    mu = resamp.mean(axis=1, keepdims=True)
    sd = np.maximum(resamp.std(axis=1, ddof=1, keepdims=True), _EPS)

    if tail == "upper":
        mask = resamp > (mu + criterion * sd)
    elif tail == "lower":
        mask = resamp < (mu - criterion * sd)
    elif tail == "both":
        mask = (resamp > (mu + criterion * sd)) | (resamp < (mu - criterion * sd))
    else:
        raise ValueError("tail must be 'upper', 'lower' or 'both'")

    lengths = _run_lengths(mask)
    if lengths.size == 0:
        return min_samples

    crit = int(np.percentile(lengths, float(significance) * 100.0))
    return max(crit, min_samples)


def find_sustained_run(values, threshold, min_run, lo=0, hi=None,
                       above=True):
    """
    Index of the first sample beginning a sustained excursion past threshold.

    Parameters
    ----------
    values    : 1-D np.ndarray  envelope (or any test statistic)
    threshold : float
    min_run   : int   required number of consecutive qualifying samples
    lo, hi    : int   half-open search bounds into ``values``
    above     : bool  True tests ``values > threshold``, False ``values <``

    Returns
    -------
    int index into ``values`` where the qualifying run starts, or None.

    A run that is still qualifying when the search bound is reached counts
    only if it already spans ``min_run`` samples; a truncated run is not
    credited, so results do not depend on where the window happens to end.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    hi = n if hi is None else min(int(hi), n)
    lo = max(0, int(lo))
    min_run = max(1, int(min_run))
    if lo >= hi:
        return None

    window = values[lo:hi]
    mask = (window > threshold) if above else (window < threshold)
    if not mask.any():
        return None

    run_start = None
    for i, flag in enumerate(mask):
        if flag:
            if run_start is None:
                run_start = i
            if (i - run_start + 1) >= min_run:
                return lo + run_start
        else:
            run_start = None
    return None


def passes_width_guard(fine_env, threshold, onset_idx, window_samples,
                       duty=0.5, slack_samples=None):
    """
    Reject candidate onsets that are transients rather than responses.

    A moving-window RMS envelope converts a SINGLE-SAMPLE artefact — a cable
    movement transient, an electrical spike, a stimulator discharge — into an
    excursion as wide as the window itself, which then satisfies any run-length
    criterion shorter than the window. Measured on synthetic data, a lone
    0.6 mV one-sample spike was accepted as an onset by the envelope and CUSUM
    detectors on 50 trials out of 50, while the derivative-based detector
    rejected 49 of them. Amplitude- and energy-based detectors need this guard;
    derivative-based ones largely do not.

    The test is applied to a SHORT envelope, whose smearing is much narrower
    than the detection window, and asks what fraction of the first
    ``min_samples`` after the candidate onset are genuinely elevated. A real
    evoked response is continuously above threshold over its first few
    milliseconds; a smeared spike is elevated only near its centre.

    Why the test is slack-tolerant
    ------------------------------
    Onset estimators do not all aim at the same point of the rise. The refined
    envelope detector lands essentially on it; CUSUM deliberately reports the
    last sample at which the accumulator was zero, which PRECEDES the change;
    an unrefined envelope anchor leads the response by about half the smoothing
    window. A window anchored rigidly at the estimate is therefore part
    baseline for some detectors, and in testing rejected every trial from the
    unrefined envelope detector and 2 of 30 clean CUSUM trials.

    The test instead asks whether a qualifying stretch BEGINS anywhere within
    ``slack_samples`` of the estimate. That is the physiologically meaningful
    question — is there a response here — and it is independent of each
    detector's convention about where on the rise to point. A single-sample
    artefact still cannot satisfy it at any offset.

    Parameters
    ----------
    fine_env      : 1-D np.ndarray  short-window envelope
    threshold     : float           elevation threshold for that envelope
    onset_idx     : int             candidate onset index
    window_samples: int             length of the stretch that must qualify
    duty          : float           minimum fraction that must exceed threshold
    slack_samples : int or None     how far past the estimate the stretch may
                    begin; defaults to ``window_samples``

    Returns
    -------
    bool
    """
    fine_env = np.asarray(fine_env, dtype=float)
    onset_idx = int(onset_idx)
    window_samples = max(1, int(window_samples))
    slack = window_samples if slack_samples is None else max(0, int(slack_samples))
    if onset_idx < 0 or onset_idx >= fine_env.size:
        return False

    elevated = (fine_env > threshold).astype(float)
    for start in range(onset_idx, onset_idx + slack + 1):
        window = elevated[start:start + window_samples]
        if window.size < window_samples:
            break
        if float(window.mean()) >= float(duty):
            return True
    return False
