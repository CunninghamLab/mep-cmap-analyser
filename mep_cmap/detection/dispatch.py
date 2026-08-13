"""
mep_cmap.detection.dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The one place that maps an onset method name onto a detector call.

Why this module exists
----------------------
The same dispatch used to be written out three times: once in
``pipeline._detect_onset_dispatch``, once inline in ``inspector.py``, and once
in ``detection.detect_mep_onset``. They drifted, as duplicated logic does, and
the drift was invisible because each copy independently produced plausible
numbers:

* The inspector's copy knew four methods. Selecting a fifth silently fell
  through to peak-fraction, so a re-detection in the inspector could use a
  different algorithm than the pipeline that produced the value it replaced.

* The inspector's copy passed no ``min_peak_amplitude``, so it used the
  detector's built-in 0.05 while the pipeline passed the configured value. An
  analyst who raised the amplitude gate got it honoured during analysis and
  ignored during review.

* The inspector's copy passed no ``peak_frac`` or ``slope_threshold`` either.
  The peak-fraction detector defaults ``slope_threshold`` to 0.05 while
  ``PipelineConfig`` defaults it to 0.08, so the two paths disagreed even at
  factory settings.

None of these raised an error. Routing every caller through one function makes
the class of bug structurally impossible rather than merely fixed, and
``tests/test_dispatch_parity.py`` asserts the pipeline and the inspector return
identical onsets for the same trace.

Parameters travel as a plain dict keyed by ``PipelineConfig`` field name — the
same dict ``detection.defaults.detector_params`` produces — so adding a method
parameter never changes a function signature anywhere.

  * dispatch_onset
"""

from .defaults import DETECTION_DEFAULTS, DEFAULT_ONSET_METHOD
from .onset_bigoni import detect_mep_onset_bigoni
from .onset_bigoni_walkback import detect_mep_onset_bigoni_walkback
from .onset_bootstrap import detect_mep_onset_bootstrap
from .onset_consensus import detect_mep_onset_consensus
from .onset_cusum import detect_mep_onset_cusum
from .onset_peak_fraction import detect_mep_onset_peak_fraction
from .onset_rms_envelope import detect_mep_onset_rms_envelope


def dispatch_onset(signal, fs, params=None, *,
                   pre_ms,
                   search_start_ms,
                   search_end_ms,
                   min_latency_ms=None,
                   max_latency_ms=None,
                   method=None):
    """
    Run the configured onset detector on one trace.

    Parameters
    ----------
    signal          : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs              : float  sampling frequency in Hz
    params          : dict or None  detection parameters keyed by
                      PipelineConfig field name. Missing keys fall back to
                      detection.defaults; None means all defaults.
    pre_ms          : float  ms of pre-stimulus data in ``signal``.

                      This MUST be the pre-stimulus length of the array being
                      passed, not a nominal baseline setting. The pipeline
                      extracts trials with ``cfg.pre_ms`` of lead-in while the
                      inspector uses ``prestim_ms``; passing the wrong one
                      mislocates the stimulus and every detector searches the
                      wrong region, returning None even for an obvious MEP.
    search_start_ms : float  ms post-stim to begin searching
    search_end_ms   : float  ms post-stim to stop searching
    min_latency_ms  : float or None  physiological floor (ms post-stim)
    max_latency_ms  : float or None  physiological ceiling (ms post-stim)
    method          : str or None  override ``params["onset_method"]``

    Returns
    -------
    latency_ms : float, or None if no confident onset was found

    Notes
    -----
    An unrecognised method falls through to peak-fraction, matching
    ``detection.detect_mep_onset``. That path is reachable only from a saved
    session naming a method this build does not provide.
    """
    p = dict(DETECTION_DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    if method is None:
        method = p.get("onset_method", DEFAULT_ONSET_METHOD)

    amp = p["min_peak_amplitude"]

    if method == "bootstrap":
        return detect_mep_onset_bootstrap(
            signal, fs,
            pre_ms=pre_ms,
            peak_search_start_ms=search_start_ms,
            peak_search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            criterion=p["onset_bootstrap_crit"],
            n_boot=p["onset_bootstrap_n"],
        )

    if method == "bigoni":
        return detect_mep_onset_bigoni(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            smooth_window_ms=p["onset_bigoni_smooth_ms"],
            min_run_ms=p["onset_bigoni_min_run_ms"],
        )

    if method == "bigoni_walkback":
        return detect_mep_onset_bigoni_walkback(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            smooth_window_ms=p["onset_bigoni_smooth_ms"],
            min_run_ms=p["onset_bigoni_min_run_ms"],
            walkback_sd_mult=p["onset_bigoni_walkback_sd"],
        )

    if method == "rms_envelope":
        return detect_mep_onset_rms_envelope(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            env_window_ms=p["onset_env_window_ms"],
            criterion=p["onset_env_criterion"],
            significance=p["onset_env_significance"],
            n_boot=p["onset_env_n_boot"],
            min_run_ms=p["onset_env_min_run_ms"],
            min_response_ms=p["onset_env_min_response_ms"],
            use_tkeo=p["onset_env_tkeo"],
            causal_window=p["onset_env_causal"],
            refine_on_raw=p["onset_env_refine"],
            refine_window_ms=p["onset_env_refine_window_ms"],
            refine_sd_mult=p["onset_env_refine_sd"],
            refine_sustain_ms=p["onset_env_refine_sustain_ms"],
        )

    if method == "cusum":
        return detect_mep_onset_cusum(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            k_mult=p["onset_cusum_k"],
            h_mult=p["onset_cusum_h"],
            max_accum_ms=p["onset_cusum_max_accum_ms"],
            min_response_ms=p["onset_cusum_min_response_ms"],
            use_tkeo=p["onset_cusum_tkeo"],
        )

    if method == "consensus":
        return detect_mep_onset_consensus(
            signal, fs,
            pre_ms=pre_ms,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            min_peak_amplitude=amp,
            methods=p["onset_consensus_methods"],
            params=p,
        )

    return detect_mep_onset_peak_fraction(
        signal, fs,
        pre_ms=pre_ms,
        poststim_start_ms=search_start_ms,
        poststim_end_ms=search_end_ms,
        peak_frac=p["peak_fraction"],
        min_consecutive=5,
        min_peak_amplitude=amp,
        slope_threshold=p["slope_threshold"],
    )
