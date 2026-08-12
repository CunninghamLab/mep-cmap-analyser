"""
mep_cmap.variability — shared statistics core for the variability add-ons.

Pure computation: no GUI, no file I/O, no add-on contract. Both the first-level
(`add_ons/single_file/variability.py`) and second-level
(`add_ons/group_level/variability_group.py`) add-ons import from here, so the
two levels can never drift apart in how they define a coefficient of variation,
an ICC, or a limit of agreement.

Scope of each level
-------------------
Within ONE recording there is no between-participant variance, so nothing here
returns a reliability coefficient at that level. What a single recording can
support is spread (CV), outlying trials (z), precision of its own mean (CI),
serial structure (AR), agreement between independent halves (LoA), and whether
amplitude drifts across the session (the block drift index). Reliability proper
needs several participants and is computed at the group level, from variance
components and the classical ICC forms.

Conventions
-----------
* No type annotations and no f-string `=` specifiers, matching the rest of the
  package and keeping the module importable on Python 3.9.
* matplotlib is imported lazily inside the figure helpers so that headless or
  figure-free runs never pay for it.
* Every function takes and returns plain arrays, dicts and DataFrames. Naming
  of output columns is the add-on's job, not this module's.

References
----------
Shrout & Fleiss (1979) Psychol Bull 86:420; McGraw & Wong (1996) Psychol
Methods 1:30; Bland & Altman (1999) Stat Methods Med Res 8:135; Vangel (1996)
Am Stat 50:21 (modified McKay interval); Searle et al. (1992) Variance
Components (unbalanced ANOVA coefficients); Efron & Tibshirani (1993) for BCa.
"""

import warnings

import numpy as np

try:
    import pandas as pd
except Exception:                       # pandas always present in the app; guard anyway
    pd = None

try:
    from scipy import stats as _st
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_moment(fn, x):
    """Skew or kurtosis that returns NaN rather than emitting a warning.

    A near-degenerate series (say 79 identical trials and one different) makes
    scipy warn about catastrophic cancellation. The value would be meaningless
    anyway, so it is reported as missing instead of surfacing a numerical
    warning in the application log.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return float(fn(x, bias=False))
    except Exception:
        return np.nan


def _safe_test(fn, *args, **kwargs):
    """Run a scipy test, returning NaNs if it warns or fails.

    A condition where every trial is identical makes the t and variance tests
    undefined, and scipy warns rather than raising. The descriptive half of the
    comparison is still perfectly meaningful, so the test statistics are
    reported as missing and the rest of the row survives.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = fn(*args, **kwargs)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return (np.nan, np.nan)


def _safe_linregress(x, y):
    """linregress that returns NaNs instead of raising on a flat predictor.

    scipy raises when every x is identical, which happens legitimately here:
    two Bland-Altman pairs with the same mean, or a condition where every trial
    came out at the same value. A degenerate slope is a missing number, not a
    reason to abort the whole recording.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"slope": np.nan, "intercept": np.nan, "rvalue": np.nan,
                "pvalue": np.nan, "stderr": np.nan}
    lr = _st.linregress(x[ok], y[ok])
    return {"slope": float(lr.slope), "intercept": float(lr.intercept),
            "rvalue": float(lr.rvalue), "pvalue": float(lr.pvalue),
            "stderr": float(lr.stderr)}


def _require_scipy(what):
    if not _HAVE_SCIPY:
        raise RuntimeError("the variability add-ons require SciPy (" + what + ").")


def clean_metric(values):
    """Coerce to a float array and drop non-finite entries.

    Trial tables carry sentinel strings such as 'Not Marked' and 'Not Detected'
    in otherwise numeric columns, so plain float() casting is never safe here.
    """
    if pd is not None:
        v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    else:
        v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def mad_normal(x):
    """Median absolute deviation scaled to be a consistent estimator of sigma."""
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Descriptives
# ─────────────────────────────────────────────────────────────────────────────
def descriptives(x):
    """Central tendency, spread and shape, on both the raw and log scale.

    MEP amplitudes are commonly closer to lognormal than normal, so the log-scale
    summaries are reported alongside rather than instead of the raw ones, and the
    caller decides which to lead with.
    """
    _require_scipy("scipy.stats")
    x = np.asarray(x, float)
    n = x.size
    out = {"n": n}
    if n == 0:
        return out

    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    out.update({
        "mean": float(np.mean(x)),
        "sd": sd,
        "sem": sd / np.sqrt(n) if n > 1 else np.nan,
        "median": float(np.median(x)),
        "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
        "mad": mad_normal(x),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        # Shape statistics are undefined for a constant series, and scipy warns
        # about catastrophic cancellation if asked for them anyway.
        "skew": (_safe_moment(_st.skew, x) if (n > 2 and np.ptp(x) > 0) else np.nan),
        "kurtosis_excess": (_safe_moment(_st.kurtosis, x)
                            if (n > 3 and np.ptp(x) > 0) else np.nan),
    })
    out["max_min_ratio"] = out["max"] / out["min"] if out["min"] > 0 else np.nan

    if np.ptp(x) == 0:
        # Every trial identical. Nothing further is defined, and Shapiro-Wilk
        # warns rather than failing, so stop here and say so explicitly.
        out["constant_series"] = True
        return out

    if n >= 3:
        w, p = _st.shapiro(x)
        out["shapiro_W"], out["shapiro_p"] = float(w), float(p)

    if n > 1 and np.all(x > 0):
        lx = np.log(x)
        out["geometric_mean"] = float(np.exp(np.mean(lx)))
        out["geometric_sd"] = float(np.exp(np.std(lx, ddof=1)))
        if n >= 3:
            wl, pl = _st.shapiro(lx)
            out["shapiro_W_log"], out["shapiro_p_log"] = float(wl), float(pl)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Coefficient of variation
# ─────────────────────────────────────────────────────────────────────────────
def mckay_ci(cv, n, alpha=0.05):
    """Modified McKay confidence interval for a CV (Vangel 1996).

    Accurate while the CV is below roughly 0.33; beyond that the bootstrap
    interval from cv_analysis() is the one to trust.
    """
    _require_scipy("scipy.stats")
    if not np.isfinite(cv) or n < 2:
        return (np.nan, np.nan)
    v = n - 1
    u1 = _st.chi2.ppf(1 - alpha / 2.0, v)
    u2 = _st.chi2.ppf(alpha / 2.0, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_lo = ((u1 + 2.0) / (v + 1.0) - 1.0) * cv ** 2 + u1 / v
        d_hi = ((u2 + 2.0) / (v + 1.0) - 1.0) * cv ** 2 + u2 / v
        lo = cv / np.sqrt(d_lo) if d_lo > 0 else np.nan
        hi = cv / np.sqrt(d_hi) if d_hi > 0 else np.nan
    return (float(lo), float(hi))


def cv_analysis(x, n_boot=2000, seed=42, alpha=0.05):
    """Coefficient of variation as a percentage, four ways, with intervals.

    The four are not redundant. The raw CV is what most papers report; the
    small-sample correction matters below ~30 trials; the log-based CV is the
    right one if amplitudes are lognormal; the robust CV ignores a stray trial
    that would otherwise inflate the others.
    """
    x = np.asarray(x, float)
    n = x.size
    out = {}
    if n < 2:
        return out
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(m) or m == 0:
        return out
    cv = sd / m

    out["cv_percent"] = 100.0 * cv
    out["cv_corrected_percent"] = 100.0 * cv * (1.0 + 1.0 / (4.0 * n))
    med = float(np.median(x))
    out["cv_robust_percent"] = 100.0 * mad_normal(x) / med if med else np.nan
    if np.all(x > 0):
        s_ln = float(np.std(np.log(x), ddof=1))
        out["cv_log_percent"] = 100.0 * np.sqrt(np.exp(s_ln ** 2) - 1.0)

    lo, hi = mckay_ci(cv, n, alpha)
    out["cv_mckay_lo_percent"] = 100.0 * lo
    out["cv_mckay_hi_percent"] = 100.0 * hi

    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        bs = x[rng.integers(0, n, size=(int(n_boot), n))]
        bm = bs.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bcv = 100.0 * bs.std(axis=1, ddof=1) / bm
        bcv = bcv[np.isfinite(bcv)]
        if bcv.size:
            out["cv_boot_lo_percent"] = float(np.percentile(bcv, 100 * alpha / 2.0))
            out["cv_boot_hi_percent"] = float(np.percentile(bcv, 100 * (1 - alpha / 2.0)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Z scores
# ─────────────────────────────────────────────────────────────────────────────
def z_scores(x, robust_thresh=3.5):
    """Robust and log-scale z scores for one condition.

    The core pipeline already writes classical within-condition and pooled z
    scores into the trial table, so those are deliberately NOT recomputed here.
    What this adds is the median/MAD version, which does not let an outlier
    inflate its own denominator, and the log-scale version, which is the correct
    one when amplitudes are lognormal.
    """
    x = np.asarray(x, float)
    n = x.size
    out = {"n": n}
    if n == 0:
        return out

    med = float(np.median(x))
    mad = mad_normal(x)
    rz = (x - med) / mad if mad and np.isfinite(mad) and mad > 0 else np.full(n, np.nan)

    zl = np.full(n, np.nan)
    if n > 1 and np.all(x > 0):
        lx = np.log(x)
        s = float(np.std(lx, ddof=1))
        if s > 0:
            zl = (lx - lx.mean()) / s

    out["z_robust"] = rz
    out["z_log"] = zl
    out["robust_z_threshold"] = robust_thresh
    finite = np.isfinite(rz)
    out["n_extreme_robust_z"] = int(np.sum(np.abs(rz[finite]) > robust_thresh))
    out["max_abs_robust_z"] = float(np.max(np.abs(rz[finite]))) if finite.any() else np.nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Confidence intervals and required trial counts
# ─────────────────────────────────────────────────────────────────────────────
def boot_bca_ci(x, n_boot=2000, alpha=0.05, seed=42):
    """Bias-corrected and accelerated bootstrap interval for the mean."""
    _require_scipy("scipy.stats")
    x = np.asarray(x, float)
    n = x.size
    if n < 3:
        return (np.nan, np.nan)
    theta = float(np.mean(x))
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(int(n_boot), n))].mean(axis=1)

    prop = float(np.mean(boots < theta))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = _st.norm.ppf(prop)

    jack = (x.sum() - x) / (n - 1.0)          # leave-one-out means, vectorised
    jbar = jack.mean()
    num = float(np.sum((jbar - jack) ** 3))
    den = 6.0 * (float(np.sum((jbar - jack) ** 2)) ** 1.5)
    a = num / den if den != 0 else 0.0

    zl = _st.norm.ppf(alpha / 2.0)
    zu = _st.norm.ppf(1 - alpha / 2.0)
    a1 = _st.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = _st.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    return (float(np.percentile(boots, 100 * a1)),
            float(np.percentile(boots, 100 * a2)))


def ci_analysis(x, r1=None, n_boot=2000, alpha=0.05, seed=42):
    """Confidence intervals for the mean of one condition.

    When `r1` (the lag-1 autocorrelation) is supplied and positive, an interval
    based on the effective sample size is added: positively dependent trials
    carry less information than their count suggests, so the naive interval is
    too narrow. Negative dependence is not used to narrow the interval, which
    would be optimistic, so n_eff is capped at n.
    """
    _require_scipy("scipy.stats")
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return {}
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(_st.t.ppf(1 - alpha / 2.0, n - 1))

    out = {
        "mean": m,
        "ci_lo": m - tcrit * sem,
        "ci_hi": m + tcrit * sem,
        "ci_halfwidth": tcrit * sem,
        "ci_halfwidth_pct_of_mean": 100.0 * tcrit * sem / m if m else np.nan,
    }

    if n >= 3 and n_boot:
        lo, hi = boot_bca_ci(x, n_boot=n_boot, alpha=alpha, seed=seed)
        out["ci_boot_lo"], out["ci_boot_hi"] = lo, hi

    if np.all(x > 0):
        lx = np.log(x)
        lm = float(lx.mean())
        lsem = float(lx.std(ddof=1)) / np.sqrt(n)
        out["geo_mean"] = float(np.exp(lm))
        out["geo_ci_lo"] = float(np.exp(lm - tcrit * lsem))
        out["geo_ci_hi"] = float(np.exp(lm + tcrit * lsem))

    if r1 is not None and np.isfinite(r1) and -0.999 < r1 < 0.999:
        n_eff = n * (1.0 - r1) / (1.0 + r1)
        out["n_eff_capped"] = bool(n_eff > n)
        n_eff = float(np.clip(n_eff, 2.0, float(n)))
        t_eff = float(_st.t.ppf(1 - alpha / 2.0, n_eff - 1))
        sem_eff = sd / np.sqrt(n_eff)
        out["n_eff"] = n_eff
        out["ci_ar_lo"] = m - t_eff * sem_eff
        out["ci_ar_hi"] = m + t_eff * sem_eff
        out["ci_ar_halfwidth_pct_of_mean"] = 100.0 * t_eff * sem_eff / m if m else np.nan
    return out


def trials_for_precision(cv_percent, targets=(5, 10, 15), alpha=0.05, n_max=500):
    """Trials needed for the CI half-width of a mean to fall within `target` %.

    Assumes the observed CV continues to hold, which is the usual planning
    assumption and worth stating whenever the number is quoted.
    """
    _require_scipy("scipy.stats")
    out = {}
    if not np.isfinite(cv_percent) or cv_percent <= 0:
        for tgt in targets:
            out["n_trials_for_" + str(tgt) + "pct_ci"] = np.nan
        return out
    cv = cv_percent / 100.0
    for tgt in targets:
        need = np.nan
        for m in range(2, int(n_max) + 1):
            hw = float(_st.t.ppf(1 - alpha / 2.0, m - 1)) * cv / np.sqrt(m) * 100.0
            if hw <= tgt:
                need = m
                break
        out["n_trials_for_" + str(tgt) + "pct_ci"] = need
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Serial structure
# ─────────────────────────────────────────────────────────────────────────────
def acf(x, nlags):
    """Sample autocorrelation, lags 0..nlags, n-denominator convention."""
    x = np.asarray(x, float)
    n = x.size
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return np.zeros(int(nlags) + 1)
    return np.array([float(np.dot(xc[:n - k], xc[k:])) / denom
                     for k in range(int(nlags) + 1)])


def pacf_levinson(r, nlags):
    """Partial autocorrelations from an ACF by Levinson-Durbin recursion."""
    nlags = int(nlags)
    pac = np.zeros(nlags + 1)
    pac[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    if nlags >= 1:
        phi[1, 1] = r[1]
        pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den if den != 0 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pac[k] = phi[k, k]
    return pac


def ljung_box(resid, nlags, n_params=0):
    """Ljung-Box portmanteau test for remaining autocorrelation."""
    _require_scipy("scipy.stats")
    resid = np.asarray(resid, float)
    n = resid.size
    nlags = int(min(nlags, max(1, n - 2)))
    r = acf(resid, nlags)
    q = n * (n + 2.0) * sum(r[k] ** 2 / (n - k) for k in range(1, nlags + 1))
    df = max(nlags - int(n_params), 1)
    return {"Q": float(q), "df": df, "p": float(_st.chi2.sf(q, df))}


def durbin_watson(resid):
    resid = np.asarray(resid, float)
    ss = float(np.sum(resid ** 2))
    if ss == 0:
        return np.nan
    return float(np.sum(np.diff(resid) ** 2) / ss)


def fit_ar_ols(x, p, p_max):
    """AR(p) by OLS on mean-centred data.

    Every candidate order is fitted on the same effective sample (t = p_max
    onward) so that AICc values are comparable across orders rather than being
    fitted to different amounts of data.
    """
    _require_scipy("scipy.stats")
    x = np.asarray(x, float)
    n = x.size
    xc = x - x.mean()
    start = int(p_max)
    y = xc[start:]
    n_eff = y.size

    if p == 0:
        X = np.ones((n_eff, 1))
    else:
        cols = [np.ones(n_eff)] + [xc[start - k:n - k] for k in range(1, p + 1)]
        X = np.column_stack(cols)

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(beta)
    rss = float(np.sum(resid ** 2))
    k = X.shape[1] + 1
    sigma2 = rss / n_eff if n_eff else np.nan
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return {"p": p, "coef": beta, "se": np.full(X.shape[1], np.nan),
                "t": np.full(X.shape[1], np.nan), "pval": np.full(X.shape[1], np.nan),
                "resid": resid, "sigma2": sigma2, "aicc": np.inf, "bic": np.inf,
                "n_eff": n_eff}

    llf = -0.5 * n_eff * (np.log(2 * np.pi * sigma2) + 1.0)
    aic = -2 * llf + 2 * k
    aicc = aic + (2.0 * k * (k + 1)) / (n_eff - k - 1) if n_eff - k - 1 > 0 else np.inf
    bic = -2 * llf + k * np.log(n_eff)

    dof = n_eff - X.shape[1]
    se = np.full(X.shape[1], np.nan)
    if dof > 0:
        se = np.sqrt(np.diag(np.linalg.pinv(X.T.dot(X))) * rss / dof)
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = beta / se
    pv = 2 * _st.t.sf(np.abs(tv), dof) if dof > 0 else np.full(beta.shape, np.nan)

    return {"p": p, "coef": beta, "se": se, "t": tv, "pval": pv, "resid": resid,
            "sigma2": sigma2, "aicc": float(aicc), "bic": float(bic), "n_eff": n_eff}


def ar_analysis(x, p_max=None, lb_lags=None):
    """Serial dependence and drift across the trial order of one condition.

    Drift and autocorrelation confound each other, so a linear trend test is
    reported beside the AR fit rather than instead of it: a slow downward drift
    in amplitude will show up as positive lag-1 autocorrelation if trend is not
    considered separately.
    """
    _require_scipy("scipy.stats")
    x = np.asarray(x, float)
    n = x.size
    if n < 6:
        return {"error": "at least 6 trials are needed for the AR analysis."}

    if p_max is None:
        p_max = int(min(5, max(1, n // 5)))
    if lb_lags is None:
        lb_lags = int(min(10, max(2, n // 4)))
    nlags = int(min(max(5, n // 2), n - 2))

    r = acf(x, nlags)
    pac = pacf_levinson(r, nlags)
    bound = 1.96 / np.sqrt(n)

    fits = [fit_ar_ols(x, p, p_max) for p in range(0, int(p_max) + 1)]
    best = min(fits, key=lambda f: f["aicc"])

    t = np.arange(1.0, n + 1.0)
    lr = _safe_linregress(t, x)
    m = float(np.mean(x))

    return {
        "acf": r, "pacf": pac, "acf_bound": float(bound),
        "r1": float(r[1]) if nlags >= 1 else np.nan,
        "r1_significant": bool(abs(r[1]) > bound) if nlags >= 1 else False,
        "p_max": int(p_max),
        "best_order": int(best["p"]),
        "best_fit": best,
        "ar_phi1": float(best["coef"][1]) if best["p"] > 0 else np.nan,
        "ar_phi1_p": float(best["pval"][1]) if best["p"] > 0 else np.nan,
        "ljung_box_raw": ljung_box(x - x.mean(), lb_lags, 0),
        "ljung_box_resid": ljung_box(best["resid"], lb_lags, best["p"]),
        "durbin_watson": durbin_watson(x - x.mean()),
        "trend_slope": lr["slope"],
        "trend_slope_se": lr["stderr"],
        "trend_p": lr["pvalue"],
        "trend_r2": lr["rvalue"] ** 2,
        "trend_pct_per_10_trials": (100.0 * lr["slope"] * 10.0 / m
                                    if m and np.isfinite(lr["slope"]) else np.nan),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agreement
# ─────────────────────────────────────────────────────────────────────────────
def bland_altman(a, b, alpha=0.05):
    """Bias and 95% limits of agreement for paired measurements."""
    _require_scipy("scipy.stats")
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    avg = (a + b) / 2.0
    n = d.size
    if n < 2:
        return {"error": "at least 2 pairs are needed."}

    bias = float(np.mean(d))
    sdd = float(np.std(d, ddof=1))
    tcrit = float(_st.t.ppf(1 - alpha / 2.0, n - 1))

    loa_lo = bias - 1.96 * sdd
    loa_hi = bias + 1.96 * sdd
    se_bias = sdd / np.sqrt(n)
    se_loa = sdd * np.sqrt(1.0 / n + (1.96 ** 2) / (2.0 * (n - 1)))

    lr = _safe_linregress(avg, d)
    grand = float(np.mean(avg))

    return {
        "n_pairs": int(n), "diff": d, "avg": avg,
        "bias": bias,
        "bias_ci_lo": bias - tcrit * se_bias,
        "bias_ci_hi": bias + tcrit * se_bias,
        "bias_p": float(_st.ttest_1samp(d, 0.0).pvalue),
        "sd_diff": sdd,
        "loa_lo": float(loa_lo), "loa_hi": float(loa_hi),
        "loa_lo_ci": (float(loa_lo - tcrit * se_loa), float(loa_lo + tcrit * se_loa)),
        "loa_hi_ci": (float(loa_hi - tcrit * se_loa), float(loa_hi + tcrit * se_loa)),
        "loa_width": float(loa_hi - loa_lo),
        "loa_width_pct_of_mean": float(100.0 * (loa_hi - loa_lo) / grand) if grand else np.nan,
        "prop_bias_slope": lr["slope"],
        "prop_bias_p": lr["pvalue"],
    }


def bland_altman_ratio(a, b):
    """Limits of agreement on the log scale, back-transformed to ratios.

    Preferable to the raw limits whenever the spread of the differences grows
    with amplitude, which is the norm for MEPs.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 2 or np.any(a <= 0) or np.any(b <= 0):
        return {}
    ba = bland_altman(np.log(a), np.log(b))
    if "error" in ba:
        return {}
    return {
        "n_pairs": ba["n_pairs"],
        "ratio_bias": float(np.exp(ba["bias"])),
        "ratio_loa_lo": float(np.exp(ba["loa_lo"])),
        "ratio_loa_hi": float(np.exp(ba["loa_hi"])),
        "log_avg": ba["avg"], "log_diff": ba["diff"],
    }


def make_pairs(x, mode="single", block_size=5):
    """Build paired measurements from one uninterrupted run of trials.

    A single session has no repeat measurement, so pairs must be constructed.
    Each mode answers a different question and none of them is test-retest
    agreement, which needs a second session and lives at the group level.

      single : odd against even trials. How far can one trial sit from another
               trial of the same condition.
      block  : consecutive block means, paired 1v2, 3v4, and so on.
      half   : first half against second half, which exposes drift.
    """
    x = np.asarray(x, float)
    n = x.size
    if mode == "single":
        k = n // 2
        return x[0:2 * k:2], x[1:2 * k:2], "odd against even trials (single-trial agreement)"

    bs = int(block_size)
    blocks = [x[i:i + bs] for i in range(0, n - bs + 1, bs)]
    bm = np.array([b.mean() for b in blocks])
    if mode == "block":
        k = bm.size // 2
        return bm[0:2 * k:2], bm[1:2 * k:2], "consecutive " + str(bs) + "-trial block means"
    if mode == "half":
        h = bm.size // 2
        return bm[:h], bm[h:2 * h], "first against second half, " + str(bs) + "-trial block means"
    raise ValueError("unknown pairing mode: " + str(mode))


def loa_vs_navg(x, k_max=None, n_rep=1000, seed=42):
    """How repeatable is a k-trial average, for k = 1 upward.

    For each k the trials are split at random into two disjoint sets of k and the
    two means are differenced. The resulting limits say how far apart two
    independent k-trial averages of the same condition can fall, which is the
    number that should drive how many stimuli a protocol delivers.
    """
    if pd is None:
        raise RuntimeError("loa_vs_navg requires pandas.")
    x = np.asarray(x, float)
    n = x.size
    if k_max is None:
        k_max = n // 2
    k_max = int(min(k_max, n // 2))
    if k_max < 1:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((int(n_rep), n)), axis=1)
    csum = np.cumsum(x[perms], axis=1)
    m = float(np.mean(x))

    rows = []
    for k in range(1, k_max + 1):
        first = csum[:, k - 1] / k
        second = (csum[:, 2 * k - 1] - csum[:, k - 1]) / k
        d = first - second
        sdd = float(d.std(ddof=1))
        rows.append({
            "k_trials_averaged": k,
            "bias": float(d.mean()),
            "sd_diff": sdd,
            "loa_lo": -1.96 * sdd,
            "loa_hi": 1.96 * sdd,
            "loa_width": 2 * 1.96 * sdd,
            "loa_width_pct_of_mean": 100.0 * 2 * 1.96 * sdd / m if m else np.nan,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Intraclass correlation
# ─────────────────────────────────────────────────────────────────────────────
def icc_from_matrix(Y, alpha=0.05):
    """ICC forms from a targets (rows) by measurements (columns) matrix.

    Follows Shrout & Fleiss (1979) and McGraw & Wong (1996). ICC(2,1) is
    absolute agreement and ICC(3,1) is consistency; the k-forms describe the
    reliability of the mean of k measurements rather than of a single one.
    """
    _require_scipy("scipy.stats")
    Y = np.asarray(Y, float)
    if Y.ndim != 2:
        return {"error": "ICC needs a 2-D targets by measurements matrix."}
    n, k = Y.shape
    if n < 2 or k < 2:
        return {"error": "ICC needs at least 2 targets and 2 measurements."}
    if not np.all(np.isfinite(Y)):
        return {"error": "ICC needs a complete matrix with no missing cells."}

    grand = Y.mean()
    row_m = Y.mean(axis=1)
    col_m = Y.mean(axis=0)

    SST = float(np.sum((Y - grand) ** 2))
    SSR = float(k * np.sum((row_m - grand) ** 2))
    SSC = float(n * np.sum((col_m - grand) ** 2))
    SSE = SST - SSR - SSC
    SSW = SST - SSR

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    MSW = SSW / (n * (k - 1))

    out = {"n_targets": n, "k_measures": k,
           "MSR": MSR, "MSC": MSC, "MSE": MSE, "MSW": MSW}

    d1 = MSR + (k - 1) * MSW
    out["ICC1"] = (MSR - MSW) / d1 if d1 else np.nan
    d2 = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    icc2 = (MSR - MSE) / d2 if d2 else np.nan
    out["ICC2"] = icc2
    d3 = MSR + (k - 1) * MSE
    out["ICC3"] = (MSR - MSE) / d3 if d3 else np.nan
    out["ICC1k"] = (MSR - MSW) / MSR if MSR else np.nan
    dk = MSR + (MSC - MSE) / n
    out["ICC2k"] = (MSR - MSE) / dk if dk else np.nan
    out["ICC3k"] = (MSR - MSE) / MSR if MSR else np.nan

    if MSW > 0:
        F = MSR / MSW
        df1, df2 = n - 1, n * (k - 1)
        FL = F / _st.f.ppf(1 - alpha / 2.0, df1, df2)
        FU = F * _st.f.ppf(1 - alpha / 2.0, df2, df1)
        out["ICC1_ci"] = (float((FL - 1) / (FL + k - 1)), float((FU - 1) / (FU + k - 1)))

    if MSE > 0:
        F = MSR / MSE
        df1, df2 = n - 1, (n - 1) * (k - 1)
        FL = F / _st.f.ppf(1 - alpha / 2.0, df1, df2)
        FU = F * _st.f.ppf(1 - alpha / 2.0, df2, df1)
        out["ICC3_ci"] = (float((FL - 1) / (FL + k - 1)), float((FU - 1) / (FU + k - 1)))

        if np.isfinite(icc2) and icc2 not in (1.0,):
            denom = n * (1.0 - icc2)
            if denom != 0:
                a_ = k * icc2 / denom
                b_ = 1.0 + k * icc2 * (n - 1) / denom
                v_num = (a_ * MSC + b_ * MSE) ** 2
                v_den = ((a_ * MSC) ** 2 / (k - 1)
                         + (b_ * MSE) ** 2 / ((n - 1) * (k - 1)))
                if v_den > 0 and np.isfinite(v_num):
                    v = v_num / v_den
                    F3U = _st.f.ppf(1 - alpha / 2.0, n - 1, v)
                    F3L = _st.f.ppf(1 - alpha / 2.0, v, n - 1)
                    L = (n * (MSR - F3U * MSE)
                         / (F3U * (k * MSC + (k * n - k - n) * MSE) + n * MSR))
                    U = (n * (F3L * MSR - MSE)
                         / (k * MSC + (k * n - k - n) * MSE + n * F3L * MSR))
                    out["ICC2_ci"] = (float(L), float(U))

    sd_total = float(np.std(Y.ravel(), ddof=1))
    if np.isfinite(icc2):
        out["SEM"] = sd_total * np.sqrt(max(1.0 - icc2, 0.0))
        out["MDC95"] = out["SEM"] * 1.96 * np.sqrt(2.0)
    return out


def spearman_brown_k(icc1, target):
    """Measurements needed to raise a single-measurement ICC to `target`."""
    if not np.isfinite(icc1) or icc1 <= 0 or icc1 >= 1 or target >= 1:
        return np.nan
    return float(target * (1 - icc1) / (icc1 * (1 - target)))


def block_drift_icc(x, block_size=5):
    """Within-recording drift index, deliberately NOT called a reliability ICC.

    Blocks of consecutive trials are the targets and trials within a block are
    interchangeable measurements. With one participant there is no between-
    participant variance, so this is not test-retest reliability and must never
    be reported as such. Read it as drift: near zero means blocks are
    exchangeable and the series is stationary, while a high value means
    amplitude shifted systematically across the session.
    """
    x = np.asarray(x, float)
    bs = int(block_size)
    n_blocks = x.size // bs
    if n_blocks < 2:
        return {"error": "at least 2 blocks of " + str(bs) + " trials are needed."}
    Y = x[:n_blocks * bs].reshape(n_blocks, bs)
    out = icc_from_matrix(Y)
    if "error" in out:
        return out
    out["block_size"] = bs
    out["n_blocks"] = n_blocks
    out["block_means"] = Y.mean(axis=1)
    out["drift_index"] = out.get("ICC1", np.nan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Variance components and reliability of an averaged measurement
# ─────────────────────────────────────────────────────────────────────────────
def variance_components(records):
    """Nested random-effects decomposition for one event type.

        y_ijk = mu + a_i + b_ij + e_ijk

    with a_i a target effect (participant, or participant and limb), b_ij a
    session nested within target, and e_ijk trial-level noise. Components come
    from the ANOVA method of moments with Searle's coefficients, so unequal
    trial counts and unequal numbers of sessions per participant are handled.

    `records` is a sequence of dicts with keys 'target', 'session' and 'x',
    where x is that recording's array of trial values. Repeated recordings of
    the same target and session are pooled.

    Negative component estimates are truncated at zero and reported in
    'clamped', because a negative variance means the design cannot separate that
    level from noise rather than that the variance is genuinely below zero.
    """
    cell_x = {}
    for r in records:
        vals = np.asarray(r["x"], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        key = (str(r["target"]), str(r["session"]))
        cell_x.setdefault(key, []).append(vals)
    cell_x = {k: np.concatenate(v) for k, v in cell_x.items()}
    if not cell_x:
        return {"error": "no usable trials."}

    targets = sorted(set(k[0] for k in cell_x))
    n_targets = len(targets)
    n_cells = len(cell_x)
    N = int(sum(v.size for v in cell_x.values()))
    if n_targets < 2:
        return {"error": "variance components need at least 2 targets."}

    all_vals = np.concatenate(list(cell_x.values()))
    grand = float(all_vals.mean())

    n_ij = {k: v.size for k, v in cell_x.items()}
    n_i = {}
    for (t, s), v in cell_x.items():
        n_i[t] = n_i.get(t, 0) + v.size

    cell_mean = {k: float(v.mean()) for k, v in cell_x.items()}
    target_mean = {}
    for t in targets:
        tot = sum(cell_mean[k] * n_ij[k] for k in cell_x if k[0] == t)
        target_mean[t] = tot / n_i[t]

    SS_tar = sum(n_i[t] * (target_mean[t] - grand) ** 2 for t in targets)
    SS_ses = sum(n_ij[k] * (cell_mean[k] - target_mean[k[0]]) ** 2 for k in cell_x)
    SS_res = sum(float(np.sum((v - cell_mean[k]) ** 2)) for k, v in cell_x.items())

    df_tar = n_targets - 1
    df_ses = n_cells - n_targets
    df_res = N - n_cells

    MS_tar = SS_tar / df_tar if df_tar > 0 else np.nan
    MS_ses = SS_ses / df_ses if df_ses > 0 else np.nan
    MS_res = SS_res / df_res if df_res > 0 else np.nan

    sum_nij2_over_ni = sum(
        sum(n_ij[k] ** 2 for k in cell_x if k[0] == t) / float(n_i[t]) for t in targets)
    sum_nij2_over_N = sum(v ** 2 for v in n_ij.values()) / float(N)
    sum_ni2_over_N = sum(v ** 2 for v in n_i.values()) / float(N)

    clamped = []
    s2_e = float(MS_res) if np.isfinite(MS_res) else np.nan
    s2_b = np.nan
    if df_ses > 0 and np.isfinite(MS_ses) and np.isfinite(s2_e):
        c1 = (N - sum_nij2_over_ni) / df_ses
        if c1 > 0:
            raw = (MS_ses - MS_res) / c1
            if raw < 0:
                clamped.append("session")
            s2_b = max(float(raw), 0.0)

    s2_a = np.nan
    if df_tar > 0 and np.isfinite(MS_tar) and np.isfinite(s2_e):
        c3 = (N - sum_ni2_over_N) / df_tar
        if c3 > 0:
            if df_ses > 0 and np.isfinite(s2_b):
                c2 = (sum_nij2_over_ni - sum_nij2_over_N) / df_tar
                raw = (MS_tar - MS_res - c2 * s2_b) / c3
            else:
                s2_b = 0.0
                raw = (MS_tar - MS_res) / c3
            if raw < 0:
                clamped.append("target")
            s2_a = max(float(raw), 0.0)

    total = float(np.nansum([s2_a, s2_b, s2_e]))
    out = {
        "n_targets": n_targets, "n_cells": n_cells, "n_trials": N,
        "grand_mean": grand,
        "var_target": s2_a, "var_session": s2_b, "var_trial": s2_e,
        "var_total": total,
        "pct_target": 100.0 * s2_a / total if total > 0 else np.nan,
        "pct_session": 100.0 * s2_b / total if total > 0 else np.nan,
        "pct_trial": 100.0 * s2_e / total if total > 0 else np.nan,
        "sessions_per_target": n_cells / float(n_targets),
        "has_session_level": df_ses > 0,
        "clamped": ", ".join(clamped),
    }
    out["icc_single_trial"] = (out["pct_target"] / 100.0
                               if np.isfinite(out["pct_target"]) else np.nan)
    return out


def reliability_table(vc, trials=(5, 10, 15, 20, 30), sessions=(1, 2)):
    """Reliability, SEM and MDC95 of a measurement averaged over trials/sessions.

    A generalisability-style D study. Reliability of the mean of `k` trials in
    each of `m` sessions is

        var_target / (var_target + var_session / m + var_trial / (k * m))

    which makes explicit why adding trials cannot fix session-level variance:
    once var_trial/(k*m) is small, only more sessions move the number.
    """
    if pd is None:
        raise RuntimeError("reliability_table requires pandas.")
    if not vc or "error" in vc:
        return pd.DataFrame()
    sa = vc.get("var_target", np.nan)
    sb = vc.get("var_session", 0.0) or 0.0
    se = vc.get("var_trial", np.nan)
    mean = vc.get("grand_mean", np.nan)
    if not np.isfinite(sa) or not np.isfinite(se):
        return pd.DataFrame()

    rows = []
    for m in sessions:
        for k in trials:
            err = sb / float(m) + se / float(k * m)
            denom = sa + err
            rel = sa / denom if denom > 0 else np.nan
            sem = np.sqrt(err)
            mdc = sem * 1.96 * np.sqrt(2.0)
            rows.append({
                "sessions": int(m),
                "trials_per_session": int(k),
                "reliability": float(rel),
                "SEM": float(sem),
                "SEM_pct_of_mean": 100.0 * sem / mean if mean else np.nan,
                "MDC95": float(mdc),
                "MDC95_pct_of_mean": 100.0 * mdc / mean if mean else np.nan,
            })
    return pd.DataFrame(rows)


def trials_for_reliability(vc, target_rel, sessions=1):
    """Trials per session needed to reach a target reliability.

    Returns NaN when the target is unreachable at that number of sessions, which
    happens whenever session-level variance alone already puts the ceiling below
    the target. That is a design finding, not a failure.
    """
    if not vc or "error" in vc:
        return np.nan
    sa = vc.get("var_target", np.nan)
    sb = vc.get("var_session", 0.0) or 0.0
    se = vc.get("var_trial", np.nan)
    if not np.isfinite(sa) or not np.isfinite(se) or sa <= 0 or target_rel >= 1:
        return np.nan
    m = float(sessions)
    # solve sa / (sa + sb/m + se/(k*m)) = target  for k
    allowed_err = sa * (1.0 - target_rel) / target_rel
    remaining = allowed_err - sb / m
    if remaining <= 0:
        return np.nan                      # ceiling already below target
    return float(np.ceil(se / (remaining * m)))


# ─────────────────────────────────────────────────────────────────────────────
# Study design interpretation
# ─────────────────────────────────────────────────────────────────────────────
def classify_design_factors(df, factors, target_col="participant_id",
                            session_col="session"):
    """Label each design factor as between-target, within-target, or constant.

    This decides how a factor may be used. A between-target factor (group, arm,
    sex) can be stratified on: each participant keeps all of their sessions, so
    between-session reliability survives inside every stratum. A within-target
    factor (timepoint, pre/post) is the repeated-measures axis itself, so
    stratifying on it would leave one session per participant and destroy the
    very comparison it describes; such a factor is used to LABEL the session
    axis instead.
    """
    out = {}
    if pd is None or df is None or len(df) == 0:
        return out
    for f in factors:
        if f not in df.columns:
            out[f] = {"role": "absent", "n_levels": 0,
                      "reason": "not a column in the group table"}
            continue
        levels = df[f].dropna().unique()
        if len(levels) < 2:
            out[f] = {"role": "constant", "n_levels": int(len(levels)),
                      "reason": "only one level present"}
            continue
        if target_col not in df.columns:
            out[f] = {"role": "unknown", "n_levels": int(len(levels)),
                      "reason": "no target column to test against"}
            continue
        per_target = df.groupby(target_col)[f].nunique(dropna=True)
        varies_within = bool((per_target > 1).any())
        info = {"n_levels": int(len(levels)),
                "levels": [str(v) for v in sorted(map(str, levels))]}
        if not varies_within:
            info["role"] = "between_target"
            info["reason"] = "each target sits in a single level"
        else:
            info["role"] = "within_target"
            info["reason"] = "varies within a target, so it is a repeated-measures axis"
            if session_col in df.columns:
                pair = df.groupby([target_col, session_col])[f].nunique(dropna=True)
                info["aligned_with_session"] = bool((pair <= 1).all())
        out[f] = info
    return out


def session_pair_role(df, session_a, session_b, within_factor=None,
                      session_col="session"):
    """Say whether a session pair measures reliability or measures change.

    Two sessions that share a level of the within-target factor (both baseline,
    or a familiarisation and a baseline) differ only by measurement error, so
    their agreement IS test-retest reliability. Two sessions that straddle an
    intervention differ by error PLUS whatever the intervention did, so the same
    arithmetic no longer estimates reliability and will look poor even when the
    measurement is excellent. The caller should label the output accordingly
    rather than let a reader assume the reliable case.
    """
    if pd is None or df is None:
        return {"role": "unknown", "reason": "no group table to classify against"}
    if within_factor is None or within_factor not in df.columns:
        # No design factor varies within a target, so nothing in the design says
        # these two sessions were meant to differ. Treat them as repeat
        # measurements, but say that this is an assumption: an intervention that
        # was never recorded as a factor is invisible here.
        return {"role": "reliability_assumed",
                "reason": ("no design factor distinguishes these sessions, so they "
                           "are treated as repeat measurements; if something was "
                           "meant to change between visits, record it as a "
                           "Second Level design factor so it can be accounted for")}
    lv = {}
    for s in (session_a, session_b):
        vals = df.loc[df[session_col].astype(str) == str(s), within_factor].dropna().unique()
        lv[s] = set(map(str, vals))
    shared = lv[session_a] & lv[session_b]
    if shared and len(lv[session_a]) == 1 and len(lv[session_b]) == 1:
        return {"role": "reliability",
                "reason": "both sessions share the level '" + list(shared)[0]
                          + "' of " + str(within_factor),
                "levels": {str(k): sorted(v) for k, v in lv.items()}}
    return {"role": "change",
            "reason": "the sessions differ on " + str(within_factor)
                      + ", so their agreement reflects measurement error plus real change",
            "levels": {str(k): sorted(v) for k, v in lv.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# One-call summary for a single condition within one recording
# ─────────────────────────────────────────────────────────────────────────────
def summarise_condition(x, block_size=5, pairing="single", n_boot=2000,
                        n_mc=1000, seed=42, robust_z_thresh=3.5):
    """Every within-recording measure for one condition, as a flat dict.

    Returned keys are generic; the add-on maps them onto its output column
    names. Sections that the trial count cannot support are simply absent rather
    than filled with placeholder values.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    res = {"n_trials": int(x.size)}
    if x.size < 3:
        res["note"] = "too few trials for a variability summary"
        return res

    res.update(descriptives(x))
    res.update(cv_analysis(x, n_boot=n_boot, seed=seed))

    zr = z_scores(x, robust_thresh=robust_z_thresh)
    res["z_robust"] = zr.get("z_robust")
    res["z_log"] = zr.get("z_log")
    res["n_extreme_robust_z"] = zr.get("n_extreme_robust_z")
    res["max_abs_robust_z"] = zr.get("max_abs_robust_z")

    ar = ar_analysis(x)
    if "error" not in ar:
        res["acf_lag1"] = ar["r1"]
        res["ar_order"] = ar["best_order"]
        res["ar_phi1"] = ar["ar_phi1"]
        res["ar_phi1_p"] = ar["ar_phi1_p"]
        res["durbin_watson"] = ar["durbin_watson"]
        res["ljung_box_p"] = ar["ljung_box_raw"]["p"]
        res["trend_slope_per_trial"] = ar["trend_slope"]
        res["trend_p"] = ar["trend_p"]
        res["trend_pct_per_10_trials"] = ar["trend_pct_per_10_trials"]
        res["_ar"] = ar

    res.update(ci_analysis(x, r1=res.get("acf_lag1"), n_boot=n_boot, seed=seed))
    res.update(trials_for_precision(res.get("cv_percent", np.nan)))

    if x.size >= 4:
        a, b, pair_label = make_pairs(x, mode=pairing, block_size=block_size)
        if a.size >= 2:
            ba = bland_altman(a, b)
            if "error" not in ba:
                res["loa_pairing"] = pair_label
                res["loa_n_pairs"] = ba["n_pairs"]
                res["loa_bias"] = ba["bias"]
                res["loa_lo"] = ba["loa_lo"]
                res["loa_hi"] = ba["loa_hi"]
                res["loa_width_pct_of_mean"] = ba["loa_width_pct_of_mean"]
                res["loa_prop_bias_p"] = ba["prop_bias_p"]
                res["_ba"] = ba
            bar = bland_altman_ratio(a, b)
            if bar:
                res["ratio_loa_lo"] = bar["ratio_loa_lo"]
                res["ratio_loa_hi"] = bar["ratio_loa_hi"]
                res["_bar"] = bar

    if pd is not None and x.size >= 4:
        res["_loa_curve"] = loa_vs_navg(x, n_rep=n_mc, seed=seed)

    drift = block_drift_icc(x, block_size=block_size)
    if "error" not in drift:
        res["block_drift_icc1"] = drift["drift_index"]
        ci = drift.get("ICC1_ci", (np.nan, np.nan))
        res["block_drift_icc1_lo"] = ci[0]
        res["block_drift_icc1_hi"] = ci[1]
        res["n_blocks"] = drift["n_blocks"]
        res["_drift"] = drift

    res["cumulative_mean"] = np.cumsum(x) / np.arange(1.0, x.size + 1.0)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Typical error and the RMSE family
# ─────────────────────────────────────────────────────────────────────────────
def typical_error(x):
    """Typical error of measurement from odd against even trials.

    The within-subject standard deviation, SD of the paired differences divided
    by root two (Hopkins 2000). Interpreted as the noise attached to a SINGLE
    measurement, which is what makes it the natural companion to the limits of
    agreement: LoA are roughly 2.77 times this.

    Within one recording the pairing is odd against even trials, so this is the
    trial-to-trial typical error, not the test-retest typical error. The latter
    needs two sessions and comes from the group-level add-on.

    One caveat that matters in practice. Odd-even pairing pairs ADJACENT trials,
    so serial dependence leaks straight into the estimate. With negative lag-1
    autocorrelation, successive trials differ more than randomly chosen ones do,
    the differences are inflated, and the typical error can come out LARGER than
    the plain SD. That is a real signal about the series rather than an error:
    check the lag-1 autocorrelation before quoting the number, and prefer the
    limits of agreement between independent k-trial averages when it happens.
    """
    x = np.asarray(x, float)
    n = x.size
    k = n // 2
    if k < 2:
        return np.nan
    d = x[0:2 * k:2] - x[1:2 * k:2]
    return float(np.std(d, ddof=1) / np.sqrt(2.0))


def rmse_family(x, ar=None):
    """Every quantity that gets called RMSE in this literature, kept separate.

    They answer different questions and are routinely confused:

      * rmse_trial_about_mean — spread of single trials about the condition
        mean. Equals the sample SD up to the n versus n-1 divisor.
      * typical_error — noise on a single measurement, from paired trials.
        Smaller than the SD because it is not inflated by real between-trial
        differences that the pairing cancels.
      * ar_resid_rmse — how well the previous trials predict the next one. If
        this is meaningfully below ar0_rmse then the series carries structure
        and trials are not exchangeable.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return {}
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))

    out = {
        "rmse_trial_about_mean": float(np.sqrt(np.mean((x - m) ** 2))),
        "sd_sample": sd,
    }
    out["rmse_pct_of_mean"] = 100.0 * out["rmse_trial_about_mean"] / m if m else np.nan

    te = typical_error(x)
    out["typical_error"] = te
    out["typical_error_pct_of_mean"] = 100.0 * te / m if (m and np.isfinite(te)) else np.nan
    if np.all(x > 0):
        te_log = typical_error(np.log(x))
        if np.isfinite(te_log):
            # Back-transformed, a typical error on the log scale is a percentage
            # error that scales with amplitude, which is how MEP noise behaves.
            out["typical_error_cv_percent"] = float(100.0 * (np.exp(te_log) - 1.0))

    if ar and "best_fit" in ar and "error" not in ar:
        bf = ar["best_fit"]
        resid = np.asarray(bf["resid"], float)
        out["ar_resid_rmse"] = float(np.sqrt(np.mean(resid ** 2)))
        base = (x - m)[int(ar["p_max"]):]
        out["ar0_rmse"] = float(np.sqrt(np.mean(base ** 2)))
        out["ar_rmse_reduction_pct"] = (
            100.0 * (1.0 - out["ar_resid_rmse"] / out["ar0_rmse"])
            if out["ar0_rmse"] else np.nan)
    return out


def trials_to_average_table(x, k_max=None, n_rep=1000, seed=42):
    """How good is a k-trial average, for k = 1 upward.

    Three columns that answer the same planning question from different angles:

      * SEM_Of_K_Mean — the analytic standard error, SD / sqrt(k). Assumes
        independent trials, so it is optimistic when the series is serially
        dependent.
      * RMSE_Of_K_Mean — Monte Carlo root mean square deviation of a k-trial
        average from this recording's own mean. Because the subsets are drawn
        without replacement from a finite set of trials, this necessarily
        shrinks to zero as k approaches n; read it for small k, not large.
      * LoA_Width(%) — the width of the 95% limits between two INDEPENDENT
        k-trial averages, which is the honest answer for "how far apart could
        two runs of my protocol land".
    """
    if pd is None:
        raise RuntimeError("trials_to_average_table requires pandas.")
    x = np.asarray(x, float)
    n = x.size
    if k_max is None:
        k_max = n // 2
    k_max = int(min(k_max, n // 2))
    if k_max < 1:
        return pd.DataFrame()

    loa = loa_vs_navg(x, k_max=k_max, n_rep=n_rep, seed=seed)
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))

    rng = np.random.default_rng(seed + 1)
    perms = np.argsort(rng.random((int(n_rep), n)), axis=1)
    csum = np.cumsum(x[perms], axis=1)

    rows = []
    for k in range(1, k_max + 1):
        sub_means = csum[:, k - 1] / k
        rmse = float(np.sqrt(np.mean((sub_means - m) ** 2)))
        sem = sd / np.sqrt(k)
        rows.append({
            "k_trials_averaged": k,
            "SEM_Of_K_Mean": float(sem),
            "SEM_Pct_Of_Mean": 100.0 * sem / m if m else np.nan,
            "RMSE_Of_K_Mean": rmse,
            "RMSE_Pct_Of_Mean": 100.0 * rmse / m if m else np.nan,
        })
    tbl = pd.DataFrame(rows)
    return tbl.merge(loa[["k_trials_averaged", "loa_width", "loa_width_pct_of_mean"]],
                     on="k_trials_averaged", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Contrasts between conditions
# ─────────────────────────────────────────────────────────────────────────────
def hedges_g(x, y):
    """Standardised mean difference with the small-sample correction."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    sp2 = (((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
           / (nx + ny - 2))
    if sp2 <= 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / np.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    return float(d * J)


def compare_two(x, y, alpha=0.05):
    """Contrast two conditions in both level and spread.

    Level and spread are reported separately on purpose. Two conditions can
    have indistinguishable means while one is far noisier, and for a
    variability analysis that difference in spread is usually the finding.
    """
    _require_scipy("scipy.stats")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 3 or y.size < 3:
        return {"error": "at least 3 trials per condition are needed."}
    if np.ptp(x) == 0 and np.ptp(y) == 0:
        # Both conditions constant: every test below is undefined, and scipy
        # warns rather than failing, which would surface as a spurious crash.
        return {"error": "both conditions are constant; no contrast to compute."}

    mx, my = float(np.mean(x)), float(np.mean(y))
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    t_stat, t_p = _safe_test(_st.ttest_ind, x, y, equal_var=False)
    nx, ny = x.size, y.size
    se = np.sqrt(sx ** 2 / nx + sy ** 2 / ny)
    dof = (sx ** 2 / nx + sy ** 2 / ny) ** 2 / (
        (sx ** 2 / nx) ** 2 / (nx - 1) + (sy ** 2 / ny) ** 2 / (ny - 1))
    tcrit = float(_st.t.ppf(1 - alpha / 2.0, dof))

    lev_W, lev_p = _safe_test(_st.levene, x, y, center="median")  # Brown-Forsythe
    cvx = 100.0 * sx / mx if mx else np.nan
    cvy = 100.0 * sy / my if my else np.nan

    return {
        "n_a": int(nx), "n_b": int(ny),
        "mean_a": mx, "mean_b": my,
        "mean_diff": mx - my,
        "mean_diff_ci_lo": (mx - my) - tcrit * se,
        "mean_diff_ci_hi": (mx - my) + tcrit * se,
        "welch_t": t_stat, "welch_p": t_p,
        "hedges_g": hedges_g(x, y),
        "sd_a": sx, "sd_b": sy,
        "cv_a_percent": cvx, "cv_b_percent": cvy,
        "cv_ratio": cvx / cvy if cvy else np.nan,
        "levene_W": lev_W, "levene_p": lev_p,
    }


def compare_conditions(by_condition, alpha=0.05):
    """All pairwise contrasts plus an omnibus test of equal spread.

    `by_condition` maps a condition label to its array of trial values. The
    omnibus is Fligner-Killeen, which is the most robust of the common
    equal-variance tests when the data are not normal, as MEP amplitudes are
    not.
    """
    if pd is None:
        raise RuntimeError("compare_conditions requires pandas.")
    _require_scipy("scipy.stats")

    clean = {}
    for k, v in by_condition.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size >= 3:
            clean[k] = v
    labels = sorted(clean)
    if len(labels) < 2:
        return pd.DataFrame(), {}

    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            res = compare_two(clean[a], clean[b], alpha=alpha)
            if "error" in res:
                continue
            row = {"Condition_A": a, "Condition_B": b}
            row.update(res)
            rows.append(row)
    pairs = pd.DataFrame(rows)

    omni = {}
    arrays = [clean[k] for k in labels if np.ptp(clean[k]) > 0]
    if len(arrays) >= 2:
        omni["fligner_W"], omni["fligner_p"] = _safe_test(_st.fligner, *arrays)
        omni["levene_W"], omni["levene_p"] = _safe_test(
            _st.levene, *arrays, center="median")
        omni["n_conditions"] = len(arrays)
    return pairs, omni


# ─────────────────────────────────────────────────────────────────────────────
# Relationships between measures
# ─────────────────────────────────────────────────────────────────────────────
def fisher_ci(r, n, alpha=0.05):
    """Confidence interval for a correlation via the Fisher z transform."""
    _require_scipy("scipy.stats")
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = float(_st.norm.ppf(1 - alpha / 2.0))
    return (float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)))


def metric_correlations(frame, metrics, alpha=0.05):
    """Pairwise correlations among trial-level measures within one condition.

    Both Pearson and Spearman are given: Pearson answers whether the
    relationship is linear on the recorded scale, Spearman survives the skew and
    the occasional huge trial that MEP data reliably produce. A large gap
    between them is itself informative.

    The usual reason to look: pre-stimulus EMG driving MEP amplitude, which
    turns apparent trial-to-trial noise into something partly explainable.
    """
    if pd is None:
        raise RuntimeError("metric_correlations requires pandas.")
    _require_scipy("scipy.stats")

    cols = {}
    for m in metrics:
        if m in frame.columns:
            cols[m] = pd.to_numeric(frame[m], errors="coerce").to_numpy(dtype=float)
    names = [m for m in metrics if m in cols]
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ok = np.isfinite(cols[a]) & np.isfinite(cols[b])
            n = int(ok.sum())
            if n < 4:
                continue
            xa, xb = cols[a][ok], cols[b][ok]
            # A constant column has no correlation to report. scipy warns and
            # returns NaN; skipping keeps the output table clean instead.
            if np.ptp(xa) == 0 or np.ptp(xb) == 0:
                continue
            pr = _st.pearsonr(xa, xb)
            sr = _st.spearmanr(xa, xb)
            lo, hi = fisher_ci(float(pr[0]), n, alpha)
            rows.append({
                "Measure_A": a, "Measure_B": b, "N": n,
                "Pearson_r": float(pr[0]), "Pearson_p": float(pr[1]),
                "Pearson_CI_Lo": lo, "Pearson_CI_Hi": hi,
                "Spearman_rho": float(sr[0]), "Spearman_p": float(sr[1]),
                "R_Squared": float(pr[0]) ** 2,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across recordings (group level)
# ─────────────────────────────────────────────────────────────────────────────
def spread_across_recordings(arrays):
    """Do recordings differ in how variable they are, beyond chance.

    Levene (median-centred) and Fligner-Killeen on the raw values test equality
    of variance, which for data whose spread scales with the mean will almost
    always reject simply because the means differ. So the coefficient of
    variation is also tested across recordings, which is the question actually
    being asked: is one recording noisier RELATIVE to its own amplitude.
    """
    _require_scipy("scipy.stats")
    arrays = [np.asarray(a, float)[np.isfinite(np.asarray(a, float))]
              for a in arrays]
    arrays = [a for a in arrays if a.size >= 3 and np.ptp(a) > 0]
    out = {"n_recordings": len(arrays)}
    if len(arrays) < 2:
        return out

    out["levene_W"], out["levene_p"] = _safe_test(_st.levene, *arrays,
                                                  center="median")
    out["fligner_W"], out["fligner_p"] = _safe_test(_st.fligner, *arrays)

    cvs = np.array([100.0 * a.std(ddof=1) / a.mean() if a.mean() else np.nan
                    for a in arrays])
    cvs = cvs[np.isfinite(cvs)]
    if cvs.size >= 2:
        out["cv_median"] = float(np.median(cvs))
        out["cv_iqr_lo"] = float(np.percentile(cvs, 25))
        out["cv_iqr_hi"] = float(np.percentile(cvs, 75))
        out["cv_min"], out["cv_max"] = float(cvs.min()), float(cvs.max())
        # A recording whose CV is far from the rest is worth a look before it
        # is allowed to influence the variance components.
        med = float(np.median(cvs))
        mad = mad_normal(cvs)
        if mad > 0:
            rz = (cvs - med) / mad
            out["n_atypical_cv"] = int(np.sum(np.abs(rz) > 3.5))
            out["max_abs_cv_robust_z"] = float(np.max(np.abs(rz)))
    return out


def serial_structure_across_recordings(r1s, slopes_pct=None, n_trials=None):
    """Is serial dependence a property of the paradigm or of one recording.

    Two corrections make this honest.

    First, the sample lag-1 autocorrelation is biased. For a series of length n
    with no serial structure at all, its expected value is about -1/(n-1), not
    zero. With the 20-trial runs typical of TMS work that bias is -0.05, and
    testing the raw values against zero flags roughly a quarter of completely
    structureless datasets as significantly negative. Supplying `n_trials` adds
    the correction back per recording, which restores the nominal error rate.

    Second, correlations are averaged through the Fisher z transform rather than
    directly, since they do not average correctly on their own scale.

    A corrected mean that still differs from zero across many recordings is
    evidence that the ordering effect is systematic rather than one noisy
    session.
    """
    _require_scipy("scipy.stats")
    r = np.asarray([v for v in r1s if v is not None], float)
    keep = np.isfinite(r) & (np.abs(r) < 0.999)
    out = {"n_recordings": int(keep.sum())}
    if keep.sum() < 2:
        return out

    if n_trials is not None:
        nt = np.asarray([v for v in n_trials], float)
        if nt.size == r.size:
            bias = np.where(nt > 1, -1.0 / (nt - 1.0), 0.0)
            corrected = r - bias           # subtracting a negative bias adds it back
            out["bias_corrected"] = True
            out["mean_null_bias"] = float(np.mean(bias[keep]))
        else:
            corrected = r
            out["bias_corrected"] = False
    else:
        corrected = r
        out["bias_corrected"] = False

    rk, ck = r[keep], corrected[keep]
    out["r1_median"] = float(np.median(rk))
    out["r1_min"], out["r1_max"] = float(rk.min()), float(rk.max())

    ck = ck[np.abs(ck) < 0.999]
    if ck.size >= 2:
        z = np.arctanh(ck)
        tt = _st.ttest_1samp(z, 0.0)
        out["r1_mean_fisher"] = float(np.tanh(z.mean()))
        out["r1_t"] = float(tt.statistic)
        out["r1_df"] = int(ck.size - 1)
        out["r1_p"] = float(tt.pvalue)

    if slopes_pct is not None:
        sl = np.asarray([v for v in slopes_pct if v is not None], float)
        sl = sl[np.isfinite(sl)]
        if sl.size >= 2:
            out["drift_median_pct_per_10"] = float(np.median(sl))
            out["drift_min_pct_per_10"] = float(sl.min())
            out["drift_max_pct_per_10"] = float(sl.max())
            tt = _st.ttest_1samp(sl, 0.0)
            out["drift_t"] = float(tt.statistic)
            out["drift_p"] = float(tt.pvalue)
    return out


def pooled_trials_to_average(tables):
    """Median and interquartile range of the trials-to-average curves.

    Pooling across recordings gives the number a protocol should be designed
    around, rather than the number one participant happened to produce.
    """
    if pd is None:
        raise RuntimeError("pooled_trials_to_average requires pandas.")
    tables = [t for t in tables if t is not None and len(t)]
    if not tables:
        return pd.DataFrame()
    allt = pd.concat(tables, ignore_index=True)
    value_cols = [c for c in ("SEM_Pct_Of_Mean", "RMSE_Pct_Of_Mean",
                              "loa_width_pct_of_mean") if c in allt.columns]
    if not value_cols:
        return pd.DataFrame()
    g = allt.groupby("k_trials_averaged")[value_cols]
    out = g.median().add_suffix("_median")
    out = out.join(g.quantile(0.25).add_suffix("_q25"))
    out = out.join(g.quantile(0.75).add_suffix("_q75"))
    out["n_recordings"] = g.size()
    return out.reset_index()
