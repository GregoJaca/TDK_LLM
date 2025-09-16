import numpy as np
from scipy.optimize import curve_fit


def exponential(t, a, lamb, c):
    return a * np.exp(lamb * t) + c


FITTERS = {
    "exponential": exponential,
}


def fit_timeseries(times, series, fitter_name="exponential", use_log=False):
    f = FITTERS.get(fitter_name)
    if f is None:
        raise ValueError(f"Unknown fitter: {fitter_name}")

    if use_log:
        # fit on log if requested
        y = np.log(series + 1e-12)
        # linear fit
        coeffs = np.polyfit(times, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        return {"slope": slope, "intercept": intercept, "params": coeffs}

    # initial guesses
    p0 = [series.max() - series.min() + 1e-6, 0.1, series.min()]
    try:
        popt, pcov = curve_fit(f, times, series, p0=p0, maxfev=20000)
    except Exception:
        popt = p0
        pcov = None

    return {"params": popt, "cov": pcov}
