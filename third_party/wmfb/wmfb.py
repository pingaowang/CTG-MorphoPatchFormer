"""WMFB -- Weighted Median Filter Baseline (Boudet et al., 2019).

Faithful Python port of ``aamwmfb.m`` from the FHRMA toolbox
(https://github.com/data-coeur/FHRMA).  The reference implementation assumes
FHR sampled at **4 Hz** (``srate=240`` samples per minute in the original code).

For 1 Hz project data we upsample to 4 Hz before analysis, then map events back
to the native sampling grid.

This port is distributed separately under GPL-3.0; see COPYING and NOTICE.md.
The main package's research-only license does not apply to this file. The
dataset-preprocessing wrapper is deliberately omitted from this source export.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, decimate, filtfilt


@dataclass(frozen=True)
class WmfbResult:
    baseline: np.ndarray
    accelerations_sec: np.ndarray  # shape (2, n_acc) start/end in seconds
    decelerations_sec: np.ndarray  # shape (2, n_dec)
    fs_hz: float


def upsample_fhr_to_4hz(fhr: np.ndarray, *, fs_in_hz: float = 1.0) -> tuple[np.ndarray, float]:
    """Upsample FHR to 4 Hz for WMFB (linear interpolation)."""
    x = np.asarray(fhr, dtype=np.float64).reshape(-1)
    if fs_in_hz == 4.0:
        return x, 4.0
    if fs_in_hz != 1.0:
        raise ValueError(f"WMFB upsample supports fs_in_hz in {{1.0, 4.0}}, got {fs_in_hz}")
    if x.size <= 1:
        return x.copy(), 4.0
    xi = np.arange(x.size, dtype=np.float64)
    xo = np.linspace(0.0, x.size - 1.0, x.size * 4)
    return np.interp(xo, xi, x).astype(np.float64), 4.0


def events_to_dense_labels(
    length: int,
    accelerations_sec: np.ndarray,
    decelerations_sec: np.ndarray,
    *,
    fs_hz: float = 1.0,
) -> np.ndarray:
    """Map WMFB event tables to dense labels ``0=background, 1=acc, 2=dec``."""
    y = np.zeros(int(length), dtype=np.int64)
    acc = np.asarray(accelerations_sec, dtype=np.float64)
    dec = np.asarray(decelerations_sec, dtype=np.float64)

    def _paint(table: np.ndarray, cls: int) -> None:
        if table.size == 0:
            return
        if table.ndim == 1:
            return
        if table.shape[0] != 2 and table.shape[1] == 2:
            table = table.T
        for j in range(table.shape[1]):
            s = int(np.floor(float(table[0, j]) * fs_hz))
            e = int(np.ceil(float(table[1, j]) * fs_hz))
            s = max(0, min(s, length - 1))
            e = max(s, min(e, length - 1))
            y[s : e + 1] = cls

    _paint(dec, 2)
    _paint(acc, 1)
    return y


def _butterfilt(
    data: np.ndarray,
    srate: float,
    f1: float,
    f2: float,
    order: int = 6,
    zero_phase: bool = True,
) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if f1 > 0 and f2 == 0:
        b, a = butter(order, 2.0 * f1 / srate, btype="high")
    elif f1 == 0 and f2 > 0:
        b, a = butter(order, 2.0 * f2 / srate, btype="low")
    elif f1 > 0 and f2 > 0 and f2 < f1:
        b, a = butter(order, [2.0 * f2 / srate, 2.0 * f1 / srate], btype="bandstop")
    elif f1 > 0 and f2 > 0 and f2 > f1:
        b, a = butter(order, [2.0 * f1 / srate, 2.0 * f2 / srate], btype="band")
    else:
        return x.copy()
    if zero_phase:
        return filtfilt(b, a, x)
    from scipy.signal import lfilter

    return lfilter(b, a, x)


def _enveloppe(x: np.ndarray, srate: float, f0: float, f1: float) -> np.ndarray:
    sig = np.asarray(x, dtype=np.float64).reshape(-1)
    n = sig.size
    siglen = n / float(srate)
    firstsamp = int(round(f0 * siglen + 1))
    lastsamp = int(round(f1 * siglen + 1))
    firstsamp = max(1, min(firstsamp, n))
    lastsamp = max(firstsamp, min(lastsamp, n))
    fftx = np.fft.fft(sig)
    ffty = np.zeros(n, dtype=np.complex128)
    band = fftx[firstsamp - 1 : lastsamp]
    m = lastsamp - firstsamp
    if band.size == 0:
        return np.zeros(n, dtype=np.float64)
    if m % 2 == 1:
        lo = (m - 3) // 2
        hi = (3 + m) // 2
        ffty[n - lo - 1 :] = band[: lo + 1]
        ffty[:hi] = band[lo + 1 : lo + 1 + hi]
    else:
        lo = (m - 2) // 2
        hi = (2 + m) // 2
        ffty[n - lo - 1 :] = band[: lo + 1]
        ffty[:hi] = band[lo + 1 : lo + 1 + hi]
    return 2.0 * np.abs(np.fft.ifft(ffty))


def _interp_upsample(y_dec: np.ndarray, decim: int, out_len: int) -> np.ndarray:
    yd = np.asarray(y_dec, dtype=np.float64).reshape(-1)
    if decim <= 1 or yd.size == 0:
        return yd[:out_len]
    x_old = np.arange(yd.size, dtype=np.float64)
    x_new = np.linspace(0.0, yd.size - 1.0, out_len)
    return np.interp(x_new, x_old, yd)


def _medgliss(
    x: np.ndarray,
    win: np.ndarray,
    coef: np.ndarray,
    decim: int,
    srate: float,
    x2: np.ndarray | None = None,
    p2: np.ndarray | None = None,
    c: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xd = _butterfilt(x, srate, 0.0, srate / 2.2 / decim, 8, True)
    xd = xd[::decim]
    if x2 is not None:
        x2d = x2[::decim]
    else:
        x2d = None
    coefd = decimate(np.asarray(coef, dtype=np.float64), decim, ftype="fir", zero_phase=False)
    coefd = coefd * (coefd > 0)
    midwin = (len(win) - 1) // 2
    yd = np.zeros(xd.shape[0], dtype=np.float64)
    mp = np.zeros(xd.shape[0], dtype=np.float64)

    mintolerated = np.zeros(xd.shape[0], dtype=np.float64)
    maxtolerated = np.full(xd.shape[0], 255.0, dtype=np.float64)
    step = int(round(srate / 2.0 / decim))
    twin_len = int(round(10.0 * srate / decim))
    for i in range(0, max(0, xd.shape[0] - twin_len), max(1, step)):
        twin = slice(i, min(i + twin_len, xd.shape[0]))
        mi = float(np.min(xd[twin]))
        ma = float(np.max(xd[twin]))
        idx = np.arange(i, min(i + twin_len, xd.shape[0]))
        mintolerated[idx[mintolerated[idx] <= mi]] = mi
        maxtolerated[idx[maxtolerated[idx] <= ma]] = ma

    for i in range(xd.shape[0]):
        if i - 1 < midwin:
            mwm = i
            mwp = min(max(mwm, (midwin - mwm) // 2), xd.shape[0] - i - 1)
        elif xd.shape[0] - i - 1 < midwin:
            mwp = xd.shape[0] - i - 1
            mwm = min(max(mwp, (midwin - mwp) // 2), i)
        else:
            mwm = midwin
            mwp = midwin
        points = np.arange(i - mwm, i + mwp + 1)
        xpoints = xd[points]
        win_slice = win[midwin - mwm : midwin + mwp + 1]
        coefwin = (
            coefd[points]
            * win_slice
            * ((xpoints >= mintolerated[i]) & (xpoints <= maxtolerated[i]))
        )
        s = float(np.sum(coefwin))
        scoef = float(np.sum(win_slice))

        if x2d is not None and p2 is not None and c is not None:
            xpoints = np.concatenate([xpoints, [x2d[i]]])
            extra = max(0.0, c * p2[i] * scoef - s)
            coefwin = np.concatenate([coefwin, [extra]])
            s = float(np.sum(coefwin))

        order = np.argsort(xpoints)
        p_sorted = xpoints[order]
        cw_sorted = coefwin[order]
        cum = np.cumsum(cw_sorted)
        pick = int(np.searchsorted(cum, s / 2.0, side="left"))
        pick = min(pick, p_sorted.size - 1)
        yd[i] = p_sorted[pick]
        mp[i] = s / scoef if scoef > 0 else 0.0

    y = _interp_upsample(yd, decim, x.shape[0])
    mp_up = _interp_upsample(mp, decim, x.shape[0])
    return y, mp_up


def _accidentcandidat(s1: np.ndarray, s2: np.ndarray, seuil: float) -> np.ndarray:
    diff = s1 - s2
    binsig = np.concatenate([[0.0], (diff > seuil).astype(np.float64), [0.0]])
    starts = np.where((binsig[1:] > 0) & (binsig[:-1] == 0))[0]
    ends = np.where((binsig[1:] == 0) & (binsig[:-1] > 0))[0] - 1
    if starts.size == 0:
        return np.zeros((3, 0), dtype=np.int64)
    n = min(starts.size, ends.size)
    starts = starts[:n]
    ends = ends[:n]
    peaks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        seg = diff[starts[i] : ends[i] + 1]
        peaks[i] = starts[i] + int(np.argmax(seg))
    return np.vstack([starts, ends, peaks])


def _adjustduration(startendlist: np.ndarray, s: np.ndarray, *, min_dur: int) -> np.ndarray:
    if startendlist.size == 0:
        return startendlist
    lst = np.asarray(startendlist, dtype=np.int64)
    if lst.shape[0] != 3:
        return lst
    out_cols: list[np.ndarray] = []
    i = 0
    while i < lst.shape[1]:
        col = lst[:, i].copy()
        s_arr = np.asarray(s, dtype=np.float64).reshape(-1)
        newend = col[2] - 1
        tail = s_arr[col[2] : col[1] + 1]
        neg = np.where(tail < 0)[0]
        if neg.size:
            newend = col[2] - 1 + int(neg[0])
        if (col[1] - newend) >= min_dur:
            seg = s_arr[newend + 1 : col[1] + 1]
            if seg.size and float(np.max(seg)) > 0:
                out_cols.append(np.array([newend + 1, col[1], newend + 1 + int(np.argmax(seg))]))
        col[1] = newend - 1
        newdeb = col[0] - 1
        head = s_arr[col[0] : col[2] + 1]
        neg2 = np.where(head < 0)[0]
        if neg2.size:
            newdeb = col[0] - 1 + int(neg2[-1])
        if (newdeb - col[0]) >= min_dur:
            seg2 = s_arr[col[0] : newdeb]
            if seg2.size and float(np.max(seg2)) > 0:
                out_cols.append(np.array([col[0], newdeb - 1, col[0] + int(np.argmax(seg2))]))
        col[0] = newdeb
        out_cols.append(col)
        i += 1
    if not out_cols:
        return np.zeros((3, 0), dtype=np.int64)
    return np.stack(out_cols, axis=1)


def _validaccident(
    accidents: np.ndarray,
    varsig: np.ndarray,
    durationthreshold_sec: float,
    amplitudethreshold: float,
    *,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if accidents.size == 0:
        return np.zeros((2, 0), dtype=np.float64), np.zeros((2, 0), dtype=np.float64)
    acc = np.asarray(accidents, dtype=np.float64)
    if acc.shape[0] == 3:
        acc = acc[:2, :]
    keep = acc[1, :] - acc[0, :] >= float(durationthreshold_sec)
    acc = acc[:, keep]
    if acc.size == 0:
        return np.zeros((2, 0), dtype=np.float64), np.zeros((2, 0), dtype=np.float64)
    scale = fs_hz
    flags = []
    for i in range(acc.shape[1]):
        lo = int(acc[0, i] * scale)
        hi = int(acc[1, i] * scale)
        lo = max(0, min(lo, varsig.size - 1))
        hi = max(lo, min(hi, varsig.size - 1))
        flags.append(float(np.max(varsig[lo : hi + 1])) >= amplitudethreshold)
    flags = np.asarray(flags, dtype=bool)
    reject = acc[:, ~flags]
    acc = acc[:, flags]
    return acc, reject


def run_wmfb(fhr: np.ndarray, *, fs_hz: float = 4.0) -> WmfbResult:
    """Run WMFB on preprocessed FHR at ``fs_hz`` (reference: 4 Hz)."""
    fhri = np.asarray(fhr, dtype=np.float64).reshape(-1)
    if fhri.size < 8:
        return WmfbResult(
            baseline=fhri.copy(),
            accelerations_sec=np.zeros((2, 0)),
            decelerations_sec=np.zeros((2, 0)),
            fs_hz=fs_hz,
        )

    srate = fs_hz * 60.0  # FHRMA convention: samples per minute

    fhr1 = _butterfilt(fhri, srate, 0.0, 1.0, 1, True)
    fhr2 = _butterfilt(fhri, srate, 0.0, 2.0, 1, True)
    fhr4 = _butterfilt(fhri, srate, 0.0, 4.0, 1, True)
    fhr8 = _butterfilt(fhri, srate, 0.0, 8.0, 1, True)
    fhr16 = _butterfilt(fhri, srate, 0.0, 16.0, 1, True)

    fcut = [0.0, 1.0, 3.0, 7.0]
    fdat: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(3):
        band = _butterfilt(fhri, srate, fcut[j], fcut[j + 1], 1, True)
        deriv = np.concatenate([[0.0], (band[1:] - band[:-1]) * srate])
        fdat.append((band, deriv))

    t = np.column_stack(
        [
            np.abs(fdat[0][1]),
            _enveloppe(fdat[0][1], srate, 0.0, 2.0 * fcut[1]),
            _enveloppe(fdat[1][1], srate, 0.0, 2.0 * fcut[2]),
            _enveloppe(fdat[2][1], srate, 0.0, 2.0 * fcut[3]),
        ]
    )
    q = np.array([-2.4744, 0.0266, 0.0413, 0.0105, 0.0036], dtype=np.float64)
    logits = q[0] + t @ q[1:]
    p = 1.0 - np.exp(logits) / (1.0 + np.exp(logits))

    distancecoef = np.concatenate([np.linspace(0.0, 1.0, 200), [1.0], np.linspace(1.0, 0.0, 200)])

    bl1, mp1 = _medgliss(fhr2, distancecoef, p, 24, srate)
    p2 = p * np.exp(3.21 - 0.19 * np.abs(fhr1 - bl1)) / (1.0 + np.exp(3.21 - 0.19 * np.abs(fhr1 - bl1)))
    bl2, _ = _medgliss(fhr2, distancecoef**2, p2, 24, srate, bl1, mp1, 0.1)
    p3 = p * np.exp(2.5 - 0.19 * np.abs(fhr4 - bl2)) / (1.0 + np.exp(2.5 - 0.19 * np.abs(fhr4 - bl2)))
    bl3, _ = _medgliss(fhr4, distancecoef**4, p3, 24, srate, bl2, mp1, 0.1)
    p4 = p * np.exp(2.0 - 0.19 * np.abs(fhr8 - bl3)) / (1.0 + np.exp(2.0 - 0.19 * np.abs(fhr8 - bl3)))
    bl4, _ = _medgliss(fhr8, distancecoef**8, p4, 24, srate, bl3, mp1, 0.1)
    p5 = p * np.exp(1.5 - 0.19 * np.abs(fhr16 - bl4)) / (1.0 + np.exp(1.5 - 0.19 * np.abs(fhr16 - bl4)))
    bl5, _ = _medgliss(fhr16, distancecoef**16, p5, 24, srate, bl4, mp1, 0.1)
    p6 = p * np.exp(1.0 - 0.19 * np.abs(fhr16 - bl5)) / (1.0 + np.exp(1.0 - 0.19 * np.abs(fhr16 - bl5)))
    baseline, _ = _medgliss(fhr16, distancecoef**16, p6, 24, srate, bl5, mp1, 0.1)

    min_dur = int(round(15.0 * fs_hz))
    acc = _accidentcandidat(fhr1, baseline, 5.0)
    acc = _adjustduration(acc, fhri - baseline, min_dur=min_dur)
    acc_sec = acc[:2, :] / fs_hz if acc.size else np.zeros((2, 0))
    acc_sec, _ = _validaccident(acc_sec, fhri - baseline, 15.0, 15.0, fs_hz=fs_hz)

    dec = _accidentcandidat(baseline, fhr1, 5.0)
    dec = _adjustduration(dec, baseline - fhri, min_dur=min_dur)
    dec_sec = dec[:2, :] / fs_hz if dec.size else np.zeros((2, 0))
    dec_sec, _ = _validaccident(dec_sec, baseline - fhri, 15.0, 15.0, fs_hz=fs_hz)

    return WmfbResult(
        baseline=baseline.astype(np.float64),
        accelerations_sec=acc_sec.astype(np.float64),
        decelerations_sec=dec_sec.astype(np.float64),
        fs_hz=fs_hz,
    )


