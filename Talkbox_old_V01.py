#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All the functions of the Scikits.Talkbox 0.2.5 toolbox, as found in:
	http://pydoc.net/Python/scikits.talkbox/0.2.5/
	
Guilherme Boaviagem - 10/12/2015
"""

import numpy as np
import scipy as sp
import scipy.signal as sig
from scipy.io import loadmat
from scipy.signal import lfilter, hamming
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.fftpack.realtransforms import dct
import warnings

# ====================================================================
# mel.py
# ====================================================================

def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log(f/700 +1)

def mel2hz(m):
    """Convert an array of frequency in Hz into mel."""
    return (np.exp(m / 1127.01048) - 1) * 700


# ====================================================================
# segmentaxis.py
# ====================================================================

"""sgementaxis code.

This code has been implemented by Anne Archibald, and has been discussed on the
ML."""

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap < 0 or length <= 0:
        raise ValueError, "overlap must be nonnegative and length must "\
                          "be positive"

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise ValueError, \
              "Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'"
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


# ====================================================================
# mfcc.py
# ====================================================================

__all__ = ['mfcc','preemp','trfbank']

def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs
    
def preemp(input, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, input)

def mfcc(input, nwin=256, nfft=512, fs=16000, nceps=13):
    """Compute Mel Frequency Cepstral Coefficients.

    Parameters
    ----------
    input: ndarray
        input from which the coefficients are computed

    Returns
    -------
    ceps: ndarray
        Mel-cepstrum coefficients
    mspec: ndarray
        Log-spectrum in the mel-domain.

    Notes
    -----
    MFCC are computed as follows:
        * Pre-processing in time-domain (pre-emphasizing)
        * Compute the spectrum amplitude by windowing with a Hamming window
        * Filter the signal in the spectral domain with a triangular
        filter-bank, whose filters are approximatively linearly spaced on the
        mel scale, and have equal bandwith in the mel scale
        * Compute the DCT of the log-spectrum

    References
    ----------
    .. [1] S.B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", IEEE Trans. Acoustics. Speech, Signal Proc.
           ASSP-28 (4): 357-366, August 1980."""

    # Number of overlapping samples in each frame
    t_overlap = 10*10**(-3) # Time in seconds of overlapping between frames
    over = int(t_overlap*fs)
# 	over = nwin - 160

    # Pre-emphasis factor (to take into account the -6dB/octave rolloff of the
    # radiation at the lips level)
    prefac = 0.97

    #lowfreq = 400 / 3.
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
    nfil = nlinfil + nlogfil

    w = hamming(nwin, sym=0)
    
    [fbank, freqs] = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil) # "fbank" is a nfil-by-nfft Numpy 2D array.
    
    '''
    # Visualizando o banco de filtros:
    plt.figure()
    nfiltros,lenfiltros = fbank.shape
    for i in range(nfiltros):
    	plt.plot(range(lenfiltros),fbank[i,:])
    plt.axis([0, lenfiltros, 0, np.max(fbank)])
    plt.show()
    '''
     
    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(input, prefac)
    framed = segment_axis(extract, nwin, over) * w

    # Compute the spectrum magnitude
    spec = np.abs(fft(framed, nfft, axis=-1))
    # Filter the spectrum through the triangle filterbank
    mspec = np.log10(np.dot(spec, fbank.T))
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]
    
    return ceps, mspec, spec

if __name__ == '__main__':
    extract = loadmat('extract.mat')['extract']
    ceps = mfcc(extract)
    

# ====================================================================
# correlations.py
# ====================================================================

__all__ = ['nextpow2', 'acorr']

def nextpow2(n):
    """Return the next power of 2 such as 2^p >= n.

    Notes
    -----

    Infinite and nan are left untouched, negative values are not allowed."""
    if np.any(n < 0):
        raise ValueError("n should be > 0")

    if np.isscalar(n):
        f, p = np.frexp(n)
        if f == 0.5:
            return p-1
        elif np.isfinite(f):
            return p
        else:
            return f
    else:
        f, p = np.frexp(n)
        res = f
        bet = np.isfinite(f)
        exa = (f == 0.5)
        res[bet] = p[bet]
        res[exa] = p[exa] - 1
        return res

def _acorr_last_axis(x, nfft, maxlag, onesided=False, scale='none'):
    a = np.real(ifft(np.abs(fft(x, n=nfft) ** 2)))
    if onesided:
        b = a[..., :maxlag]
    else:
        b = np.concatenate([a[..., nfft-maxlag+1:nfft],
                            a[..., :maxlag]], axis=-1)
    #print b, a[..., 0][..., np.newaxis], b / a[..., 0][..., np.newaxis]
    if scale == 'coeff':
        return b / a[..., 0][..., np.newaxis]
    else:
        return b

def acorr(x, axis=-1, onesided=False, scale='none'):
    """Compute autocorrelation of x along given axis.

    Parameters
    ----------
        x : array-like
            signal to correlate.
        axis : int
            axis along which autocorrelation is computed.
        onesided: bool, optional
            if True, only returns the right side of the autocorrelation.
        scale: {'none', 'coeff'}
            scaling mode. If 'coeff', the correlation is normalized such as the
            0-lag is equal to 1.

    Notes
    -----
        Use fft for computation: is more efficient than direct computation for
        relatively large n.
    """
    if not np.isrealobj(x):
        raise ValueError("Complex input not supported yet")
    if not scale in ['none', 'coeff']:
        raise ValueError("scale mode %s not understood" % scale)

    maxlag = x.shape[axis]
    nfft = 2 ** nextpow2(2 * maxlag - 1)

    if axis != -1:
        x = np.swapaxes(x, -1, axis)
    a = _acorr_last_axis(x, nfft, maxlag, onesided, scale)
    if axis != -1:
        a = np.swapaxes(a, -1, axis)
    return a
    
# ====================================================================
# ffilter.py
# ====================================================================

def slfilter(b, a, x):
    """Filter a set of frames and filter coefficients. More precisely, given
    rank 2 arrays for coefficients and input, this computes:

    for i in range(x.shape[0]):
        y[i] = lfilter(b[i], a[i], x[i])

    This is mostly useful for processing on a set of windows with variable
    filters, e.g. to compute LPC residual from a signal chopped into a set of
    windows.

    Parameters
    ----------
        b: array
            recursive coefficients
        a: array
            non-recursive coefficients
        x: array
            signal to filter

    Note
    ----

    This is a specialized function, and does not handle initial conditions,
    rank > 2 nor  arbitrary axis handling."""

    if not x.ndim == 2:
        raise ValueError("Only input of rank 2 support")

    if not b.ndim == 2:
        raise ValueError("Only b of rank 2 support")

    if not a.ndim == 2:
        raise ValueError("Only a of rank 2 support")

    nfr = a.shape[0]
    if not nfr == b.shape[0]:
        raise ValueError("Number of filters should be the same")

    if not nfr == x.shape[0]:
        raise ValueError, \
              "Number of filters and number of frames should be the same"

    y = np.empty((x.shape[0], x.shape[1]), x.dtype)

    for i in range(nfr):
        y[i] = lfilter(b[i], a[i], x[i])

    return y
    
# ====================================================================
# preprocessing.py
# ====================================================================

__all__ = ["demean"]

def demean(x, axis=-1):
    return x - np.mean(x,axis)
    

# ====================================================================
# py_lpc.py
# ====================================================================

def lpc_ref(signal, order):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)

    Notes
    ----
    This is just for reference, as it is using the direct inversion of the
    toeplitz matrix, which is really slow"""
    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype = signal.dtype)

def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in xrange(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k
    
# ====================================================================
# levinson_lpc.py
# Obs.: "levinson" nao foi copiado aqui.
# ====================================================================    
    
__all__ = ['lpc']

def acorr_lpc(x, axis=-1):
    """Compute autocorrelation of x along the given axis.

    This compute the biased autocorrelation estimator (divided by the size of
    input signal)

    Notes
    -----
        The reason why we do not use acorr directly is for speed issue."""
    if not np.isrealobj(x):
        raise ValueError("Complex input not supported yet")

    maxlag = x.shape[axis]
    nfft = 2 ** nextpow2(2 * maxlag - 1)

    if axis != -1:
        x = np.swapaxes(x, -1, axis)
    a = _acorr_last_axis(x, nfft, maxlag)
    if axis != -1:
        a = np.swapaxes(a, -1, axis)
    return a

def lpc(signal, order, axis=-1):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)

    Returns
    -------
    a : array-like
        the solution of the inversion.
    e : array-like
        the prediction error.
    k : array-like
        reflection coefficients.

    Notes
    -----
    This uses Levinson-Durbin recursion for the autocorrelation matrix
    inversion, and fft for the autocorrelation computation.

    For small order, particularly if order << signal size, direct computation
    of the autocorrelation is faster: use levinson and correlate in this case."""
    n = signal.shape[axis]
    if order > n:
        raise ValueError("Input signal must have length >= order")

    r = acorr_lpc(signal, axis)
    # Obs.: onde há "levinson_1d" abaixo, era "levinson"!
    return levinson_1d(r, order)

def _acorr_last_axis(x, nfft, maxlag):
    a = np.real(ifft(np.abs(fft(x, n=nfft) ** 2)))
    return a[..., :maxlag+1] / x.shape[-1]

# ====================================================================
# peak_picking.py
# ====================================================================

def find_peaks(x, neighbours):
    peaks = []
    nx = x.size

    assert 2 * neighbours + 1 <= nx

    if nx == 1:
        peaks.append(0)
        return peaks
    elif nx == 2:
        if x[0] > x[1]:
            peaks.append(0)
        else:
            peaks.append(1)
            return peaks

    # Handle points which have less than neighs samples on their left
    for i in range(neighbours):
        cur = x[i]
        m = x[i+1]
        # look at the left of the current position
        for j in range(i):
            if m < x[j]:
                m = x[j]
        # look at the right of the current position
        for j in range(i+1, i+neighbours):
            if m < x[j]:
                m = x[j]

        if cur > m:
            peaks.append(i)
            #assert(pkcnt <= (nx / neighbours + 1))

    # Handle points which have at least neighs samples on both their left
    # and right
    for i in range(neighbours, nx - neighbours):
        cur = x[i]
        m = x[i+1];
        # look at the left
        for j in range(i - neighbours, i):
            if m < x[j]:
                m = x[j]
        # look at the right
        for j in range(i+1, i+neighbours):
            if m < x[j]:
                m = x[j]

        if cur > m:
            peaks.append(i)
            #assert(pkcnt <= (nx / neighbours + 1))

    # Handle points which have less than neighs samples on their right
    for i in range(nx - neighbours, nx):
        cur = x[i]
        m = x[i-1]
        # look at the left
        for j in range(i - neighbours, i):
            if m < x[j]:
                m = x[j]

        # look at the right
        for j in range(i+1, nx):
            if m < x[j]:
                m = x[j]

        if cur > m:
            peaks.append(i)
            #assert(pkcnt <= (nx / neighbours + 1))

    return peaks

# ====================================================================
# SPECTRAL ANALYSIS
# basic.py
# ====================================================================

def periodogram(x, nfft=None, fs=1):
    """Compute the periodogram of the given signal, with the given fft size.

    Parameters
    ----------
    x : array-like
        input signal
    nfft : int
        size of the fft to compute the periodogram. If None (default), the
        length of the signal is used. if nfft > n, the signal is 0 padded.
    fs : float
        Sampling rate. By default, is 1 (normalized frequency. e.g. 0.5 is the
        Nyquist limit).

    Returns
    -------
    pxx : array-like
        The psd estimate.
    fgrid : array-like
        Frequency grid over which the periodogram was estimated.

    Examples
    --------
    Generate a signal with two sinusoids, and compute its periodogram:

    >>> fs = 1000
    >>> x = np.sin(2 * np.pi  * 0.1 * fs * np.linspace(0, 0.5, 0.5*fs))
    >>> x += np.sin(2 * np.pi  * 0.2 * fs * np.linspace(0, 0.5, 0.5*fs))
    >>> px, fx = periodogram(x, 512, fs)

    Notes
    -----
    Only real signals supported for now.

    Returns the one-sided version of the periodogram.

    Discrepency with matlab: matlab compute the psd in unit of power / radian /
    sample, and we compute the psd in unit of power / sample: to get the same
    result as matlab, just multiply the result from talkbox by 2pi"""
    # TODO: this is basic to the point of being useless:
    #   - support Daniel smoothing
    #   - support windowing
    #   - trend/mean handling
    #   - one-sided vs two-sided
    #   - plot
    #   - support complex input
    x = np.atleast_1d(x)
    n = x.size

    if x.ndim > 1:
        raise ValueError("Only rank 1 input supported for now.")
    if not np.isrealobj(x):
        raise ValueError("Only real input supported for now.")
    if not nfft:
        nfft = n
    if nfft < n:
        raise ValueError("nfft < signal size not supported yet")

    pxx = np.abs(fft(x, nfft)) ** 2
    if nfft % 2 == 0:
        pn = nfft / 2 + 1
    else:
        pn = (nfft + 1 )/ 2

    fgrid = np.linspace(0, fs * 0.5, pn)
    return pxx[:pn] / (n * fs), fgrid

def arspec(x, order, nfft=None, fs=1):
    """Compute the spectral density using an AR model.

    An AR model of the signal is estimated through the Yule-Walker equations;
    the estimated AR coefficient are then used to compute the spectrum, which
    can be computed explicitely for AR models.

    Parameters
    ----------
    x : array-like
        input signal
    order : int
        Order of the LPC computation.
    nfft : int
        size of the fft to compute the periodogram. If None (default), the
        length of the signal is used. if nfft > n, the signal is 0 padded.
    fs : float
        Sampling rate. By default, is 1 (normalized frequency. e.g. 0.5 is the
        Nyquist limit).

    Returns
    -------
    pxx : array-like
        The psd estimate.
    fgrid : array-like
        Frequency grid over which the periodogram was estimated.
    """

    x = np.atleast_1d(x)
    n = x.size

    if x.ndim > 1:
        raise ValueError("Only rank 1 input supported for now.")
    if not np.isrealobj(x):
        raise ValueError("Only real input supported for now.")
    if not nfft:
        nfft = n
    if nfft < n:
        raise ValueError("nfft < signal size not supported yet")

    a, e, k = lpc(x, order)

    # This is not enough to deal correctly with even/odd size
    if nfft % 2 == 0:
        pn = nfft / 2 + 1
    else:
        pn = (nfft + 1 )/ 2

    px = 1 / np.fft.fft(a, nfft)[:pn]
    pxx = np.real(np.conj(px) * px)
    pxx /= fs / e
    fx = np.linspace(0, fs * 0.5, pxx.size)
    return pxx, fx

def taper(n, p=0.1):
    """Return a split cosine bell taper (or window)

    Parameters
    ----------
        n: int
            number of samples of the taper
        p: float
            proportion of taper (0 <= p <= 1.)

    Note
    ----
    p represents the proportion of tapered (or "smoothed") data compared to a
    boxcar.
    """
    if p > 1. or p < 0:
        raise ValueError("taper proportion should be betwen 0 and 1 (was %f)"
                         % p)
    w = np.ones(n)
    ntp = np.floor(0.5 * n * p)
    w[:ntp] = 0.5 * (1 - np.cos(np.pi * 2 * np.linspace(0, 0.5, ntp)))
    w[-ntp:] = 0.5 * (1 - np.cos(np.pi * 2 * np.linspace(0.5, 0, ntp)))

    return w


# ====================================================================
# dct.py
# ====================================================================

def dctii(x):
    """Compute a Discrete Cosine Transform, type II.

    The DCT type II is defined as:

        \forall u \in 0...N-1, 
        dct(u) = a(u) sum_{i=0}^{N-1}{f(i)cos((i + 0.5)\pi u}

    Where a(0) = sqrt(1/(4N)), a(u) = sqrt(1/(2N)) for u > 0

    Parameters
    ==========
    x : array-like
        input signal

    Returns
    =======
    y : array-like
        DCT-II

    Note
    ====
    Use fft.
    """
    if not np.isrealobj(x):
        raise ValueError("Complex input not supported")
    n = x.size
    y = np.zeros(n * 4, x.dtype)
    y[1:2*n:2] = x
    y[2*n+1::2] = x[-1::-1]
    y = np.real(fft(y))[:n]
    y[0] *= np.sqrt(.25 / n)
    y[1:] *= np.sqrt(.5 / n)
    return y

# ====================================================================
# dct_ref.py
# ====================================================================

def direct_dctii(x):
    """Direct implementation (O(n^2)) of dct II.

    Notes
    -----

    Use it as a reference only, it is not suitable for any real computation."""
    n = x.size
    a = np.empty((n, n), dtype = x.dtype)
    for i in xrange(n):
        for j in xrange(n):
            a[i, j] = x[j] * np.cos(np.pi * (0.5 + j) * i / n)

    a[0] *= np.sqrt(1. / n)
    a[1:] *= np.sqrt(2. / n)

    return a.sum(axis = 1)

def direct_dctii_2(x):
    """Direct implementation (O(n^2)) of dct."""
    # We are a bit smarter here by computing the coefficient matrix directly,
    # but still O(N^2)
    n = x.size

    a = np.cos(np.pi / n * np.linspace(0, n - 1, n)[:, None]
                         * np.linspace(0.5, 0.5 + n - 1, n)[None, :])
    a *= x
    a[0] *= np.sqrt(1. / n)
    a[1:] *= np.sqrt(2. / n)

    return a.sum(axis = 1)

if __name__ == "__main__":
    a = np.linspace(0, 10, 11)
    print direct_dctii_2(a)
    a = np.linspace(0, 10, 11)
    print direct_dctii_2(a)