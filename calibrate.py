# observer toolkit incl calibration flag

WINDOWLEN = 201
FS = 9e6

import numpy as np
from scipy.optimize import least_squares, curve_fit


def C2RE(x):
#Separate the real (even indices) from imaginary (odd indices) parts of a complex array in a real array
    result = np.empty(2*np.array(x).shape[0], dtype=float)

    result[0::2] = np.real(x)
    result[1::2] = np.imag(x)
    return result


def RE2C(x):
    #Merge the real (even indices) and imaginary (odd indices) parts of a real array in a complex array'''

    x = np.array(x)
    return x[0::2] + 1.0j * x[1::2]

def calibrate_energy(hbw, probe_cmplx, vforw_cmplx, vrefl_cmplx, probe_sq_deriv,probe_cmplx_decay=None, vforw_cmplx_decay=None, vrefl_cmplx_decay=None):
    max_probe_recip = 1.0/np.max(np.abs(probe_cmplx))
    probe_cmplx_conj = np.conjugate(probe_cmplx)

    # Eq 12 (=18, 19)
    C = probe_sq_deriv/(2*hbw)
    D = C + np.abs(probe_cmplx)**2

    # check for optional decay data
    if (probe_cmplx_decay is None or vforw_cmplx_decay is None or vrefl_cmplx_decay is None):
        probe_cmplx_decay = np.zeros(0)
        vforw_cmplx_decay = np.zeros(0)
        vrefl_cmplx_decay = np.zeros(0)

    # Optimization routine. least squares tries to minimize ||fun(abcd)||
    def fun(abcd):
        abcd = RE2C(abcd)

        # apply current calibration
        vforw_calib = abcd[0] * vforw_cmplx + abcd[1] * vrefl_cmplx
        vrefl_calib = abcd[2] * vforw_cmplx + abcd[3] * vrefl_cmplx
        vforw_calib_decay = abcd[0] * vforw_cmplx_decay + abcd[1] * vrefl_cmplx_decay

        # Error of Eq. 15 = Eq. 6: prb = fwd + ref
        eprobe = vforw_calib + vrefl_calib - probe_cmplx

        # Error of Eq. 17 = Eq. 12
        eD = (2.0 * np.real(probe_cmplx_conj * vforw_calib) - D) * max_probe_recip

        # Error of Eq. 16 = Eq. 13
        eC = (np.abs(vforw_calib)**2 - np.abs(vrefl_calib)**2 - C) * max_probe_recip

        # Error of Eq. 8
        evforw_calib_decay = vforw_calib_decay

        return C2RE(np.concatenate([eprobe, eD, eC, evforw_calib_decay]))

    # The initial guess for the least squares algorithm is (a=1, b=0, c=0, d=1)
    return RE2C(least_squares(fun, C2RE([1.0, 0.0, 0.0, 1.0]), method="lm").x)

def omega12(prb_decay, timev=None, dt=None):
    if timev is None:
        assert dt is not None
        timev = [i*dt for i in range(len(prb_decay))]
    elif dt is None:
        assert len(timev) == len(prb_decay)
        timev = np.array(timev)-timev[0]
        # avoid numerical issues during scaling
    if prb_decay[0] == 0 or prb_decay[0] == np.nan:
        raise Exception
    prb_decay = prb_decay/prb_decay[0]

    def func(x, w):
        return np.exp(-w*x)
    popt, pcov = curve_fit(func, timev, prb_decay, bounds=(1,10000))
    return popt[0]

def AP2IQ(AP):
    """
    Convert Amplitude/Phase data to IQ

    Parameters
    ----------
    AP : np.array
        array of Amplitude and Phase data, either (2,n) or (n,2).

    Returns
    -------
    IQ : np.array
        array of I and Q data, (n,2).

    """
    if not AP.shape[1] == 2:
        AP=np.transpose(AP)
    IQ=np.zeros(AP.shape)
    IQ[:,0] = AP[:,0]*np.cos(AP[:,1]/180*np.pi)
    IQ[:,1] = AP[:,0]*np.sin(AP[:,1]/180*np.pi)
    return IQ

def IQ2c(IQ):
    return np.array([iq[0]+1.0j*iq[1] for iq in IQ])

def AP2c(AP):
    return IQ2c(AP2IQ(AP))


def c2IQ(c):
    IQ = np.zeros((len(c),2))
    IQ[:,0] = np.real(c)
    IQ[:,1] = np.imag(c)
    return IQ

def c2AP(c, deg=False):
    AP = np.zeros((len(c),2))
    AP[:,0] = np.abs(c)
    AP[:,1] = np.angle(c, deg)
    return AP

def savgol_derivative(tracestack, windowlen=WINDOWLEN, polyorder= 3, dt = 1/FS):
    from scipy.signal import savgol_filter
    dstackdt = np.zeros(tracestack.shape)
    for i, trace in enumerate(tracestack):
        trace2dt = savgol_filter(trace**2, window_length=windowlen, polyorder = polyorder, deriv=1, delta=dt)
        dstackdt [i,:] = trace2dt
    return dstackdt

def findtimeindices(forwardA, delaylim = 0.2, filllim = 0.9, flatlim = 0.1):
    maxA = np.max(forwardA)
    dellim = delaylim*maxA
    fillim = filllim *maxA
    flalim = flatlim*maxA
    delay = -1
    filling = -1
    flattop = -1
    for i, val in enumerate(forwardA):
        if val > dellim and delay < 0:
            delay = i
        elif val > fillim:
            filling = i
        elif val > flalim:
            flattop = i
    return delay, filling, flattop

def cal_c(fwd, ref, prb, windowlen=WINDOWLEN, decimation=1, fs = FS, beam=False):
    dprb2 = savgol_derivative(np.abs(prb), windowlen, dt = 1/fs)

    i_fill, i_flattop, i_decay = findtimeindices(np.abs(fwd[0]))
    is_filling = list(range(i_fill+windowlen, i_flattop-windowlen))
    is_flattop = list(range(i_flattop+windowlen, i_decay-windowlen))
    is_decay = list(range(i_decay+windowlen, len(fwd[0])-windowlen))
    if beam:
        is_flattop = []
    print(i_fill, i_flattop, i_decay)

    hbw_nominal = [omega12(np.abs(p), dt = 1/fs) for p in prb[:,is_decay]]
    hbw = np.mean(hbw_nominal)

    fwd_cal = fwd[:,is_filling+is_flattop+is_decay].flatten()
    ref_cal = ref[:,is_filling+is_flattop+is_decay].flatten()
    prb_cal = prb[:,is_filling+is_flattop+is_decay].flatten()
    dprb2_cal = dprb2[:,is_filling+is_flattop+is_decay].flatten()
    fwd_cal_decay = fwd[:,is_decay].flatten()
    ref_cal_decay = ref[:,is_decay].flatten()
    prb_cal_decay = prb[:,is_decay].flatten()

    S_abcd = calibrate_energy(hbw,
                                  prb_cal[::decimation],
                                  fwd_cal[::decimation],
                                  ref_cal[::decimation],
                                  dprb2_cal[::decimation],
                                  prb_cal_decay[::decimation],
                                  fwd_cal_decay[::decimation],
                                  ref_cal_decay[::decimation]
                              )

    return np.array([S_abcd[0:2], S_abcd[2:4]]), hbw/2/np.pi

def calibrate(S, fwd, ref):
    return np.dot(S,np.vstack((fwd, ref)))
