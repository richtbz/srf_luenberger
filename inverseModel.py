from scipy.signal import butter, sosfilt, lfilter
import numpy as np

def inverseModel(fwd_c, prb_c, fs, f12, f0=1.3e9, samplesIgnored=0,
                 smoothingWindow=400, filtering=False, filtF=36_100,
                 minGradient=1, decimation=1):
    w12 = f12*2*np.pi

    pulseLength = prb_c.shape[0]

    if filtering:
        sos = butter(2, filtF, fs=fs, output='sos')
        prb = sosfilt(sos, prb_c)
        fwd = sosfilt(sos, fwd_c)
    else:
        prb = prb_c
        fwd = fwd_c

    # decimation
    prb = prb[::decimation]
    fwd = fwd[::decimation]

    pulseLengthDecimated = prb.shape[0]

    Pdot_IQ = np.zeros((pulseLengthDecimated, ), dtype=complex)
    Pdot_IQ[1:] = (prb[1:]-prb[:-1])*fs/decimation

    w12det = np.zeros((pulseLengthDecimated, 2))*np.nan

    Psq = np.abs(prb)**2
    F = (2*np.pi*f12*2*fwd - Pdot_IQ)
    w12det[:, 0] = (np.real(prb)*np.real(F) + np.imag(prb)*np.imag(F))/Psq
    w12det[:, 1] = (-np.imag(prb)*np.real(F) + np.real(prb)*np.imag(F))/Psq

    # undecimation
    if decimation != 1:
        temp = np.repeat(w12det[:-2], decimation, axis=0)
        w12det = np.vstack((np.zeros((decimation, 2)), temp, np.repeat(w12det[[-2], :], pulseLength-temp.shape[0], axis=0)))
        w12det = w12det[1:-decimation+1]
    w12det[:samplesIgnored, :] = np.nan
    w12det[np.abs(prb_c)**2 < minGradient, 0] = w12
    w12det[np.abs(prb_c)**2 < minGradient, 1] = 0

    if filtering:
        return w12det
    else:
        smoothingWindow = (smoothingWindow//2) * 2
        smoothed = np.zeros(w12det.shape)*np.nan
        for i in range(smoothingWindow//2, pulseLength-smoothingWindow//2+1):
            smoothed[i] = np.sum(w12det[i-smoothingWindow//2:i+smoothingWindow//2, :], axis=0)/smoothingWindow
        return smoothed
