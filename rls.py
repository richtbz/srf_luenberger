# Open loop recursive least squares based SRF cavity parameter identification
# Copyright (C) 2023 V. Ziemann
# Source: https://github.com/volkziem/SysidRFcavity
#
# The content of this file was originally part of the SysidRFcavity project
# and is redistributed here under the same license terms.
#
# Modifications:
# - 2025 - Bozo Richter - reduce to SRF cavity application
#                       - correct implementation of eq. 21
#                       - introduce minimum probe amplitude threshold
#                       - provide algorithm as a function
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def c2iq(c):
    return np.array([c.real, c.imag])


def rls(dt, Vf, Vcav, f12, ffilt, fdet0, amin, noise=None, pT0=1, init=False):
    """
    recursive least squares parameter identification, (c) V. Ziemann 2023

    Parameters
    ----------
    dt : TYPE
        sampling time step.
    Vf : TYPE
        complex list of calibrated forward voltages.
    Vcav : TYPE
        complex list of calibrated cavity voltages.
    f12 : TYPE
        cavity half bandwidth in Hertz.
    ffilt : TYPE
        corresponding filtering frequency. Sets forgetting factor.
    fdet0 : TYPE
        initial detuning estimate.
    amin : TYPE
        Minimum cavity voltage threshold to enable estimation.
    noise : TYPE, optional
        additional complex 2D noise sequence to add to Vf ([:, 0]) and
        Vcav ([:, 1]). The default is None.
    pT0 : TYPE, optional
        Initial estimate of pT (proportional to fit variance). The default is 1.
    init : TYPE, optional
        Whether to initialize the estimate with the given f12 and fdet0. The
        default is False.

    Returns
    -------
    hbw : TYPE
        estimated half bandwidth in Hertz.
    det : TYPE
        estimated detuning in Hertz.

    """

    # time horizon by desired filter cutoff frequency
    Nforget = 1/(2*np.pi*ffilt*dt)
    # forgetting factor
    alpha = 1-1/Nforget
    # initial value of pT
    pT = pT0

    # initial parameter estimate
    qT = np.zeros((2,))

    # input matrix, specifically for SRF cavities and input voltages
    B = 2 * f12*2*np.pi * dt * np.eye(2)

    hbw = np.zeros_like(Vf, dtype=float)
    det = np.zeros_like(Vf, dtype=float)
    pt = np.zeros_like(Vf, dtype=float)
    if noise is None:
        noise=np.zeros((Vf.shape[0], 2))

    if init:
        qT[0] = f12*2*np.pi*dt
        qT[1] = fdet0*2*np.pi*dt
        hbw[:2] += f12
        det[:2] += fdet0

    for T in range(Vf.size-2):
        if np.abs(Vcav[T] + noise[T, 1]) >= amin:
            # collect required measurement data
            uT1 = c2iq(Vf[T+1]   + noise[T+1, 0])
            VT =  c2iq(Vcav[T]   + noise[T,   1])
            VT1 = c2iq(Vcav[T+1] + noise[T+1, 1])
            VT2 = c2iq(Vcav[T+2] + noise[T+2, 1])

            # eq. 11
            yT2 = VT2 - VT1 - B@uT1
            # eq. 15
            V2T = VT@VT
            # eq. 20 (bracket)
            tmp = 1/(alpha+pT*V2T)
            # eq. 21
            qT1 = tmp*(alpha*qT + pT *
                        np.array([-VT1[0]*yT2[0] - VT1[1]*yT2[1],
                                  -VT1[1]*yT2[0] + VT1[0]*yT2[1]])
                        )
            # advancing variables for next iteration
            qT = qT1
            # eq. 20
            pT = tmp*pT

        # save estimates to memory
        # available at T+2 since using measurements from T+2
        hbw[T+2] = qT[0]/(2*np.pi*dt)
        det[T+2] = qT[1]/(2*np.pi*dt)
        pt[T+2] = pT

    return hbw, det, pt

def lpf(x, fc, dt):
    tau = 1.0 / (2*np.pi*fc)
    alpha = dt / (tau + dt)
    y = np.zeros_like(x)
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = (1 - alpha) * y[n-1] + alpha * x[n]
    return y
