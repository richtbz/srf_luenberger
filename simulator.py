import numpy as np

def simcav(tvec,
           f12_op, df_0,
           tau_lfd, klfd, init_lfd, mechModes, initMechM,
           ffRf, ffPiezo):
    """
    Simulate a cavity at using a fixed time step Runge-Kutta4 discretization.

    Parameters
    ----------
    tvec : 1D array of float
        sampling timestamps (not necessarily equidistant).

    f12_op : float
        operational half bandwidth [Hz].
    df_0 : float
        initial detuning.
    tau_lfd : float
        first order LFD time constant.
    klfd : float
        first order LFD coefficient.
    init_lfd : float
        first order LFD initial state.
    mechModes : list of lists of float
        mechanical modes eigenfrequencies w, quality factors q and LFD coeffs k.
        must be populated, at least one mechanical mode! (none: set k to zero)
        e.g. [[w, q, k]]
    initMechM : 1D array of float
        intital states of mechanical modes.

    Returns
    -------
    tuple of:
    x: 2D array of float
        array of simulated states.
    cl: 2D array of float
        array of applied feedback (IQ).
    """
    tvec = np.append(tvec, [2*tvec[-1]-tvec[-2]])

    nmechM = len(mechModes)
    nsteps = len(tvec)

    # cavity and drive parameters
    w12_op = 2*np.pi*f12_op                 # operational bandwidth
    dw_0 = 2*np.pi*df_0                     # predetuning

    # reserve mem and initialize state vector trajectory
    #  Icav
    #  Qcav
    #  dbw/w12_0            # bandwidth deviation relative to operational bw
    #  dwlfd/w12_0          # detuning relative to operational bw
    x = np.empty((nsteps, 4+2*nmechM), dtype=np.float64)
    x[0, 0] = .0
    x[0, 1] = .0
    x[0, 2] = .0
    x[0, 3] = init_lfd      # absolute direct LFD
    x[0, 4:] = initMechM

    # differential equations
    def cav_dyn(x, If, Qf, Up):
        dx = np.empty((4+2*nmechM), dtype=np.float64)

        # prepare frequently used values
        w12 = w12_op * (1+x[2])
        IQabs2 = x[0]**2 + x[1]**2
        dw = dw_0 + x[3] + x[4::2].sum()  # + dw_ext ??

        # RF dynamics
        dx[0] = - w12 * x[0] - dw * x[1] + w12_op * 2 * If
        dx[1] = - w12 * x[1] + dw * x[0] + w12_op * 2 * Qf

        # no known dynamics in bandwidth
        dx[2] = 0

        # direct LFD dynamics (first order)
        dx[3] = -1/tau_lfd*(x[3] - IQabs2*2*np.pi*klfd)

        # mechanical dynamics
        for i in range(nmechM):
            w, q, k = mechModes[i]
            x_mech = x[4+2*i]
            dx_mech = x[4+2*i+1]
            dx[4+2*i] = dx_mech
            dx[4+2*i+1] = - w / q * dx_mech - (w ** 2) * \
                (x_mech - 2*np.pi * (Up + IQabs2 * k))

        return dx

    def checkImmediateLFD(x, ts):
        if tau_lfd < 2*ts:
            x[3] = (x[0]**2 + x[1]**2)*klfd*2*np.pi
        return x

    # discrete time approximation of differential equation (RungeKutta4)
    def RK4(x, ts, If, Qf, Up):
        k1 = cav_dyn(checkImmediateLFD(x, ts), If, Qf, Up)
        k2 = cav_dyn(checkImmediateLFD(x + k1*ts/2, ts), If, Qf, Up)
        k3 = cav_dyn(checkImmediateLFD(x + k2*ts/2, ts), If, Qf, Up)
        k4 = cav_dyn(checkImmediateLFD(x + k1*ts, ts), If, Qf, Up)
        return checkImmediateLFD(x + ts * (1/6) * (k1 + 2*k2 + 2*k3 + k4), ts)

    # ODE integration loop
    for i in range(nsteps-1):
        x[i+1] = RK4(
            x[i],
            tvec[i+1]-tvec[i],
            ffRf[i].real, ffRf[i].imag,
            ffPiezo[i])

    return x[:-1, :]


def ffPiezo(tvec, trigger, AC, DC, w, numsines=1):
    """
    Generate piezo drive signal (DC + integer multiple sinusoidals).

    Parameters
    ----------
    tvec : 1D array of float
        vector of timestamps.
    trigger: float
        timestamp in seconds when to start sinusoid.
    AC : TYPE
        sinusoidal piezo amplitude.
    DC : TYPE
        constant piezo amplitude.
    w : float
        frequency of sine in radians/s.
    numsines : int, optional
        number of applied sine periods. The default is 1.

    Returns
    -------
    Up : 1D array of floats
        piezo forward drive trace.

    """
    Up = np.zeros(tvec.shape)
    tmax = trigger + numsines*2*np.pi/w
    for i, t in enumerate(tvec):
        if trigger < t < tmax:
            Up[i] = AC * np.sin(w*(t-trigger))
        Up[i] += DC
    return Up


def ffRF(tvec, tdelay, tfill, tflat, ufill, ratio,
         phaseoffset=0., maxphase=0., timeconstant=200., SP_ampl=1.):
    """
    Generate feed forward signal for pulsed operation.

    Optional: Setpoint generation (requires w12).

    Parameters
    ----------
    tvec : 1D array of float
        vector of timestamps.
    tdelay : float
        timestamp where filling begins.
    tfill : float
        timestamp where filling ends (flattop follows).
    tflat : float
        timestamp where flattop ends (decay follows)..
    ufill : float
        forward drive during filling.
    ratio : float
        forward drive during flattop as fraction of drive during filling.
    phaseoffset : float, , optional
        phase offset for phase roll. The default is 0.
    maxphase : float, optional
        total phase for phase roll. The default is 0.
    timeconstant : float, optional
        exponential phase roll functions time constant in microseconds.
        The default is 200.
    w12 : float, optional
        Half bandwdith for Setpoint generation. The default is None
    SP_ampl: float, optional
        Maximum Setpoint amplitude. The default is 1.

    Returns
    -------
    Vf: 1D array of complex
        complex feed forward voltages
    """
    # https://gitlab.desy.de/msk-sw/low-level-radio-frequency/llrfctrl/llrfctrl-server/-/blob/master/src/TableGeneration.cc#L390
    fwd = np.zeros(tvec.shape, dtype=complex)
    phase = np.ones(tvec.shape)*phaseoffset

    # find indices of given time stamps
    i_start = np.searchsorted(tvec, tdelay, side='right') - 1
    i_fill = np.searchsorted(tvec, tfill, side='right') - 1
    i_flat = np.searchsorted(tvec, tflat, side='right') - 1

    uflat = ufill*ratio

    for i in range(i_start, i_fill):
        fwd[i] = ufill
    for i in range(i_fill, i_flat+1):
        fwd[i] = uflat

    # PhaseModulatingSubAlgo::modulatePhase(int32_t idx)
    # https://gitlab.desy.de/msk-sw/low-level-radio-frequency/llrfctrl/llrfctrl-server/-/blob/master/src/TableGeneration.cc#L76
    n = i_fill-i_start
    dt = np.concatenate((np.array([tvec[1]-tvec[0]]), tvec[1:] - tvec[:-1]))
    tau_n = timeconstant*1e-6/dt  # timeconstant in microseconds

    for i in range(n):
        fraction = (
            np.exp(-i/tau_n[i_start+i])-np.exp(-n/tau_n[i_start+i]))/(
            1.0 - np.exp(-n/tau_n[i_start+i]))
        if not 0 <= fraction <= 1:
            fraction = 0.0
        phase[i_start + i] += fraction * maxphase

    fwd *= np.exp(1j*phase*np.pi/180)

    return fwd
