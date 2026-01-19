import matplotlib as mpl
import colorsys
import numpy as np
from observer import luenberger_observer as observer
from scipy.signal import lfilter
import wget
from os.path import exists

import simulator
from rls import rls

plotRLS = False

plt = mpl.pyplot

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 11
mpl.rcParams['font.family'] = 'Latin Modern Roman'



# %% SIMULATION BASED observer results

mechModes = list()  # omega, Q, klfd
mechModes.append([250*2*np.pi, 10, -0.15])
mechModes.append([5000*2*np.pi, 100, -0.2])

fs = 9e6
time_trace3 = np.arange(-7e-3, 2e-3, 1/fs)

Up = simulator.ffPiezo(time_trace3, trigger=-5.628e-3, AC=-1, DC=0, w=mechModes[0][0])
tfill = 0
tflat = 750e-6
tdecy = 1500e-6
Urf = simulator.ffRF(time_trace3, tfill, tflat, tdecy,
           ufill=8.2, ratio=0.49,
           phaseoffset=0, maxphase=-10, timeconstant=300)


half_bandwidth = 141.
init_detuning = 62.0
result = simulator.simcav(time_trace3, f12_op=half_bandwidth, df_0=init_detuning,
                tau_lfd=2e-6, klfd=-0.65, init_lfd=0.,
                mechModes=mechModes, initMechM=np.zeros((2*len(mechModes))),
                ffRf=Urf, ffPiezo=Up)


prbC = result[:, 0] + 1j*result[:, 1]
fwdC = Urf
hbw = (result[:, 2] + 1) * half_bandwidth
det = init_detuning + \
    (result[:, 3]+np.sum(result[:, 4::2], axis=1))/(2*np.pi)

fig, axs = plt.subplots(nrows=3, sharex=True)
axs[0].plot(time_trace3, np.abs(prbC))
axs[0].set_ylabel("amplitude (MV)")
axs[1].plot(time_trace3, np.angle(prbC, deg=True))
axs[1].set_ylabel("phase (deg)")
axs[2].plot(time_trace3, det)
axs[2].set_ylabel("detuning (Hz)")
[ax.grid() for ax in axs]
axs[0].set_title("simulated cavity signals")
fig.align_ylabels(axs)
axs[-1].set_xlabel("time (s)")



# %% plot functions


def addmarker(axs, hsp, x, label, yoffs=0, bar=False, left=True, right=True):
    axl = axs[0].get_ylim()
    # vertical lines across all plots
    def vline(xval):
        bot = axl[0] - 2*(1+hsp) * (axl[1]-axl[0])
        top = axl[1] + 0.2 * (axl[1]-axl[0])
        axs[0].plot([xval, xval], [bot, top], 'k--', linewidth=0.5, clip_on=False)
    if left:
        vline(x[0])
    if right:
        vline(x[1])

    texty = axl[1] + (0.15-yoffs) * (axl[1]-axl[0])

    axs[0].text((x[0]+x[1])/2, texty, label, fontsize=10, ha="center", va="center", backgroundcolor="w",
                bbox=dict(facecolor='w', edgecolor='w', pad=1.5))
    # horizontal line besides text
    if bar:
        axs[0].plot(x, [texty, texty], 'k-', linewidth=1, clip_on=False)


def plotSim(axs, leglabel, prbC, fwdC, obsBw, Amin, hbw=half_bandwidth,
            linespec={'linestyle':'-'}, marker=None, markeroffset=(0.1, 0.05), ms=5,
            df0=0, fs=fs, fbw=1, fdet=1):
    x = observer(probe=np.array([np.real(prbC), np.imag(prbC)], dtype=np.float64).T,
                 forward=np.array([np.real(fwdC), np.imag(fwdC)], dtype=np.float64).T,
                 half_bandwidth_ext=hbw,
                 detuning_init=df0,
                 sample_rate=fs,
                 observer_bandwidth=obsBw,
                 amplitude_threshold=Amin,
                 bandwidth_gain_factor=fbw,
                 detuning_gain_factor=fdet
                 )

    axs[0].plot(time_trace3*1e6, (hbw+x[:, 2])*100/half_bandwidth, **linespec,
                marker=marker, markersize=ms, markevery=(markeroffset[0]/200, 0.13),
                label=leglabel)
    axs[1].plot(time_trace3*1e6, -x[:, 3], **linespec,
                marker=marker, markersize=ms, markevery=(markeroffset[1]/100, 0.13),
                label=leglabel)

    axs[2].plot(time_trace3*1e6, det+x[:, 3], **linespec,
                marker=marker, markersize=ms, markevery=(markeroffset[1]/100, 0.13))
    axs[3].plot(time_trace3*1e6, det+x[:, 3], **linespec,
                marker=marker, markersize=ms, markevery=(markeroffset[1]/50, 0.3))
    return x

def digitalFilterEquivalent(fc, fs, fdet=1):
    lam = np.exp(-2*np.pi*fc/fs)
    lam1 = lam + (1 - lam)*np.sqrt(1 - fdet + 0j)
    lam2 = lam - (1 - lam)*np.sqrt(1 - fdet + 0j)
    print(lam)
    print("\ndisc. time pole:")
    print(lam1)
    print("\ncont. time pole: [rad/s]")
    print(fs*np.log(lam1))
    print(f"\nnatural frequency: {np.abs(fs*np.log(lam1))/2/np.pi:.0f} Hz")
    print(f"damping: {np.cos(np.pi-np.angle(fs*np.log(lam1))):.2f}")
    print(f"Qfactor: {1/2/np.cos(np.pi-np.angle(fs*np.log(lam1))):.2f}")
    print(f"theta: {180-np.angle(fs*np.log(lam1), deg=True):.2f} deg")
    den = np.real([1., -lam1-lam2, lam1*lam2])
    num = np.real(1 - lam1 - lam2 + lam1*lam2)
    return num, den


# %% plotting and observer application


figsize = (4, 5.2) # in inches
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=figsize)
hsp = 0.15
xlim = (-70, 1750)
xlim3 = (1000, 1325)
ylim2 = (-4.5, 3.2)
yoffset3 = 0.02
axs = np.append(axs, axs[2].inset_axes([
    (xlim3[0]-xlim[0])/np.diff(xlim)[0],
    yoffset3,
    np.diff(xlim3)[0]/np.diff(xlim)[0],
    (-2 - ylim2[0])/np.diff(ylim2)[0]-yoffset3
    ]))
axs[3].tick_params(labelbottom=False)
plt.subplots_adjust(hspace=hsp)

markeroffsets = ((-8, 1.0),
                 (-25, 5.7),
                 (3, 8.0),
                 (10, 9.0))


# virtual beam compensation (additional fwd signal without probe alteration)

deltaBandwidth = 0.1*half_bandwidth
deltaDetuning = 20
beampos = (tflat + 250e-6, tflat + 500e-6)
beam = np.zeros(time_trace3.shape, dtype=complex)
beam[time_trace3 > beampos[0]] = (deltaBandwidth - 1j * deltaDetuning)/half_bandwidth*np.abs(prbC[np.isclose(time_trace3, (beampos[0]+beampos[1])/2)][0])
beam[time_trace3 > beampos[1]] = 0


# plot parameter groud truth

simlabel = "ground truth"
axs[1].plot(time_trace3*1e6, det, 'k--', label=simlabel, zorder=10)
axs[0].plot(time_trace3*1e6, hbw/half_bandwidth*100, 'k--', zorder=5, label=simlabel)
axs[2].set_xlabel("time ($\\mu s$)")
[ax.grid(zorder=-1) for ax in axs]
axs[1].set_ylabel("$\Delta\hat\omega$ (Hz)")
axs[0].set_ylabel("$\hat{\omega}_{1/2}/\omega_{1/2}$ (\%)")
axs[2].set_ylabel("$\Delta\omega - \Delta\hat\omega$ (Hz)")


# observer application

rfBandwidth = 10_000
lam0 = np.exp(-2*np.pi*rfBandwidth/fs)
print(f"0 < f < {2/(1-lam0):.2f}")
Amin = 1

# observer correct initial conditions, incl beam
xbeam = plotSim(axs, r"1. $\kappa = 1$", prbC, fwdC+beam/2, rfBandwidth, Amin,
                marker='o', markeroffset=markeroffsets[0],
                df0=0*det[np.argmin(np.abs(np.abs(prbC)-Amin))])

# observer wrong coupling/QL/QE
xdbw = plotSim(axs, r"2. $\kappa=1.1$",
               prbC, fwdC+beam/2, rfBandwidth, Amin, half_bandwidth*1.1, marker='^', markeroffset=markeroffsets[1], ms=6,
               df0=-0*det[np.argmin(np.abs(np.abs(prbC)-Amin))])

# observer correct initial conditions, calibration error (fwd rotation [->det] or scalling [->hbw] offset)
xrot = plotSim(axs, r"3. $\kappa = 1$, $\mathbf{u}$ rotated", prbC, (beam/2+fwdC)*np.exp(-10j/180*np.pi), rfBandwidth, Amin, marker="*", ms=7, markeroffset=markeroffsets[3])


# RLS comparison

if plotRLS:
# RLS ideal
    # hbwrls, detrls, _ = rls(1/fs, (fwdC+0*beam/2)*np.exp(-25j/180*np.pi), prbC,
    hbwrls, detrls, _ = rls(1/fs, fwdC+beam/2, prbC,
                         half_bandwidth*1, rfBandwidth, 0,
                         amin=Amin, noise=np.zeros((time_trace3.size, 2)),
                         pT0=1e-6, init=True)
    axs[0].plot(time_trace3*1e6, hbwrls*100/half_bandwidth, label=r"RLS, $\kappa = $1")
    axs[1].plot(time_trace3*1e6, detrls)
    axs[2].plot(time_trace3*1e6, det-detrls)


# detuning estimation error: idealized filter Gedx

axs[2].plot(time_trace3*1e6, det-lfilter(
    *digitalFilterEquivalent(rfBandwidth, fs), det
    ), 'k', label=r"$\Delta\omega$ filtered by $G_{edx}$")
leg = axs[2].legend(framealpha=1)
leg.remove()
axs[0].add_artist(leg)



# axes configuration

axs[0].set_xlim(xlim)
axs[0].set_yticks(np.arange(80, 140, 10))
axs[0].set_ylim(85, 130)
axs[0].set_zorder(2)

axs[1].set_ylim([-5, 80])
axs[1].set_yticks(np.arange(0, 100, 25))

axs[2].set_ylim(ylim2)
axs[2].set_xticks([0, 750, 1500])

axs[3].set_xlim(xlim3)
axs[3].set_ylim([-26, -2])
axs[3].set_xticks([])
axs[3].grid(False)

fig.align_ylabels(axs)
legloc = (0.45, 1.02)
fig.legend([_l.get_label() for _l in axs[0].lines if not _l.get_label().startswith('_')],
           ncols=2, bbox_to_anchor=legloc, loc="center")

addmarker(axs, hsp, [0, tflat*1e6], "RF filling", bar=True)
addmarker(axs, hsp, [beampos[0]*1e6, beampos[1]*1e6], "Beam", yoffs=0.065, bar=True)
addmarker(axs, hsp, [tflat*1e6, tdecy*1e6], "RF flattop", yoffs=-0.075, bar=True, left=False, right=False)
addmarker(axs, hsp, [tdecy*1e6, axs[0].get_xlim()[1]], "RF\ndecay", right=False)

axs[3].set_visible(False)

plt.savefig("figureSim.pdf", bbox_inches='tight')

# %%

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 3))
plt.subplots_adjust(hspace=0.15)
axs[1].plot(time_trace3*1e6, det, 'k--', label=simlabel, zorder=10)
axs[0].plot(time_trace3*1e6, hbw/half_bandwidth*100, 'k--', zorder=5, label=simlabel)
axs[1].set_xlabel("time ($\\mu s$)")
[ax.grid(zorder=-1) for ax in axs]
axs[1].set_ylabel("$\Delta\hat\omega$ (Hz)")
axs[0].set_ylabel("$\hat{\omega}_{1/2}/\omega_{1/2}$ (\%)")

x = observer(probe=np.array([np.real(prbC), np.imag(prbC)], dtype=np.float64).T,
             forward=np.array([np.real(fwdC+beam/2), np.imag(fwdC+beam/2)], dtype=np.float64).T,
             half_bandwidth_ext=half_bandwidth,
             detuning_init=0,
             sample_rate=fs,
             observer_bandwidth=rfBandwidth,
             amplitude_threshold=Amin,
             bandwidth_gain_factor=1,
             detuning_gain_factor=1
             )

axs[0].plot(time_trace3*1e6, (half_bandwidth+x[:, 2])*100/half_bandwidth,
            label="LO", marker=".", markevery=2000, markersize=7)
axs[1].plot(time_trace3*1e6, -x[:, 3], label="LO", marker=".", markevery=1000,
            markersize=7)

hbwrls, detrls, _ = rls(1/fs, fwdC+beam/2, prbC,
                     half_bandwidth*1, rfBandwidth, 0,
                     amin=1, pT0=1e-6, init=True)

axs[0].plot(time_trace3*1e6, hbwrls*100/half_bandwidth, label="RLS")
axs[1].plot(time_trace3*1e6, detrls, label="RLS")


axs[0].set_xlim(xlim)
axs[0].set_xticks([0, 750,1500])
axs[0].set_yticks(np.arange(80, 140, 10))
axs[0].set_ylim(95, 117)
axs[1].set_ylim(-27, 64)
axs[1].set_yticks([-25, 0, 25,50])

axs[0].legend(framealpha=1)
fig.align_ylabels()
plt.savefig("figureSimRls.pdf", bbox_inches='tight')
