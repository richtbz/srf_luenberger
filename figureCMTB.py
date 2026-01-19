import matplotlib as mpl
import colorsys
import numpy as np
from observer import luenberger_observer as observer
import wget
from os.path import exists

from inverseModel import inverseModel
from rls import rls

plotRLS = False

filename = "CW_CMTB.npz"
if not exists(filename):
    file_url = "https://zenodo.org/api/records/18292997/files/" + filename + "/content"
    wget.download(file_url)

plt = mpl.pyplot
mtf = mpl.transforms
mcl = mpl.colors

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 11
mpl.rcParams['font.family'] = 'Latin Modern Roman'

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def darken_color(color, amount=0.7):
    """
    Darken the given color by multiplying its lightness by 'amount'.
    'color' can be a matplotlib color string, hex string, or RGB tuple.
    'amount' should be between 0 (black) and 1 (original color).
    """
    try:
        c = mcl.cnames[color]
    except:
        c = color
    rgb = mcl.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, max(0, l * amount), s)


# %% CMTB quench CW (SEL)

data = np.load("CW_CMTB.npz")
sel = np.s_[-1600:]
time_trace2 = data["timevector"][sel]
toffs = 0.933
fs = 1/(time_trace2[1]-time_trace2[0])
prbC2 = data["probeAmpl"][sel] * np.exp(1j*data["probePhas"][sel]/180*np.pi)
fwdC = 1.28*data["forwdAmpl"][sel] * np.exp(1j*data["forwdPhas"][sel]/180*np.pi)
refC = data["refleAmpl"][sel] * np.exp(1j*data["reflePhas"][sel]/180*np.pi)

bandwidth2 = 141
pole = 2000

x2 = observer(
    probe=np.array([np.real(prbC2), np.imag(prbC2)], dtype=np.float64).T,
    forward=np.array([np.real(fwdC), np.imag(fwdC)], dtype=np.float64).T,
    half_bandwidth_ext=bandwidth2,
    detuning_init=0,
    sample_rate=fs,
    observer_bandwidth=pole,
    amplitude_threshold=1,
    bandwidth_gain_factor=1,
    detuning_gain_factor=1,
    I_init=np.real(prbC2[0]),
    Q_init=np.imag(prbC2[0]))

f12det = inverseModel(fwdC, prbC2, fs=fs, f12=bandwidth2, smoothingWindow=40,
                      filtering=1, filtF=pole)/2/np.pi

hbwrls, detrls, _ = rls(1/fs, fwdC, prbC2,
                     f12=bandwidth2, ffilt=pole, fdet0=0,
                     amin=1, noise=np.zeros((time_trace2.size, 2)), pT0=100)

def stretchplot(ax, time, tfactor, tsplit, data, **kwargs):
    s2 = np.s_[np.argwhere(time >= tsplit)[0][0]:]

    timenew = time.copy()
    timenew[s2] = (timenew[s2]-tsplit)*tfactor+tsplit

    ax.step(timenew, data, **kwargs)

time = (time_trace2-toffs)*1e3
tfactor = 20
tsplit = 50
isplit = np.argmin(np.abs(time-tsplit))

loest = (bandwidth2 + x2[:, 2])/bandwidth2*100
rffwd = np.abs(fwdC)/np.mean(np.abs(fwdC[:isplit]))*100
rfprb = np.abs(prbC2)/np.mean(np.abs(prbC2[:isplit]))*100

plt.figure(figsize=(4, 2.25))
axs = plt.gca()
stretchplot(axs, time, tfactor, tsplit,
            loest,
            label="LO bandwidth estimate", where="post", color=colors[0])
stretchplot(axs, time, tfactor, tsplit,
            rffwd, alpha=1, color=colors[1],
            label="RF drive amplitude")
stretchplot(axs, time, tfactor, tsplit,
            rfprb, alpha=1, color=colors[2],
            label="RF probe amplitude")

# statistical bars
plt.fill_between(
    [0, 100],
    100+6*np.ones((2,))*np.std(loest[:isplit]),
    100-6*np.ones((2,))*np.std(loest[:isplit]),
    color=colors[0], alpha=0.15
)
plt.fill_between(
    [0, 100],
    100+6*np.ones((2,))*np.std(rfprb[:isplit]),
    100-6*np.ones((2,))*np.std(rfprb[:isplit]),
    color=colors[2], alpha=0.3
)


if plotRLS:
    stretchplot(axs, (time_trace2-toffs)*1e3, tfactor, tsplit,
                hbwrls/bandwidth2*100, where="post", color=colors[4],
                label="RLS bandwidth estimate")
    plt.ylim(-7, 117)
else:
    plt.ylim(98, 104)



plt.ylabel(r"relative to nominal ($\%$)")
plt.legend()

tickspacing = 10
ticks = [t for t in np.arange(0, tsplit, tickspacing)]+\
        [t for t in np.arange(tsplit, tsplit*2, tfactor)]
axs.set_xticks(ticks)
axs.set_xticklabels(
    [str(t) for t in ticks[:tsplit//tickspacing+1]] +
    [str(tsplit + (t-tsplit)/tfactor) for t in ticks[tsplit//tickspacing+1:]]
    )
plt.xticks(ticks+[tsplit-tickspacing/20])
plt.xlim(1, 5 + 1e3*(1-toffs))
plt.xlabel("time (ms)")

axs.grid()
plt.legend(framealpha=1)

plt.savefig("figureCMTB.pdf", bbox_inches='tight')

# %%



fig = plt.figure(figsize=(4.4, 2.4))
axs = plt.gca()
stretchplot(axs, time, tfactor, tsplit,
            loest,
            label="LO $\Delta\hat\omega_{1/2}/\Delta\hat\omega_{1/2}^{\mathrm{ext}}$ (\%)",
            where="post", color=colors[0])
stretchplot(axs, (time_trace2-toffs)*1e3, tfactor, tsplit,
            hbwrls/bandwidth2*100, where="post", color=colors[0],
            label=r"RLS $\Delta\hat\omega_{1/2}/\Delta\hat\omega_{1/2}^{\mathrm{ext}}$ (\%)",
            linestyle='--')
stretchplot(axs, (time_trace2-toffs)*1e3, tfactor, tsplit,
            -x2[:, 3], where="post", color=colors[1],
            label=r"LO $\Delta\hat\omega$ (Hz)")
stretchplot(axs, (time_trace2-toffs)*1e3, tfactor, tsplit,
            detrls, where="post", color=colors[1],
            label=r"RLS $\Delta\hat\omega$ (Hz)",
            linestyle='--')
plt.ylim(-7, 122)

tickspacing = 10
ticks = [t for t in np.arange(0, tsplit, tickspacing)]+\
        [t for t in np.arange(tsplit, tsplit*2, tfactor)]
axs.set_xticks(ticks)
axs.set_xticklabels(
    [str(t) for t in ticks[:tsplit//tickspacing+1]] +
    [str(tsplit + (t-tsplit)/tfactor) for t in ticks[tsplit//tickspacing+1:]]
    )
plt.xticks(ticks+[tsplit-tickspacing/20])
plt.xlim(1, 10 + 1e3*(1-toffs))
plt.xlabel("time (ms)")

axs.grid()
# plt.legend(framealpha=1)
fig.legend(ncols=2,
           bbox_to_anchor=(0.48, 1.03), loc="center")

plt.savefig("figureCMTBrls.pdf", bbox_inches='tight')
