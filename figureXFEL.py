
import matplotlib as mpl
import colorsys
import numpy as np
from observer import luenberger_observer as observer
import wget
from os.path import exists

from inverseModel import inverseModel
from calibrate import cal_c
from rls import rls

filename = "SP_XFEL.npz"
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


# %% XFEL quench calibrated with beam (from trip snapshot)

def AP2C(AP):
    return (AP[0, :] * np.exp(1j/180*np.pi*AP[1, :])).T

delaysamples = 22
calibrationpulses = 100
data = np.load("SP_XFEL.npz")
prbRaw = AP2C(data['prbAP'])
fwdRaw = AP2C(data['fwdAP'])
refRaw = AP2C(data['refAP'])
time_trace = data['total_time_us']/prbRaw.shape[1] * np.arange(prbRaw.shape[1])
S, bandwidth = cal_c(
    fwdRaw[:calibrationpulses],
    refRaw[:calibrationpulses],
    prbRaw[:calibrationpulses],
    windowlen=21, fs = 1e6, beam=True)
time_trace = time_trace[:-delaysamples]
fs = 1e6/(time_trace[1] - time_trace[0])

prbC = prbRaw[:, delaysamples:]
fwdC = S[0, 0] * fwdRaw[:, delaysamples:] + S[1, 0] * refRaw[:, delaysamples:]
refC = S[0, 1] * fwdRaw[:, delaysamples:] + S[1, 1] * refRaw[:, delaysamples:]

def plotXFEL(t, hbw0, hbw, det, hbw2=None, det2=None, axs=None, label=None, **kwargs):
    p0, = axs[0].step(t, hbw/hbw0*100, label="LO"+label, where="post", **kwargs)
    c0 = darken_color(p0.get_color())
    if hbw2 is not None:
        axs[0].step(t, hbw2/hbw0*100, '--', color=c0, alpha=1,
                    label="(4)"+label, linewidth=0.9, **kwargs)

    axs[1].step(t, det, label="LO"+label, where="post", **kwargs)
    if det2 is not None:
        axs[1].step(t, det2, '--', color=c0, alpha=1,
                    label="(4)"+label, linewidth=0.9, **kwargs)

    axs[0].set_ylabel(r"$\hat\omega_{1/2}/\hat\omega_{1/2}^{\mathrm{ext}}(\%$)")
    axs[1].set_ylabel(r"$\Delta\hat\omega$ (Hz)")
    axs[1].set_xlabel(r"time ($\mu$s)")
    axs[0].grid('on', zorder=-1)
    axs[1].grid('on', zorder=-1)
    return axs

def getLabel(index):
    labels = [" nominal", " quench"]
    try:
        return labels[index]
    except:
        return str(200-i)
amplthresh = 1
pole = 10_000

estimates = np.empty((prbC.shape[0], prbC.shape[1], 6))
for i in range(prbC.shape[0]):
    prbCi = np.ascontiguousarray(prbC[i])
    fwdCi = np.ascontiguousarray(fwdC[i])

    estimates[i, :, 0:2] = inverseModel(
        fwdCi, prbCi, fs=fs, f12=bandwidth,
        filtering=1, filtF=pole,
        minGradient=amplthresh, decimation=1)/2/np.pi

    estimates[i, :, 2:4] = observer(
        probe=np.array([np.real(prbCi), np.imag(prbCi)], dtype=np.float64).T,
        forward=np.array([np.real(fwdCi), np.imag(fwdCi)], dtype=np.float64).T,
        half_bandwidth_ext=bandwidth,
        detuning_init=0,
        sample_rate=fs,
        observer_bandwidth=pole,
        amplitude_threshold=amplthresh,
        bandwidth_gain_factor=1,
        detuning_gain_factor=1)[:, 2:4]

    hbwrls, detrls, _ = rls(
        1/fs, fwdCi, prbCi,
        f12=bandwidth, ffilt=pole, fdet0=0, amin=1,
        noise=np.zeros((time_trace.size, 2), dtype=complex),
        pT0=1, init=1
    )

    estimates[i, :, 4] = hbwrls
    estimates[i, :, 5] = detrls


# %% Plots

plotRLS = False

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 3))
plt.subplots_adjust(hspace=0.1)

# 0: nominal, 1: soft quench, 2: hard quench
for i in [0, 1]:

    plotXFEL(time_trace, bandwidth,
             hbw=bandwidth + estimates[200+i, :, 2],
             det=estimates[200+i, :, 3],
             hbw2=estimates[200+i, :, 0],
             det2=estimates[200+i, :, 1],
             axs=axs,
             label=getLabel(i)
     )

    if plotRLS:
        axs[0].plot(time_trace, estimates[200+i, :, 4]/bandwidth*100,
                    label=f"RLS {getLabel(i)}")
        axs[1].plot(time_trace, -estimates[200+i, :, 5])


statistics = 200
means = np.mean(estimates[:statistics], axis=0)
stds = np.std(estimates[:statistics], axis=0)

col = ['k', 'g']
for i in [0, 1]:
    delta = 6*stds[:, 2*i]
    axs[0].fill_between(
        time_trace,
        100*(i+(means[:, 2*i]-delta)/bandwidth),
        100*(i+(means[:, 2*i]+delta)/bandwidth),
        color=col[i], alpha=0.3,   # alpha controls transparency
        label='Confidence interval'
    )
    delta = 6*stds[:, 1+2*i]
    # delta = 3*stds[:, 1+2*i]/np.sqrt(statistics)
    axs[1].fill_between(
        time_trace,
        means[:, 1+2*i]-delta,
        means[:, 1+2*i]+delta,
        color=col[i], alpha=0.3,   # alpha controls transparency
        label='Confidence interval'
    )


if plotRLS:
    axs[0].set_ylim(-150, 250)
    axs[1].set_ylim(-45, 245)
else:
    axs[0].set_ylim(94, 115)
    axs[1].set_ylim(-45, 85)

axs[0].set_xlim(-0, 1600)


fig.align_ylabels(axs)
fig.legend([_l.get_label() for _l in axs[0].lines if not _l.get_label().startswith('_')], ncols=2,
           bbox_to_anchor=(0.51, 0.97+plotRLS*0.04), loc="center")

xticks = np.array([0, 750, 775, 1360, 1400])
xtickls = [str(xt) for xt in xticks]
axs[-1].set_xticks(xticks, xtickls)
labels = axs[-1].get_xticklabels()
offset = mtf.ScaledTranslation(10/72, 0, fig.dpi_scale_trans)
labels[1].set_transform(labels[1].get_transform() +
                        mtf.ScaledTranslation(-8/72, 0, fig.dpi_scale_trans))
labels[2].set_transform(labels[2].get_transform() +
                        mtf.ScaledTranslation(16/72, 0.05, fig.dpi_scale_trans))
labels[3].set_transform(labels[3].get_transform() +
                        mtf.ScaledTranslation(-16/72, 0.05, fig.dpi_scale_trans))
labels[4].set_transform(labels[4].get_transform() +
                        mtf.ScaledTranslation(12/72, 0, fig.dpi_scale_trans))

plt.savefig("figureXFEL.pdf", bbox_inches='tight')


# %%


fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 3))
plt.subplots_adjust(hspace=0.1)


for i in [0, 1]:

    plotXFEL(time_trace, bandwidth,
             hbw=bandwidth + estimates[200+i, :, 2],
             det=estimates[200+i, :, 3],
             axs=axs,
             label=getLabel(i),
             marker=".",
             markevery=(50, 100),
             markersize=(1-i)*7
     )

    axs[0].plot(time_trace, estimates[200+i, :, 4]/bandwidth*100,
                label=f"RLS {getLabel(i)}", marker=".", markevery=100,
                            markersize=(1-i)*7)
    axs[1].plot(time_trace, -estimates[200+i, :, 5], marker=".",
                markevery=100,
                markersize=(1-i)*7)

    # hbwrls, detrls, _ = rls(
    #     1/fs, fwdC[200+i]*np.exp(-12j/180*np.pi), prbC[200+i],
    #     bandwidth, ffilt=pole, fdet0=150, amin=1,
    #     noise=np.zeros((time_trace.size, 2), dtype=complex),
    #     pT0=1, init=True
    # )
    # axs[1].plot(time_trace, -detrls)

    # hbwrls, detrls, _ = rls(
    #     1/fs, 4.8*fwdC[200+i]*np.exp(78j/180*np.pi), prbC[200+i],
    #     bandwidth, ffilt=pole, fdet0=150, amin=1, init=True
    # )
    # axs[0].plot(time_trace, hbwrls/bandwidth*100, label=f"RLS {getLabel(i)} (scaled/rotated fwd)")

axs[0].set_ylim(-180, 250)
axs[1].set_ylim(-45, 245)
fig.align_ylabels()
axs[0].set_xticks([750, 1400])
axs[0].set_yticks([-100, 0, 100, 200])

fig.legend([_l.get_label() for _l in axs[0].lines if not _l.get_label().startswith('_')], ncols=2,
           bbox_to_anchor=(0.51, 0.97+plotRLS*0.04), loc="center")

plt.savefig("figureXFELrls.pdf", bbox_inches='tight')
