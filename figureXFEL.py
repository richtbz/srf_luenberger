
import matplotlib as mpl
import colorsys
import numpy as np
from observer import luenberger_observer as observer
import wget
from os.path import exists

from inverseModel import inverseModel
from calibrate import cal_c
from rls import rls, lpf

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
        minGradient=amplthresh, decimation=1,
        # zeta=1/2**0.5, filtOrd=2)/2/np.pi
        zeta=1, filtOrd=2)/2/np.pi

    estimates[i, :, 2:4] = observer(
        probe=np.array([np.real(prbCi), np.imag(prbCi)], dtype=np.float64).T,
        forward=np.array([np.real(fwdCi), np.imag(fwdCi)], dtype=np.float64).T,
        half_bandwidth_ext=bandwidth,
        detuning_init=0,
        sample_rate=fs,
        observer_bandwidth=pole,
        amplitude_threshold=amplthresh,
        # bandwidth_gain_factor=1.4, detuning_gain_factor=1.4
        bandwidth_gain_factor=1, detuning_gain_factor=1
        )[:, 2:4]

    hbwrls, detrls, _ = rls(
        1/fs, fwdCi, prbCi,
        f12=bandwidth, ffilt=pole, fdet0=0, amin=1,
        noise=np.zeros((time_trace.size, 2), dtype=complex),
        pT0=1, init=1
    )

    estimates[i, :, 4] = hbwrls
    estimates[i, :, 5] = detrls

estimates[..., 2] += bandwidth
estimates[..., 5] *= -1

# %% Plots

plotRLS = False
filterRLS = False
plotInv = False

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 3))
plt.subplots_adjust(hspace=0.1)

# 0: nominal, 1: soft quench, 2: hard quench
for i in [0, 1]:

    axs[0].step(time_trace, (estimates[200+i, :, 2]/bandwidth)*100,
                label=getLabel(i), where="post")
    axs[1].step(time_trace, estimates[200+i, :, 3],
                label=getLabel(i), where="post")

    if plotInv:
        axs[0].step(time_trace, (estimates[200+i, :, 0]/bandwidth)*100,
                    label="inv"+getLabel(i), where="post",
                    color = darken_color(f"C{i}", 0.5), linestyle="--", linewidth=1)
        axs[1].step(time_trace, estimates[200+i, :, 1],
                    label="inv"+getLabel(i), where="post",
                    color = darken_color(f"C{i}", 0.5), linestyle="--", linewidth=1)


    if plotRLS:
        if filterRLS:
            hbw_plt = lpf(estimates[200+i, :, 4], pole, 1/fs)
            det_plt = lpf(estimates[200+i, :, 5], pole, 1/fs)
        else:
            hbw_plt = estimates[200+i, :, 4]
            det_plt = estimates[200+i, :, 5]
        axs[0].plot(time_trace, hbw_plt/bandwidth*100,
                    label=f"RLS {getLabel(i)}",
                    zorder=-1, linewidth=1, alpha=0.5)
        axs[1].plot(time_trace, det_plt,
                    zorder=-1, linewidth=1, alpha=0.5)

statistics = 200
means = np.mean(estimates[:statistics], axis=0)
stds = np.std(estimates[:statistics], axis=0)

if filterRLS:
    means[..., 4:] = np.mean(lpf(estimates[:statistics, :, 4:], pole, 1/fs), axis=0)
    stds[..., 4:] = np.std(lpf(estimates[:statistics, :, 4:], pole, 1/fs), axis=0)

col = ['g', 'k', 'r']
arr = [1]
if plotInv:
    arr.append(0)
if plotRLS:
    arr.append(2)
for i in arr:
    delta = 6*stds[:, 2*i]
    axs[0].fill_between(
        time_trace,
        100*((means[:, 2*i]-delta)/bandwidth),
        100*((means[:, 2*i]+delta)/bandwidth),
        color=col[i], alpha=0.3,   # alpha controls transparency
        label=r'mean $\pm 6\sigma$',
        edgecolor='none'
    )
    delta = 6*stds[:, 1+2*i]
    # delta = 3*stds[:, 1+2*i]/np.sqrt(statistics)
    axs[1].fill_between(
        time_trace,
        means[:, 1+2*i]-delta,
        means[:, 1+2*i]+delta,
        color=col[i], alpha=0.3,   # alpha controls transparency
        label='Confidence interval',
        edgecolor='none'
    )


axs[0].set_ylim(94, 115)
axs[1].set_ylim(-45, 85)

axs[0].set_xlim(-0, 1600)

axs[0].set_ylabel(r"$\hat\omega_{1/2}/\hat\omega_{1/2}^{\mathrm{ext}}(\%$)")
axs[1].set_ylabel(r"$\Delta\hat\omega$ (Hz)")
axs[1].set_xlabel(r"time ($\mu$s)")
axs[0].grid('on', zorder=-1, alpha=0.3)
axs[1].grid('on', zorder=-1, alpha=0.3)


fig.align_ylabels(axs)
fig.legend(*axs[0].get_legend_handles_labels(), ncols=3,
           bbox_to_anchor=(0.44, 0.97+plotRLS*0.04), loc="center")

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

estimates2 = estimates.copy()
for i in range(prbC.shape[0]):
    prbCi = np.ascontiguousarray(prbC[i])
    fwdCi = np.ascontiguousarray(fwdC[i])

    estimates2[i, :, 0:2] = inverseModel(
        fwdCi, prbCi, fs=fs, f12=bandwidth,
        filtering=1, filtF=pole,
        minGradient=amplthresh, decimation=1,
        zeta=1/2**0.5, filtOrd=2)/2/np.pi

    estimates2[i, :, 2:4] = observer(
        probe=np.array([np.real(prbCi), np.imag(prbCi)], dtype=np.float64).T,
        forward=np.array([np.real(fwdCi), np.imag(fwdCi)], dtype=np.float64).T,
        half_bandwidth_ext=bandwidth,
        detuning_init=0,
        sample_rate=fs,
        observer_bandwidth=pole,
        amplitude_threshold=amplthresh,
        bandwidth_gain_factor=1.4, detuning_gain_factor=1.4
        )[:, 2:4]

estimates2[..., 2] += bandwidth
statistics = 200
stds = np.std(estimates2[:statistics], axis=0)
means = np.mean(estimates2[:statistics], axis=0)

# %%
#
hsp = 0.1
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 3))
plt.subplots_adjust(hspace=hsp)

axrect1 = [0.53, 0.65,
          0.29, 1.2]
axrect2 = [0.53, 0.18,
          0.29, 0.44]

axs = np.append(axs, axs[1].inset_axes(axrect1, zorder=20))
axs = np.append(axs, axs[1].inset_axes(axrect2, zorder=20))

rect1 = [735, 880, 136.5, 154]
rect2 = [735, 880, -5, 15]

mean = 1
if mean:
    plotme=means
    lab = ''
else:
    plotme=estimates2[201]
    for j in [0, 2]:
        axs[j].step(time_trace, estimates2[200, :, 2],
                    label="LO nominal", where="post", color='k')
    axs[1].step(time_trace, estimates2[200, :, 3],
                label="LO nominal", where="post", color='k')
    lab = 'quench'
    i=1


hbw_plt = plotme[:, 4]
det_plt = plotme[:, 5]
for j in [0, 2]:
    axs[j].step(time_trace, plotme[:, 2],
                label="LO "+lab, where="post",
                marker='o', markersize=4, markevery=200)
    axs[j].step(time_trace, plotme[:, 0],
                label="inv"+lab, where="post")
    axs[j].plot(time_trace, hbw_plt,
                label=f"RLS {lab}",
                zorder=-1,  alpha=0.75)

for j in [1, 3]:
    axs[j].step(time_trace, plotme[:, 3],
            label="LO "+lab, where="post",
            marker='o', markersize=4, markevery=200)
    axs[j].step(time_trace, plotme[:, 1],
                label="inv"+lab, where="post")
    axs[j].plot(time_trace, det_plt,
                zorder=-1,  alpha=0.75)



ylim1=(135, 155)
axs[0].set_ylim(ylim1)
ylim2=(-25, 185)
axs[1].set_ylim(ylim2)

xlim = (0, 1600)
axs[0].set_xlim(xlim)

axs[0].set_ylabel(r"$\mu(\hat\omega_{1/2})$ (Hz)")
axs[1].set_ylabel(r"$\mu(\Delta\hat\omega)$ (Hz)")
axs[1].set_xlabel(r"time ($\mu$s)")
axs[0].grid('on', zorder=-1, alpha=0.3)
axs[1].grid('on', zorder=-1, alpha=0.3)

axs[2].set_xlim(rect1[:2])
axs[2].set_ylim(rect1[2:])
axs[0].plot([rect1[0],rect1[1],rect1[1],rect1[0],rect1[0]],
            [rect1[2],rect1[2],rect1[3],rect1[3],rect1[2]],
            'k', linewidth=1, zorder=-1)
axs[0].plot([rect1[1], xlim[0]+(axrect1[0]+axrect1[2])*np.diff(xlim)[0]],
            [rect1[3], ylim1[0]+(axrect1[1]+axrect1[3]-1-hsp)*np.diff(ylim1)[0]],
            'k', linewidth=1, zorder=0)
axs[0].plot([rect1[0], xlim[0]+(axrect1[0])*np.diff(xlim)[0]],
            [rect1[3], ylim1[0]+(axrect1[1]+axrect1[3]-1-hsp)*np.diff(ylim1)[0]],
            'k', linewidth=1, zorder=0)

con = mpl.patches.ConnectionPatch(
    xyA=(rect1[0], rect1[2]), xyB=(rect1[0], rect1[2]),
    coordsA="data", coordsB="data",
    axesA=axs[0], axesB=axs[2],
    arrowstyle="-", linewidth=1, color="k",
    zorder=10
)
fig.patches.append(con)

axs[3].set_xlim(rect2[:2])
axs[3].set_ylim(rect2[2:])
axs[1].plot([rect2[0],rect2[1],rect2[1],rect2[0],rect2[0]],
            [rect2[2],rect2[2],rect2[3],rect2[3],rect2[2]],
            'k', linewidth=1, zorder=10)
axs[1].plot([rect2[0], xlim[0]+(axrect2[0])*np.diff(xlim)[0]],
            [rect2[3], ylim2[0]+(axrect2[1]+axrect2[3])*np.diff(ylim2)[0]],
            'k', linewidth=1, zorder=10)
axs[1].plot([rect2[0], xlim[0]+(axrect2[0])*np.diff(xlim)[0]],
            [rect2[2], ylim2[0]+(axrect2[1])*np.diff(ylim2)[0]],
            'k', linewidth=1, zorder=10)
axs[1].plot([rect2[1], xlim[0]+(axrect2[0]+axrect2[2])*np.diff(xlim)[0]],
            [rect2[2], ylim2[0]+(axrect2[1])*np.diff(ylim2)[0]],
            'k', linewidth=1, zorder=10)

axs[2].set_xticks([750, 775])
axs[2].set_yticks([140])
axs[2].set_xticklabels([])
axs[2].set_yticklabels([])
axs[2].grid(zorder=-1, alpha=0.3)
axs[3].set_xticks([750, 775])
axs[3].set_yticks([0])
axs[3].set_xticklabels([])
axs[3].set_yticklabels([])
axs[3].grid(zorder=-1, alpha=0.3)

fig.align_ylabels(axs)
fig.legend(*axs[0].get_legend_handles_labels(), ncols=2+mean,
           bbox_to_anchor=(0.51, 0.97+plotRLS*0.04), loc="center")

xticks = np.array([0, 750, 775, 1400])
xtickls = [str(t) for t in xticks]
xtickls[1] = '750 775'
xtickls[2] = ''
axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xtickls)


plt.savefig("figureCompare.pdf", bbox_inches='tight')

# %%


col = ["C1", "C0", "C2"]
leg = ["inv.", "LO", "RLS"]
mark = ['', 'o', '']
alp = [1, 1, 0.75]

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 3))
plt.subplots_adjust(hspace=0.1)

for i in [1, 0, 2]:
    axs[0].plot(
        time_trace,
        stds[:, 2*i],
        color=col[i],
        marker = mark[i],
        markersize=4,
        markevery=200,
        label=leg[i],
        alpha = alp[i]
    )
    axs[1].plot(
        time_trace,
        stds[:, 1+2*i],
        color=col[i],
        marker = mark[i],
        markersize=4,
        markevery=200,
        label=leg[i],
        alpha = alp[i]
    )

axs[1].set_xlim(0, 1800)

axs[0].set_ylim(-1, 7)
axs[1].set_ylim(-1, 7)

axs[0].set_yticks([0, 2, 4, 6])
axs[1].set_yticks([0, 2, 4, 6])

fig.legend(*axs[0].get_legend_handles_labels(), ncols=3,
           bbox_to_anchor=(0.51, 0.97), loc="center")

xticks = np.array([0, 750, 1400])
axs[-1].set_xticks(xticks)
for ax in axs:
    ax.grid(zorder=-1, alpha=0.3)

axs[0].set_ylabel(r"$\sigma(\hat\omega_{1/2})$ (Hz)")
axs[1].set_ylabel(r"$\sigma(\Delta\hat\omega)$ (Hz)")
axs[1].set_xlabel(r"time ($\mu$s)")
fig.align_ylabels()
plt.savefig("figureCompareStd.pdf", bbox_inches='tight')
