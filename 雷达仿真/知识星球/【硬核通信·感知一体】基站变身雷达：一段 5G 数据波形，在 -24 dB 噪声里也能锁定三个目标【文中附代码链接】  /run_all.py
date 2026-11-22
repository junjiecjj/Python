"""Runs every experiment and writes figures and numbers to results/."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isac import *

OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
p = Params()
print(f"range res {p.range_res:.2f} m, max {p.range_max:.0f} m, speed res {p.vel_res:.2f} m/s, max {p.vel_max:.1f} m/s")
rng = np.random.default_rng(7)
targets = [(40.0, 12.0, 1.0), (75.0, -20.0, 0.5), (118.0, 6.0, 0.25)]

# 1. Range-Doppler map
tx = qam(16, (p.n_sub, p.n_sym), rng)
rx = echo(p, targets, tx, -5, rng)
rd = range_doppler(p, rx, tx)
det = cfar(rd)
vel = (np.arange(p.n_sym) - p.n_sym // 2) * p.vel_res
rng_ax = np.arange(p.n_sub) * p.range_res
show = rng_ax <= 160
fig, a = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
im = a.pcolormesh(vel, rng_ax[show], 10 * np.log10(rd[show] / rd.max() + 1e-6), cmap="magma", vmin=-40, vmax=0, shading="auto")
for r, v, _ in targets:
    a.add_patch(plt.Circle((v, r), 4, fc="none", ec="cyan", lw=1.4))
a.set(xlabel="Speed (m/s)", ylabel="Range (m)", title="Three targets seen by a 16-QAM data signal at -5 dB SNR")
a.grid(False)
fig.colorbar(im, label="Power (dB)")
fig.savefig(OUT / "range_doppler.png", dpi=170)
plt.close(fig)

# 2. Detection vs SNR, sharing resources between data and sensing
TRIALS, SNRS = 200, np.arange(-34, -5, 2)
TGT = (60.0, 15.0, 1.0)
res = {}
for spacing in (1, 2, 4, 8):
    mask = pilot_mask(p, spacing)
    pd, fa, rmse = [], [], []
    for snr in SNRS:
        hit, false, err = 0, 0, []
        for t in range(TRIALS):
            r = np.random.default_rng(10000 * spacing + 100 * int(snr + 50) + t)
            tx = qam(16, (p.n_sub, p.n_sym), r)
            rd = range_doppler(p, echo(p, [TGT], tx, snr, r), tx, mask)
            d = cfar(rd)
            ck, cm = cell_of(p, TGT[0], TGT[1])
            near = np.zeros_like(d)
            near[max(ck - 3, 0):ck + 4, max(cm - 3, 0):cm + 4] = True
            found = (d & near).any()
            hit += found
            false += int((d & ~near).sum())
            if found:
                sub = np.where(near, rd, 0)
                pk = np.unravel_index(np.argmax(sub), sub.shape)
                err.append(refine_range(rd, pk, p) - TGT[0])
        pd.append(float(hit) / TRIALS)
        fa.append(false / TRIALS)
        rmse.append(float(np.sqrt(np.mean(np.square(err)))) if len(err) >= 20 else None)
    res[spacing] = dict(pd=pd, false_per_frame=fa, rmse=rmse)
    print(spacing, [round(float(x), 2) for x in pd])

need = {}
for s, v in res.items():
    ok = [int(x) for x, y in zip(SNRS, v["pd"]) if y >= .9]
    need[s] = ok[0] if ok else None
print("SNR needed for 90% detection", need)

cols = {1: "#1a73e8", 2: "#12b5cb", 4: "#e8710a", 8: "#d01884"}
lab = lambda s: f"{100 / s:.1f}% of subcarriers"
fig, a = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
for s, v in res.items():
    a.plot(SNRS, 100 * np.array(v["pd"]), "o-", ms=4, lw=2, c=cols[s], label=lab(s))
a.set(xlabel="SNR per resource element (dB)", ylabel="Detection probability (%)", title="Sensing with fewer subcarriers needs a stronger echo")
a.legend(frameon=False)
fig.savefig(OUT / "detection_vs_snr.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 4.3), constrained_layout=True)
for s, v in res.items():
    x = [i for i, e in zip(SNRS, v["rmse"]) if e is not None]
    y = [e for e in v["rmse"] if e is not None]
    a.semilogy(x, y, "o-", ms=4, lw=2, c=cols[s], label=lab(s))
a.axhline(p.range_res, c="k", ls="--", label=f"Range cell {p.range_res:.2f} m")
a.set(xlabel="SNR per resource element (dB)", ylabel="Range RMSE (m)", title="Range accuracy of detected targets")
a.legend(frameon=False)
fig.savefig(OUT / "range_rmse.png", dpi=170)
plt.close(fig)

# 3. Data rate against sensing
fig, a = plt.subplots(figsize=(7.5, 4.3), constrained_layout=True)
x = [100 * (1 - 1 / s) for s in res]
y = [need[s] if need[s] is not None else np.nan for s in res]
a.plot(x, y, "o-", c="#1a73e8", lw=2.4, ms=8)
for xi, yi, s in zip(x, y, res):
    a.annotate(f"1 of every {s}", (xi, yi), textcoords="offset points", xytext=(6, 8))
a.set(xlabel="Subcarriers not used for sensing (%)", ylabel="SNR needed for 90% detection (dB)", title="Pilot-only sensing: more data subcarriers, less sensitivity")
fig.savefig(OUT / "tradeoff.png", dpi=170)
plt.close(fig)

ber = {o: [ber_awgn(p, o, s, np.random.default_rng(3)) for s in (10, 15, 20)] for o in (4, 16, 64)}
print("symbol error", ber)
json.dump({"params": dict(fc=p.fc, df=p.df, n_sub=p.n_sub, n_sym=p.n_sym), "range_res_m": p.range_res,
           "range_max_m": p.range_max, "speed_res_ms": p.vel_res, "speed_max_ms": p.vel_max,
           "snr_axis": [int(s) for s in SNRS], "by_spacing": res, "snr_for_90pct": need, "ser": ber},
          open(OUT / "summary.json", "w"), indent=2)
for s, v in res.items():
    print(s, "false/frame at -14", v["false_per_frame"][list(SNRS).index(-14)], "rmse at -14", v["rmse"][list(SNRS).index(-14)])

# More studies
def detect_rate(par, order, snr, trials, seed, tgt=TGT):
    ok = 0
    for t in range(trials):
        r = np.random.default_rng(seed + t)
        tx = qam(order, (par.n_sub, par.n_sym), r)
        d = cfar(range_doppler(par, echo(par, [tgt], tx, snr, r), tx))
        ck, cm = cell_of(par, tgt[0], tgt[1])
        ok += d[max(ck - 3, 0):ck + 4, max(cm - 3, 0):cm + 4].any()
    return float(ok) / trials


snrs2 = np.arange(-30, -15, 1)
mods = {o: [detect_rate(p, o, int(s), 150, 5000 + o * 1000 + int(s + 40) * 300) for s in snrs2] for o in (4, 16, 64)}
print("modulation", {k: [round(x, 2) for x in v] for k, v in mods.items()})
fig, a = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
for (o, v), c in zip(mods.items(), ("#12b5cb", "#1a73e8", "#d01884")):
    a.plot(snrs2, 100 * np.array(v), "o-", ms=4, lw=2, c=c, label=f"{o}-QAM")
a.set(xlabel="SNR per resource element (dB)", ylabel="Detection probability (%)", title="Does the data modulation hurt the radar?")
a.legend(frameon=False)
fig.savefig(OUT / "modulation_effect.png", dpi=170)
plt.close(fig)

frames = {}
for m_ in (32, 64, 128, 256):
    q_ = Params(n_sym=m_)
    frames[m_] = [detect_rate(q_, 16, int(s), 100, 90000 + m_ * 100 + int(s + 40) * 200) for s in snrs2]
need_f = {m_: next((int(s) for s, y in zip(snrs2, v) if y >= .9), None) for m_, v in frames.items()}
print("frame length", need_f)
fig, a = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
for (m_, v), c in zip(frames.items(), ("#f4b400", "#12b5cb", "#1a73e8", "#d01884")):
    a.plot(snrs2, 100 * np.array(v), "o-", ms=4, lw=2, c=c, label=f"{m_} symbols")
a.set(xlabel="SNR per resource element (dB)", ylabel="Detection probability (%)", title="Longer frames collect more energy")
a.legend(frameon=False)
fig.savefig(OUT / "frame_length.png", dpi=170)
plt.close(fig)

# Separating two equal targets at the same speed
seps = [2, 3, 4, 5, 6, 8, 10, 14, 20, 30]
split = []
for sep in seps:
    ok = 0
    for t in range(100):
        r = np.random.default_rng(300000 + sep * 1000 + t)
        tx = qam(16, (p.n_sub, p.n_sym), r)
        ph = np.exp(2j * np.pi * r.random(2))  # real reflectors have unrelated phases
        tg = [(60.0, 15.0, ph[0]), (60.0 + sep, 15.0, ph[1])]
        rd = range_doppler(p, echo(p, tg, tx, -5, r), tx)
        _, cm = cell_of(p, 60.0, 15.0)
        cut = rd[:, cm]
        k1, k2 = cell_of(p, 60.0, 15.0)[0], cell_of(p, 60.0 + sep, 15.0)[0]
        if k2 > k1 + 1 and cut[k1 + 1:k2].size:
            ok += cut[k1 + 1:k2].min() < 0.5 * min(cut[max(k1 - 1, 0):k1 + 2].max(), cut[k2 - 1:k2 + 2].max())
    split.append(ok / 100)
print("separation", dict(zip(seps, split)))
fig, a = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
a.plot(seps, 100 * np.array(split), "o-", c="#d01884", lw=2.4)
a.axvline(p.range_res, c="k", ls="--", label=f"Range cell {p.range_res:.2f} m")
a.set(xlabel="Distance between two targets (m)", ylabel="Seen as two targets (%)", title="Separating two targets at the same speed")
a.legend(frameon=False)
fig.savefig(OUT / "target_separation.png", dpi=170)
plt.close(fig)

# Hann window against no window: strong and weak target together
r = np.random.default_rng(11)
tx = qam(16, (p.n_sub, p.n_sym), r)
rx = echo(p, [(60.0, 15.0, 1.0), (84.0, 15.0, 0.01)], tx, 5, r)
_, cm = cell_of(p, 60.0, 15.0)
est = rx / tx
raw_ = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifft(est, axis=0), axis=1), axes=1)[:, cm]) ** 2
han = range_doppler(p, rx, tx)[:, cm]
fig, a = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
ax_ = np.arange(p.n_sub) * p.range_res
s_ = ax_ <= 130
a.plot(ax_[s_], 10 * np.log10(raw_[s_] / raw_.max() + 1e-9), c="#9aa0a6", lw=1.6, label="No window")
a.plot(ax_[s_], 10 * np.log10(han[s_] / han.max() + 1e-9), c="#1a73e8", lw=2, label="Hann window")
a.axvline(84, c="#d01884", ls="--", label="Weak target, -40 dB")
a.set(xlabel="Range (m)", ylabel="Power (dB)", ylim=(-70, 3), title="A window shows weak targets beside a strong one")
a.legend(frameon=False)
fig.savefig(OUT / "window_effect.png", dpi=170)
plt.close(fig)

snr3 = np.arange(0, 31, 2)
fig, a = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
for o, c in zip((4, 16, 64), ("#12b5cb", "#1a73e8", "#d01884")):
    y = [ber_awgn(p, o, int(s), np.random.default_rng(4)) for s in snr3]
    a.semilogy(snr3, np.maximum(y, 1e-6), "o-", ms=4, c=c, lw=2, label=f"{o}-QAM")
a.set(xlabel="SNR (dB)", ylabel="Symbol error rate", ylim=(1e-5, 1), title="Communication side: the same symbols carry data")
a.legend(frameon=False)
fig.savefig(OUT / "symbol_error.png", dpi=170)
plt.close(fig)

s_ = json.load(open(OUT / "summary.json"))
s_.update(modulation={str(k): v for k, v in mods.items()}, modulation_snr=[int(x) for x in snrs2], snr90_by_frame={str(k): v for k, v in need_f.items()}, separation=dict(zip(map(str, seps), split)))
json.dump(s_, open(OUT / "summary.json", "w"), indent=2)

fig, ax = plt.subplots(1, 4, figsize=(12, 2.5), facecolor="#2a0a1e")
for a, (v, l), c in zip(ax, [("-24 dB", "SNR for 90% detection"), ("0.20 m", "range error at -14 dB"), ("4.88 m", "range cell"), ("1 signal", "data and radar together")], ["#ff8fd0", "#7ee787", "#ffd166", "#79c0ff"]):
    a.set_facecolor("#3a1230"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color("#7a2a60")
    a.text(.5, .6, v, ha="center", va="center", fontsize=27, fontweight="bold", color=c, transform=a.transAxes)
    a.text(.5, .22, l, ha="center", va="center", fontsize=12, color="#f5e6f0", transform=a.transAxes)
fig.tight_layout(pad=1)
fig.savefig(OUT / "overview.png", dpi=170, facecolor=fig.get_facecolor())
