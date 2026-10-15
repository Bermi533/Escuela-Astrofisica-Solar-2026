"""
tools.py  --  THE TOOLBOX: everything that is NOT the interesting physics.
Loading the data, choosing cuts, the fitting machinery, and every plot and GIF.
Students do not need to read this file: the exercises call it in one line.

Sections
  1. Data and cuts           Cube, choose_cut, choose_cuts, td_along_line
  2. Fitting machinery       fit_curve, fit_sine, phase_amplitude, darkest_pixel
  3. Reference solutions     ref_build_td, ref_fit_profiles, ref_measure_cuts
                             (used by later exercises, so exercise 4 still
                              works even if exercise 2 is not finished)
  4. Pictures                plot_* (PNG figures) and gif_* (animations)
  5. Printing                report
"""
import os
import sys
import numpy as np
import matplotlib
INLINE = "ipykernel" in sys.modules          # True inside Colab / Jupyter
if not INLINE:
    matplotlib.use("Agg")                    # scripts: just save the files
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from astropy.io import fits
from scipy.optimize import curve_fit

DATA = os.environ.get("DATA", "data")        # folder holding the cube
GIFS, FIGS, FPS = "gifs", "figures", 12


# =========================================================================
# 1. DATA AND CUTS
# =========================================================================
class Cube:
    """Loads the FITS file and keeps the physical scales from the header."""

    def __init__(self, path=None):
        path = path or os.path.join(DATA, "fibril_cube.fits")
        self.data, self.h = fits.getdata(path, header=True)
        self.data = self.data.astype(float)                 # shape (t, y, x)
        self.dt = self.h["CDELT3"]                          # s
        self.pix_km = self.h["CDELT1"] * 725.0              # km per pixel
        self.nt, self.ny, self.nx = self.data.shape
        self.t = np.arange(self.nt) * self.dt
        self.y = np.arange(self.ny)
        print(f"Cube (t, y, x) = {self.data.shape}; cadence {self.dt:.0f} s; "
              f"pixel {self.pix_km:.1f} km; field {self.ny * self.pix_km / 1e3:.1f} x "
              f"{self.nx * self.pix_km / 1e3:.1f} Mm; duration {self.nt * self.dt:.0f} s")

    def km(self, pix):
        return np.asarray(pix) * self.pix_km

    def extent_img(self):                                   # in Mm, for imshow
        return [0, self.nx * self.pix_km / 1e3, 0, self.ny * self.pix_km / 1e3]


# Every student can work at their OWN cut positions:
#   a number -> that column; "random" -> a new random column each run;
#   any name -> a random column that is always the same for that name.
MARGIN = 5


def _rng(choice):
    if choice == "random":
        return np.random.default_rng()
    return np.random.default_rng(sum(ord(ch) * 31 ** i for i, ch in enumerate(choice.lower())))


def choose_cut(choice, nx):
    """One column (pix) for exercises 1-3."""
    choice = str(choice)
    if choice.lstrip("-").isdigit():
        xc = int(np.clip(int(choice), MARGIN, nx - 1 - MARGIN))
    else:
        xc = int(_rng(choice).integers(MARGIN, nx - MARGIN))
    print(f"Your cut: column x = {xc}")
    return xc


def choose_cuts(choice, nx, n=8):
    """Several columns for exercises 4-6: "default" (every 10 pix), "random",
    a name, or a list "12,40,77,130"."""
    choice = str(choice)
    if choice == "default":
        cuts = list(range(10, nx - 9, 10))
    elif "," in choice:
        cuts = sorted(int(np.clip(int(s), MARGIN, nx - 1 - MARGIN)) for s in choice.split(","))
    else:
        rng = _rng(choice)
        while True:
            cuts = np.sort(rng.choice(np.arange(MARGIN, nx - MARGIN), n, replace=False))
            if np.all(np.diff(cuts) >= 6) and cuts[-1] - cuts[0] > nx / 2:
                cuts = [int(v) for v in cuts]
                break
    print("Your cuts (columns):", cuts)
    return cuts


def td_along_line(data, p0, p1, width=1, step=1.0):
    """OPTIONAL: time-distance diagram along ANY straight line (an oblique cut),
    for real fibrils that are tilted or curved. p0, p1 = (x, y) end points in pix.
    The image is interpolated along the line; width > 1 averages parallel lines
    spaced ALONG the fibril. Keep the line PERPENDICULAR to the fibril: tilted
    by theta, it measures displacement and width too large by 1/cos(theta).
    Returns (td[s, t], s in pix)."""
    from scipy.ndimage import map_coordinates
    (x0, y0), (x1, y1) = p0, p1
    L = np.hypot(x1 - x0, y1 - y0)
    s = np.arange(0, L + 1e-9, step)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    offs = np.arange(width) - (width - 1) / 2
    td = np.zeros((len(s), data.shape[0]))
    for o in offs:
        xs, ys = x0 + ux * s + uy * o, y0 + uy * s - ux * o
        for k in range(data.shape[0]):
            td[:, k] += map_coordinates(data[k], [ys, xs], order=1, mode="nearest")
    return td / len(offs), s


# =========================================================================
# 2. FITTING MACHINERY (the maths library; the physics is in the exercises)
# =========================================================================
def fit_curve(model, x, y, p0):
    """Least-squares fit of model(x, *params) to y. Returns the best params."""
    p, _ = curve_fit(model, x, y, p0=p0, maxfev=10000)
    return p


def darkest_pixel(profile):
    """Darkest pixel, refined with a parabola (a good first guess)."""
    i = int(np.argmin(profile))
    if 0 < i < len(profile) - 1:
        a, b, c = profile[i - 1], profile[i], profile[i + 1]
        return i + 0.5 * (a - c) / (a - 2 * b + c)
    return float(i)


def _sine(t, c0, m, A, P, phi):
    return c0 + m * t + A * np.sin(2 * np.pi * t / P + phi)


def fit_sine(t, series):
    """Fits  c0 + m t + A sin(2 pi t / P + phi)  (a sine on top of a slow drift).
    Returns a dict with P (period), A (amplitude), and 'curve' / 'trend' for plots."""
    s = series - np.polyval(np.polyfit(t, series, 1), t)
    f = np.fft.rfftfreq(len(s), t[1] - t[0])
    P0 = 1 / f[1:][np.argmax(np.abs(np.fft.rfft(s))[1:])]           # first guess from FFT
    p, cov = curve_fit(_sine, t, series, p0=[series.mean(), 0, s.std() * np.sqrt(2), P0, 0],
                       maxfev=10000)
    if p[2] < 0:
        p[2] *= -1; p[4] += np.pi
    e = np.sqrt(np.diag(cov))
    return dict(P=p[3], A=p[2], P_err=e[3], A_err=e[2], mean=p[0],
                curve=_sine(t, *p), trend=p[0] + p[1] * t)


def phase_amplitude(t, series, P):
    """Phase (rad) and amplitude of the oscillation with a KNOWN period P
    (a linear least-squares fit of a sin + b cos + c + d t; very robust)."""
    w = 2 * np.pi / P
    M = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t), t])
    a, b, *_ = np.linalg.lstsq(M, series, rcond=None)[0]
    return np.arctan2(b, a), np.hypot(a, b)


def nuwt_track(td, t, depth=0.05, max_jump=3.0, max_gap=3, min_len=30):
    """Automatic thread tracking in the spirit of NUWT (Northumbria University Wave
    Tracking: Morton et al. 2013, Weberg et al. 2018), simplified for teaching.
      1) at every time, find the local intensity MINIMA of the cut (dark features)
         deeper than `depth` below the local median, refined to sub-pixel with a parabola;
      2) link them in time: a thread continues if the next minimum is within `max_jump`
         pixels, with at most `max_gap` missing frames;
      3) keep threads with at least `min_len` points, and for each one take the FFT of
         the detrended position: the strongest peak gives period and amplitude.
    td is (y, t). Returns a list of dicts: t, y, P, A (pix)."""
    ny, nt = td.shape
    peaks = []
    for k in range(nt):
        p = td[:, k]
        med = np.median(p)
        found = []
        for i in range(1, ny - 1):
            if p[i] < p[i - 1] and p[i] <= p[i + 1] and med - p[i] > depth * med:
                a, b, cc = p[i - 1], p[i], p[i + 1]
                found.append(i + 0.5 * (a - cc) / (a - 2 * b + cc))
        peaks.append(found)
    threads, open_ = [], []                      # each thread: [ks, ys]
    for k in range(nt):
        free = list(peaks[k])
        for th in open_:
            if free and k - th[0][-1] <= max_gap + 1:
                j = int(np.argmin([abs(y - th[1][-1]) for y in free]))
                if abs(free[j] - th[1][-1]) <= max_jump:
                    th[0].append(k); th[1].append(free.pop(j))
        still = [th for th in open_ if k - th[0][-1] <= max_gap]
        threads += [th for th in open_ if th not in still]
        open_ = still + [[[k], [y]] for y in free]
    threads += open_
    out = []
    for ks, ys in threads:
        if len(ks) < min_len:
            continue
        tt = t[ks[0]:ks[-1] + 1]
        yy = np.interp(tt, t[ks], ys)            # fill the gaps linearly
        s = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
        F = np.fft.rfft(s)
        f = np.fft.rfftfreq(len(s), tt[1] - tt[0])
        i = 1 + int(np.argmax(np.abs(F[1:])))
        out.append(dict(t=tt, y=yy, P=1 / f[i], A=2 * np.abs(F[i]) / len(s)))
    return out


# =========================================================================
# 3. REFERENCE SOLUTIONS (what students write in exercises 1-3)
# =========================================================================
def ref_build_td(data, xc):
    return data[:, :, xc].T


def gauss_abs(y, background, depth, centre, sigma):
    return background - depth * np.exp(-((y - centre) ** 2) / (2 * sigma ** 2))


def ref_fit_profiles(td):
    """Gaussian fit at every time: returns centre, sigma, depth (pix)."""
    y = np.arange(td.shape[0])
    out = np.array([fit_curve(gauss_abs, y, td[:, k],
                              [np.median(td[:, k]), 0.3, darkest_pixel(td[:, k]), 3.5])
                    for k in range(td.shape[1])])
    return out[:, 2], np.abs(out[:, 3]), out[:, 1]


def ref_measure_cuts(c, cuts):
    """Centre and width series at every cut, plus kink and sausage periods."""
    r = dict(cuts=np.array(cuts), x_km=c.km(np.array(cuts)), centres=[], widths=[])
    for xc in cuts:
        yc, sig, _ = ref_fit_profiles(ref_build_td(c.data, xc))
        r["centres"].append(yc); r["widths"].append(sig)
    mid = len(cuts) // 2
    r["P_kink"] = fit_sine(c.t, r["centres"][mid])["P"]
    r["P_sausage"] = fit_sine(c.t, r["widths"][mid])["P"]
    return r


# =========================================================================
# 4. PICTURES
# =========================================================================
def grey(c):
    vmin, vmax = np.percentile(c.data, [1, 99.5])
    return dict(origin="lower", cmap="gray", vmin=vmin, vmax=vmax)


def save_gif(fig, update, n, name, fps=FPS):
    os.makedirs(GIFS, exist_ok=True)
    FuncAnimation(fig, update, frames=n).save(os.path.join(GIFS, name), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("   GIF ->", os.path.join(GIFS, name))
    if INLINE:                                    # show it in the notebook
        from IPython.display import Image, display
        display(Image(filename=os.path.join(GIFS, name)))


def save_fig(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name), dpi=120, bbox_inches="tight")
    if INLINE:
        plt.show()
    plt.close(fig)
    print("   figure ->", os.path.join(FIGS, name))


# ---- exercise 0: mode shapes ------------------------------------------
def gif_modes(preset="moderate"):
    """Clean sketches of the kink and sausage modes (uses the simulator)."""
    try:
        import simulation as sim
    except ImportError:
        print("   (simulation.py not here: this GIF is made by the instructor)")
        return
    clean = dict(noise=0.0, jitter_pix=0.0, blur_pix=(0.0, 0.0), texture=0.0, nt=48, dt_s=4.0)
    out = []
    for mode in ["kink", "sausage"]:
        p = sim.params(preset, kink=(mode == "kink"), sausage=(mode == "sausage"), **clean)
        p = sim.replace(p, kink_A_km=3 * p.kink_A_km, saus_width_pct=min(1.4 * p.saus_width_pct, 60))
        cube = sim.simulate(p)[0]
        x_km = np.arange(p.nx) * p.pix_km
        ext = [0, p.nx * p.pix_km / 1e3, 0, p.ny * p.pix_km / 1e3]
        y0 = p.y0_pix * p.pix_km / 1e3
        P = p.kink_P_s if mode == "kink" else p.saus_P_s
        n = int(round(P / p.dt_s)) * 2
        if mode == "kink":
            LD = p.kink_LD(); env = p.kink_A_km * sim.envelope(x_km, LD) / 1e3
            title = "KINK (m = 1): the axis shakes"
        else:
            LD = p.sausage_LD()
            env = 2.355 * p.sigma0_pix * p.pix_km / 2e3 * (1 + p.saus_width_pct / 100 * sim.envelope(x_km, LD))
            title = "SAUSAGE (m = 0): the tube breathes"
        fig = plt.figure(figsize=(11, 3.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.2])
        a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        im = a1.imshow(cube[0], extent=ext, origin="lower", cmap="gray", vmin=0.6, vmax=1.02, aspect="auto")
        a1.plot(x_km / 1e3, y0 + env, "c--", lw=1.2); a1.plot(x_km / 1e3, y0 - env, "c--", lw=1.2)
        a1.set_title(f"{title}   ·   {'no damping' if LD is None else f'L_D = {LD / 1e3:.1f} Mm'}"
                     "   ·   exaggerated", fontsize=10)
        a1.set_xlabel("x along the tube (Mm)"); a1.set_ylabel("y (Mm)"); a1.set_ylim(y0 - 0.9, y0 + 0.9)
        vx = a1.axvline(0, color="y", lw=1)
        a0.set_xlim(-1, 1); a0.set_ylim(-1, 1); a0.set_aspect("equal"); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title("Cross-section (x = 0)", fontsize=10)
        a0.axhline(0, color="0.8", lw=0.8); a0.axvline(0, color="0.8", lw=0.8)
        circ = Circle((0, 0), 0.35, fc="0.25", ec="k"); a0.add_patch(circ)
        plt.tight_layout()
        vf = p.kink_vphase_kms if mode == "kink" else p.saus_vphase_kms

        def f(i, cube=cube, P=P, mode=mode, p=p, vf=vf, im=im, circ=circ, vx=vx):
            im.set_data(cube[i % cube.shape[0]])
            ti = i * p.dt_s; s = np.sin(2 * np.pi * ti / P)
            if mode == "kink":
                circ.center = (0, 0.45 * s)
            else:
                circ.set_radius(0.35 * (1 + 1.3 * p.saus_width_pct / 100 * s))
                circ.set_facecolor(str(0.25 + 0.25 * s))                 # wider -> lighter
            vx.set_xdata([(vf * ti) % (p.nx * p.pix_km) / 1e3] * 2)
            return im, circ, vx

        save_gif(fig, f, n, f"0_{mode}_mode.gif")
        out.append((p, cube, mode))

    fig, ax = plt.subplots(2, 2, figsize=(11, 5), gridspec_kw=dict(width_ratios=[1, 3.2]))
    for row, (p, cube, mode) in enumerate(out):
        a0, a1 = ax[row]
        y0 = p.y0_pix * p.pix_km / 1e3
        a1.imshow(cube[6], extent=[0, p.nx * p.pix_km / 1e3, 0, p.ny * p.pix_km / 1e3],
                  origin="lower", cmap="gray", vmin=0.6, vmax=1.02, aspect="auto")
        a1.set_ylim(y0 - 0.9, y0 + 0.9); a1.set_ylabel("y (Mm)")
        a0.set_xlim(-1, 1); a0.set_ylim(-1, 1); a0.set_aspect("equal"); a0.axis("off")
        if mode == "kink":
            for dy, al in [(-0.45, 0.3), (0, 0.6), (0.45, 1)]:
                a0.add_patch(Circle((0, dy), 0.35, fc="0.25", ec="k", alpha=al))
            a0.set_title("kink: shifts", fontsize=9)
            a1.set_title("KINK (m = 1): moving axis, fixed width", fontsize=10)
        else:
            for r_, g in [(0.5, "0.55"), (0.35, "0.35"), (0.22, "0.15")]:
                a0.add_patch(Circle((0, 0), r_, fc=g, ec="k"))
            a0.set_title("sausage: radius changes", fontsize=9)
            a1.set_title("SAUSAGE (m = 0): fixed axis, width and darkness in antiphase", fontsize=10)
    ax[1, 1].set_xlabel("x along the tube (Mm)")
    plt.tight_layout(); save_fig(fig, "0_mode_shapes.png")


# ---- exercise 1: the time-distance diagram -----------------------------
def plot_td(c, td, xc):
    kw, ext = grey(c), c.extent_img()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax[0].imshow(c.data[0], extent=ext, **kw); ax[0].axvline(c.km(xc) / 1e3, color="r", ls="--")
    ax[0].set_xlabel("x (Mm)"); ax[0].set_ylabel("y (Mm)"); ax[0].set_title("Frame 0  (red = your cut)")
    ax[1].imshow(td, aspect="auto", extent=[0, c.t[-1], 0, ext[3]], **kw)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("y (Mm)"); ax[1].set_title("Your time-distance diagram")
    plt.tight_layout(); save_fig(fig, "1_data.png")


def gif_movie(c):
    kw, ext = grey(c), c.extent_img()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    im = ax.imshow(c.data[0], extent=ext, **kw)
    ax.set_xlabel("x (Mm)"); ax.set_ylabel("y (Mm)"); title = ax.set_title("")
    plt.tight_layout()

    def f(i):
        im.set_data(c.data[i]); title.set_text(f"Synthetic fibril   t = {c.t[i]:3.0f} s")
        return im, title
    save_gif(fig, f, c.nt, "1_fibril.gif")


def gif_td_build(c, td, xc):
    kw, ext = grey(c), c.extent_img()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    im = ax[0].imshow(c.data[0], extent=ext, **kw); ax[0].axvline(c.km(xc) / 1e3, color="r", ls="--")
    ax[0].set_xlabel("x (Mm)"); ax[0].set_ylabel("y (Mm)"); title = ax[0].set_title("")
    empty = np.full_like(td, np.nan)
    im_td = ax[1].imshow(empty, aspect="auto", extent=[0, c.t[-1], 0, ext[3]], **kw)
    cur = ax[1].axvline(0, color="r")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("y (Mm)"); ax[1].set_title("Time-distance diagram")
    plt.tight_layout()

    def f(i):
        im.set_data(c.data[i]); empty[:, : i + 1] = td[:, : i + 1]; im_td.set_data(empty)
        cur.set_xdata([c.t[i]] * 2)
        title.set_text(f"t = {c.t[i]:3.0f} s   (cut at x = {c.km(xc) / 1e3:.1f} Mm)")
        return im, im_td, cur, title
    save_gif(fig, f, c.nt, "1_fibril_td.gif")


# ---- exercise 2: kink ----------------------------------------------------
def plot_kink(c, td, centre, fit, v):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].imshow(td, aspect="auto", extent=[0, c.t[-1], 0, c.ny], **grey(c))
    ax[0].plot(c.t, centre, "r-", lw=1, label="Gaussian centre y₀(t)")
    ax[0].legend(loc="upper right", fontsize=8); ax[0].set_ylabel("y (pix)")
    ax[1].plot(c.t, c.km(centre - fit["trend"]), "k.", ms=3, label="measured")
    ax[1].plot(c.t, c.km(fit["curve"] - fit["trend"]), "r",
               label=f"P = {fit['P']:.0f} s, ξ = {c.km(fit['A']):.0f} km")
    ax[1].set_ylabel("displacement (km)"); ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"Kink:  v = 2πξ/P = {v:.1f} km/s")
    plt.tight_layout(); save_fig(fig, "2_kink.png")


def gif_kink_fit(c, td, centre, sigma):
    y, kw = c.y, grey(c)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
    pt, = ax[0].plot(y, td[:, 0], "k.", ms=4, label="data")
    pf, = ax[0].plot(y, td[:, 0], "r-", label="Gaussian fit")
    pc = ax[0].axvline(0, color="r", ls=":")
    ax[0].set_ylim(td.min() - 0.02, td.max() + 0.02)
    ax[0].set_xlabel("y (pix)"); ax[0].set_ylabel("intensity"); ax[0].legend(loc="lower left", fontsize=8)
    title = ax[0].set_title("")
    ax[1].imshow(td, aspect="auto", extent=[0, c.t[-1], 0, c.ny], **kw)
    track, = ax[1].plot([], [], "r-", lw=1.5); dot, = ax[1].plot([], [], "ro")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("y (pix)"); ax[1].set_title("The centre y₀(t) is the kink")
    plt.tight_layout()

    def f(i):
        bg = np.median(td[:, i])
        pt.set_ydata(td[:, i])
        pf.set_ydata(gauss_abs(y, bg, bg - td[:, i].min(), centre[i], sigma[i]))
        pc.set_xdata([centre[i]] * 2)
        track.set_data(c.t[: i + 1], centre[: i + 1]); dot.set_data([c.t[i]], [centre[i]])
        title.set_text(f"Profile  t = {c.t[i]:3.0f} s   y₀ = {centre[i]:.1f} pix")
        return pt, pf, pc, track, dot, title
    save_gif(fig, f, c.nt, "2_kink_fit.gif")


# ---- exercise 3: sausage -------------------------------------------------
def plot_sausage(c, fwhm_km, depth, fit, r):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(c.t, fwhm_km, "b"); ax.set_ylabel("FWHM (km)", color="b"); ax.set_xlabel("time (s)")
    a2 = ax.twinx(); a2.plot(c.t, depth, color="orange"); a2.set_ylabel("depth (darkness)", color="orange")
    ax.set_title(f"Sausage: P = {fit['P']:.0f} s,  width ±{100 * fit['A'] / fit['mean']:.0f} %,  "
                 f"correlation = {r:+.2f}")
    plt.tight_layout(); save_fig(fig, "3_sausage.png")


def gif_sausage_fit(c, td, centre, sigma, depth):
    y = c.y
    fwhm = c.km(2.355 * sigma)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
    pt, = ax[0].plot(y, td[:, 0], "k.", ms=4); pf, = ax[0].plot(y, td[:, 0], "b-")
    bar, = ax[0].plot([], [], "b-", lw=4, alpha=0.6)
    ax[0].set_ylim(td.min() - 0.02, td.max() + 0.02)
    ax[0].set_xlabel("y (pix)"); ax[0].set_ylabel("intensity"); title = ax[0].set_title("")
    ax[1].set_xlim(0, c.t[-1]); ax[1].set_ylim(fwhm.min() * 0.95, fwhm.max() * 1.05)
    l1, = ax[1].plot([], [], "b-"); ax[1].set_ylabel("FWHM (km)", color="b"); ax[1].set_xlabel("time (s)")
    a2 = ax[1].twinx(); a2.set_ylim(depth.min() * 0.95, depth.max() * 1.05)
    l2, = a2.plot([], [], color="orange"); a2.set_ylabel("depth (darkness)", color="orange")
    ax[1].set_title("Width and darkness: in antiphase")
    plt.tight_layout()

    def f(i):
        bg = np.median(td[:, i]); h = 1.177 * sigma[i]
        pt.set_ydata(td[:, i]); pf.set_ydata(gauss_abs(y, bg, depth[i], centre[i], sigma[i]))
        bar.set_data([centre[i] - h, centre[i] + h], [bg - depth[i] / 2] * 2)
        l1.set_data(c.t[: i + 1], fwhm[: i + 1]); l2.set_data(c.t[: i + 1], depth[: i + 1])
        title.set_text(f"t = {c.t[i]:3.0f} s   FWHM = {fwhm[i]:.0f} km")
        return pt, pf, bar, l1, l2, title
    save_gif(fig, f, c.nt, "3_sausage_fit.gif")


# ---- exercise 4: propagation ---------------------------------------------
def plot_phase_vs_x(x_km, phase_k, phase_s, v_k, v_s):
    fig, ax = plt.subplots(figsize=(7, 4))
    for f, col, mk, name, v in [(phase_k, "r", "o", "kink", v_k), (phase_s, "b", "s", "sausage", v_s)]:
        ax.plot(x_km / 1e3, f, col + mk, label=f"{name}: v_phase = {v:.0f} km/s")
        ax.plot(x_km / 1e3, np.polyval(np.polyfit(x_km, f, 1), x_km), col + "-")
    ax.set_xlabel("cut position (Mm)"); ax.set_ylabel("phase (rad)")
    ax.set_title("The phase drops along a straight line: the wave travels")
    ax.legend(); plt.tight_layout(); save_fig(fig, "4_phase_vs_x.png")


def plot_and_gif_propagation(c, r, v_k):
    cuts, x, kw = r["cuts"], r["x_km"], grey(c)
    P_k = r["P_kink"]
    t0 = c.t[np.argmax(r["centres"][0][: int(P_k / c.dt) + 1])]
    sel = sorted(set([0, len(cuts) // 3, 2 * len(cuts) // 3, len(cuts) - 1]))
    fig, ax = plt.subplots(1, len(sel), figsize=(3.5 * len(sel), 3.4), sharey=True)
    for a, j in zip(np.atleast_1d(ax), sel):
        a.imshow(ref_build_td(c.data, cuts[j]), aspect="auto", extent=[0, c.t[-1], 0, c.ny], **kw)
        a.plot(c.t, r["centres"][j], "r-", lw=0.8)
        a.axvline(t0, color="c", ls="--"); a.axvline(t0 + (x[j] - x[0]) / v_k, color="y")
        a.set_xlim(0, 250); a.set_ylim(c.ny / 2 - 15, c.ny / 2 + 15)
        a.set_title(f"x = {x[j] / 1e3:.1f} Mm"); a.set_xlabel("t (s)")
    np.atleast_1d(ax)[0].set_ylabel("y (pix)")
    fig.suptitle("Cyan: crest at the first cut · Yellow: the same crest, arriving later", fontsize=10)
    plt.tight_layout(); save_fig(fig, "4_cuts.png")

    C3 = [cuts[0], cuts[len(cuts) // 2], cuts[-1]]
    COL = ["#00c8ff", "#ffd000", "#ff4fd8"]
    NT3 = min(c.nt, int(3.2 * P_k / c.dt))
    ext = c.extent_img()
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    a0 = fig.add_subplot(gs[0, :])
    im = a0.imshow(c.data[0], extent=ext, **kw)
    for xc, col in zip(C3, COL):
        a0.axvline(c.km(xc) / 1e3, color=col, lw=2)
    crest = a0.axvline(0, color="lime", lw=2, ls="--")
    a0.set_xlabel("x (Mm)"); a0.set_ylabel("y (Mm)"); title = a0.set_title("")
    panels = []
    for j, (xc, col) in enumerate(zip(C3, COL)):
        a = fig.add_subplot(gs[1, j])
        tdj = ref_build_td(c.data, xc)[:, :NT3]; emp = np.full_like(tdj, np.nan)
        ij = a.imshow(emp, aspect="auto", extent=[0, c.t[NT3 - 1], 0, ext[3]], **kw)
        cu = a.axvline(0, color=col)
        a.set_ylim(ext[3] / 2 - 0.65, ext[3] / 2 + 0.65); a.set_title(f"x = {c.km(xc) / 1e3:.1f} Mm")
        for s in a.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3)
        a.set_xlabel("time (s)")
        if j == 0:
            a.set_ylabel("y (Mm)")
        panels.append((tdj, emp, ij, cu))
    fig.suptitle(f"The crest reaches the far cuts later: v_phase ≈ {v_k:.0f} km/s", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    L = c.nx * c.pix_km

    def f(i):
        im.set_data(c.data[i])
        for tdj, emp, ij, cu in panels:
            emp[:, : i + 1] = tdj[:, : i + 1]; ij.set_data(emp); cu.set_xdata([c.t[i]] * 2)
        crest.set_xdata([((c.t[i] - t0) * v_k) % L / 1e3] * 2)
        title.set_text(f"t = {c.t[i]:3.0f} s   (green = a crest moving at v_phase)")
        return [im, title, crest] + [p[2] for p in panels] + [p[3] for p in panels]
    save_gif(fig, f, NT3, "4_propagation.gif")


# ---- exercise 5: damping -------------------------------------------------
def plot_amplitude_vs_x(x_km, A_k, A0, L_D, lam, width_pct):
    xx = np.linspace(0, x_km.max(), 200)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x_km / 1e3, A_k, "ro", label="measured")
    ax[0].plot(xx / 1e3, A0 * np.exp(-xx / L_D), "r-", label=f"A0 exp(-x/L_D), L_D = {L_D / 1e3:.1f} Mm")
    ax[0].set_ylim(0, None); ax[0].set_xlabel("x (Mm)"); ax[0].set_ylabel("kink amplitude (km)")
    ax[0].set_title(f"Kink: L_D/λ = {L_D / lam:.2f}"); ax[0].legend()
    ax[1].plot(x_km / 1e3, width_pct, "bs")
    ax[1].set_ylim(0, None); ax[1].set_xlabel("x (Mm)"); ax[1].set_ylabel("width change (%)")
    ax[1].set_title("Sausage")
    plt.tight_layout(); save_fig(fig, "5_amplitude_vs_x.png")


def gif_damping(c, A0, L_D, y0_pix):
    kw, ext = grey(c), c.extent_img()
    xx = np.linspace(0, ext[1] * 1e3, 200)
    env = A0 * np.exp(-xx / L_D) / 1e3
    y0 = c.km(y0_pix) / 1e3
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(c.data[0], extent=ext, **kw)
    ax.plot(xx / 1e3, y0 + env, "c--"); ax.plot(xx / 1e3, y0 - env, "c--")
    ax.set_xlabel("x (Mm)"); ax.set_ylabel("y (Mm)"); title = ax.set_title("")
    plt.tight_layout()

    def f(i):
        im.set_data(c.data[i])
        title.set_text(f"t = {c.t[i]:3.0f} s · measured envelope: L_D = {L_D / 1e3:.1f} Mm")
        return im, title
    save_gif(fig, f, c.nt, "5_damping.gif")


def gif_compare_damping():
    """Same kink wave with three damping levels (uses the simulator)."""
    try:
        import simulation as sim
    except ImportError:
        print("   (simulation.py not here: this GIF is made by the instructor)")
        return
    clean = dict(noise=0.005, jitter_pix=0.0, texture=0.0, nt=48, sausage=False, kink_A_km=260.0)
    cubes = [(sim.simulate(sim.params(pre, **clean))[0], sim.params(pre, **clean), name)
             for pre, name in [("no_damping", "no damping"), ("moderate", "moderate"), ("strong", "strong")]]
    fig, ax = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    ims = []
    for a, (cb, p, name) in zip(ax, cubes):
        ims.append(a.imshow(cb[0], extent=[0, p.nx * p.pix_km / 1e3, 0, p.ny * p.pix_km / 1e3],
                            origin="lower", cmap="gray", vmin=0.6, vmax=1.02, aspect="auto"))
        LD = p.kink_LD()
        a.set_title(f"{name}:  " + ("L_D = ∞" if LD is None else
                    f"L_D = {LD / 1e3:.1f} Mm = {LD / p.kink_wavelength_km():.2f} λ"), fontsize=10)
        a.set_ylim(0.3, 2.3); a.set_ylabel("y (Mm)")
        xk = np.arange(p.nx) * p.pix_km
        e = p.kink_A_km * sim.envelope(xk, LD) / 1e3; yy = p.y0_pix * p.pix_km / 1e3
        a.plot(xk / 1e3, yy + e, "c--", lw=1); a.plot(xk / 1e3, yy - e, "c--", lw=1)
    ax[-1].set_xlabel("x (Mm)")
    fig.suptitle("The same kink wave with different damping (exaggerated amplitude)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def f(i):
        for im_, (cb, _, _) in zip(ims, cubes):
            im_.set_data(cb[i])
        return ims
    save_gif(fig, f, 48, "5_compare_damping.gif")


# ---- exercise 6: energy --------------------------------------------------
NEEDED = [("quiet-Sun chromosphere", 4e3), ("active-region chromosphere", 2e4),
          ("quiet-Sun corona", 3e2)]          # Withbroe & Noyes (1977)


def plot_energy(x_km, F_before, F_after, rho_range, title):
    """F_before, F_after: functions F(rho_i, x_km) in W/m^2."""
    xx = np.linspace(0, x_km.max(), 200)
    fig, ax = plt.subplots(figsize=(10, 5))
    for fun, col, name in [(F_before, "tab:red", "BEFORE: ½ ρᵢ v² c_k"),
                           (F_after, "tab:blue", "AFTER: ½ f (ρᵢ+ρₑ) v² c_k")]:
        ax.fill_between(xx / 1e3, fun(rho_range[0], xx), fun(rho_range[1], xx), color=col, alpha=0.25,
                        label=f"{name}  (ρᵢ = 10⁻¹⁰–10⁻⁹ kg/m³)")
    for (name, val), ls in zip(NEEDED, ["-", "--", ":"]):
        ax.axhline(val, color="k", ls=ls, lw=1); ax.text(x_km.max() / 1e3, val * 1.08, name, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1, 1e5)
    ax.set_xlabel("x along the fibril (Mm)"); ax.set_ylabel("energy flux (W/m²)")
    ax.set_title(title, fontsize=10); ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); save_fig(fig, "6_energy.png")


def gif_energy(c, A0_km, L_D_km, P, c_k, rho, correction):
    kw, ext = grey(c), c.extent_img()
    xk = np.arange(c.nx) * c.pix_km
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2]))
    im = ax[0].imshow(c.data[0], extent=ext, aspect="auto", **kw); ax[0].set_ylabel("y (Mm)")
    title = ax[0].set_title("", fontsize=10)
    lb, = ax[1].plot([], [], color="tab:red", label="BEFORE (naive): ½ ρᵢ v²")
    la, = ax[1].plot([], [], color="tab:blue", label="AFTER (corrected): ½ f (ρᵢ+ρₑ) v²")
    v0 = 2 * np.pi * A0_km * 1e3 / P
    e_max = 0.5 * rho * v0 ** 2
    ax[1].plot(xk / 1e3, e_max * np.exp(-2 * xk / L_D_km), "k--", lw=0.8, label="envelope  exp(−2x/L_D)")
    ax[1].set_ylim(0, e_max * 1.1); ax[1].set_xlim(0, ext[1])
    ax[1].set_xlabel("x (Mm)"); ax[1].set_ylabel("energy density (J/m³)"); ax[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    def f(i):
        im.set_data(c.data[i])
        vy = v0 * np.exp(-xk / L_D_km) * np.cos(2 * np.pi * (c.t[i] / P - xk / (c_k * P)))
        e = 0.5 * rho * vy ** 2
        lb.set_data(xk / 1e3, e); la.set_data(xk / 1e3, e * correction)
        title.set_text(f"t = {c.t[i]:3.0f} s · energy travels at c_k = {c_k:.0f} km/s and fades")
        return im, lb, la, title
    save_gif(fig, f, c.nt, "6_energy.gif")


# =========================================================================
# 5. PRINTING
# =========================================================================
def report(title, rows):
    """rows: list of (label, value, unit[, comment])."""
    print(f"\n===== {title} =====")
    for r in rows:
        label, val, unit = r[:3]
        txt = f"{val:10.3g}" if isinstance(val, (int, float, np.floating)) else f"{str(val):>10s}"
        print(f"  {label:34s}{txt} {unit:6s}{'  ' + r[3] if len(r) > 3 else ''}")
