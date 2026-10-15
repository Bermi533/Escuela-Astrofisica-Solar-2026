"""
simulation.py  --  THE SIMULATOR .
Builds a synthetic data cube (t, y, x) of a dark H-alpha fibril carrying a
kink wave and a sausage wave that travel along it, in the style of
Morton et al. (2012)

Everything is controlled with PHYSICAL PARAMETERS (see the Params class)
or with a ready-made PRESET:

    python simulation.py                          # preset "moderate", seed 42
    python simulation.py --preset strong
    python simulation.py --preset big_sausage --seed 7
    python simulation.py --preset kink_only --folder data_kink
    python simulation.py --set kink_LD_km=2500 --set saus_width_pct=50
    python simulation.py --list                   # show presets and parameters

Output (in --folder, default "data/"):
    fibril_cube.fits                -> what you hand out

The shapes of the two modes
---------------------------
  KINK (m = 1):    the tube AXIS moves from side to side; the cross-section
                   keeps its size.             y_c(x,t) = y0 + A(x) sin(wt - kx)
  SAUSAGE (m = 0): the axis stays put and the tube gets WIDER and NARROWER;
                   when it widens the gas thins out and it looks less dark
                   (antiphase).                sigma(x,t) = sigma0 [1 + eps(x) sin(wt - kx)]

Controllable damping (dissipation)
----------------------------------
A travelling wave loses amplitude along its path:
    A(x) = A0 exp(-x / L_D)          (L_D = damping length)
Small L_D = strong damping.  L_D = None = no damping.
For the kink, L_D can be set by hand (kink_LD_km) or computed from the
physics of resonant absorption (kink_LD_km = "physical"): Terradas,
Goossens & Verth (2010, A&A 524, A23, eq. 13), sinusoidal layer, m = 1:
    L_D / lambda = (2/pi) (R/l) (rho_i + rho_e)/(rho_i - rho_e)
with  lambda = v_phase * P,  l/R = relative thickness of the tube boundary
layer (0 = sharp edge, 2 = the whole tube is boundary) and zeta = rho_i/rho_e.
Thicker boundary or higher density contrast -> stronger damping.
Limit: with l/R = 2 and very large zeta, L_D -> lambda/pi (the strongest possible).
"""
import argparse
import json
import os
from dataclasses import dataclass, asdict, fields, replace

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

KM_PER_ARCSEC = 725.0


@dataclass
class Params:
    # --- instrument ----------------------------------------------------
    pix_arcsec: float = 0.06       # 0.06" -> 43.5 km per pixel
    dt_s: float = 4.0              # cadence (s)
    nt: int = 150                  # 150 x 4 s = 10 min
    ny: int = 60                   # 2.6 Mm across the fibril
    nx: int = 160                  # 7.0 Mm along it (longer = damping is easier to see)
    # --- fibril (flux tube) --------------------------------------------
    y0_pix: float = 30.0           # position of the axis
    sigma0_pix: float = 3.5        # Gaussian half-width -> FWHM ~ 360 km (Morton: 360 +- 120)
    depth: float = 0.35            # how dark it is (0 = invisible, 1 = black)
    zeta: float = 3.0              # density contrast rho_i / rho_e
    l_over_R: float = 1.5          # relative boundary thickness (0..2), for the physical L_D
    # --- kink wave -----------------------------------------------------
    kink: bool = True
    kink_P_s: float = 96.0         # period
    kink_A_km: float = 130.0       # amplitude at x = 0
    kink_vphase_kms: float = 60.0  # phase speed (~ kink speed c_k)
    kink_LD_km: object = "physical"  # number (km), "physical" or None
    # --- sausage wave --------------------------------------------------
    sausage: bool = True
    saus_P_s: float = 64.0
    saus_width_pct: float = 35.0       # how much it widens/narrows (exaggerated; Morton ~10 %)
    saus_intensity_pct: float = 25.0   # how much the depth changes (antiphase)
    saus_vphase_kms: float = 100.0
    saus_LD_km: object = None          # number (km) or None
    # --- things that spoil the data ------------------------------------
    noise: float = 0.02
    jitter_pix: float = 0.25       # seeing: image shimmer
    blur_pix: tuple = (0.3, 0.7)   # seeing: variable blurring
    texture: float = 0.025         # chromospheric background
    seed: int = 42

    # ------------------------------------------------------------------
    @property
    def pix_km(self):
        return self.pix_arcsec * KM_PER_ARCSEC

    def kink_wavelength_km(self):
        return self.kink_vphase_kms * self.kink_P_s

    def kink_LD(self):
        """Kink damping length in km (None = no damping)."""
        if self.kink_LD_km == "physical":
            return LD_resonant_absorption(self.kink_wavelength_km(), self.l_over_R, self.zeta)
        return None if self.kink_LD_km is None else float(self.kink_LD_km)

    def sausage_LD(self):
        return None if self.saus_LD_km is None else float(self.saus_LD_km)


def LD_resonant_absorption(wavelength_km, l_over_R, zeta):
    """Terradas, Goossens & Verth (2010), eq. 13 with m = 1 (sinusoidal profile)."""
    if l_over_R <= 0:
        return None                                # sharp edge: no resonant absorption
    return wavelength_km * (2 / np.pi) * (1 / l_over_R) * (zeta + 1) / (zeta - 1)


def envelope(x_km, LD):
    """A(x)/A0 = exp(-x/L_D)."""
    return np.ones_like(x_km, dtype=float) if LD is None else np.exp(-x_km / LD)


# ---------------------------------------------------------------- presets
PRESETS = {
    "no_damping":        dict(kink_LD_km=None, saus_LD_km=None),
    "weak":              dict(l_over_R=0.5),                     # L_D ~ 2.5 lambda (hardly any)
    "moderate":          dict(),                                 # l/R = 1.5 -> L_D ~ 0.85 lambda
    "strong":            dict(l_over_R=2.0, zeta=10.0),          # L_D ~ 0.39 lambda (near the limit)
    "very_strong":       dict(kink_LD_km=1800.0, saus_LD_km=3500.0),  # by hand (~ lambda/pi), sausage too
    "big_sausage":       dict(saus_width_pct=50.0, saus_intensity_pct=35.0),
    "realistic_sausage": dict(saus_width_pct=10.0, saus_intensity_pct=8.0),
    "kink_only":         dict(sausage=False),
    "sausage_only":      dict(kink=False),
}


def simulate(p: Params):
    """Returns (cube[t, y, x], FITS header, answers)."""
    rng = np.random.default_rng(p.seed)
    t = np.arange(p.nt) * p.dt_s
    y = np.arange(p.ny)[:, None]
    x_km = np.arange(p.nx)[None, :] * p.pix_km

    tex = gaussian_filter(rng.normal(size=(p.nt, p.ny, p.nx)), sigma=(12, 12, 8))
    tex = 1 + p.texture * tex / tex.std()

    LDk, LDs = p.kink_LD(), p.sausage_LD()
    A_k = (p.kink_A_km / p.pix_km) * envelope(x_km, LDk) * p.kink       # pix
    eps = (p.saus_width_pct / 100) * envelope(x_km, LDs) * p.sausage
    dlt = (p.saus_intensity_pct / 100) * envelope(x_km, LDs) * p.sausage

    cube = np.empty((p.nt, p.ny, p.nx), dtype=np.float32)
    for i, ti in enumerate(t):
        sk = np.sin(2 * np.pi * (ti / p.kink_P_s - x_km / (p.kink_vphase_kms * p.kink_P_s)))
        ss = np.sin(2 * np.pi * (ti / p.saus_P_s - x_km / (p.saus_vphase_kms * p.saus_P_s)))
        yc = p.y0_pix + A_k * sk                      # KINK: the axis moves
        sigma = p.sigma0_pix * (1 + eps * ss)         # SAUSAGE: the width changes
        depth = p.depth * (1 - dlt * ss)              # ... and so does the darkness, in antiphase
        jy = rng.normal(0, p.jitter_pix)
        img = tex[i] * (1 - depth * np.exp(-((y - yc - jy) ** 2) / (2 * sigma ** 2)))
        img = gaussian_filter(img, rng.uniform(*p.blur_pix))
        img += rng.normal(0, p.noise, img.shape)
        cube[i] = img

    h = fits.Header()
    h["OBJECT"] = ("Synthetic fibril", "simulated data for teaching")
    h["TELESCOP"] = "SYNTHETIC"
    h["WAVELNTH"] = (6563.0, "[Angstrom] nominal H-alpha")
    for ax, name in [(1, "x"), (2, "y")]:
        h[f"CTYPE{ax}"], h[f"CUNIT{ax}"] = ("HPLN-TAN", "HPLT-TAN")[ax - 1], "arcsec"
        h[f"CDELT{ax}"] = (p.pix_arcsec, f"[arcsec] pixel size in {name}")
        h[f"CRPIX{ax}"], h[f"CRVAL{ax}"] = 1.0, 0.0
    h["CTYPE3"], h["CUNIT3"] = "TIME", "s"
    h["CDELT3"] = (p.dt_s, "[s] cadence")
    h["CRPIX3"], h["CRVAL3"] = 1.0, 0.0
    h.add_comment("Intensity normalised to the background (~1). Dark fibril along x.")
    h.add_comment("In python: fits.getdata(...) has shape (t, y, x).")

    cuts = list(range(10, p.nx - 9, 10))
    amp_k = lambda xc: p.kink * p.kink_A_km * float(envelope(np.array(xc * p.pix_km), LDk))
    amp_s = lambda xc: p.sausage * p.saus_width_pct * float(envelope(np.array(xc * p.pix_km), LDs))
    answers = {
        "params": {k: v for k, v in asdict(p).items()},
        "kink_period_s": p.kink_P_s if p.kink else None,
        "kink_phase_speed_kms": p.kink_vphase_kms if p.kink else None,
        "kink_wavelength_km": p.kink_wavelength_km(),
        "kink_amplitude_km_at_x0": p.kink_A_km if p.kink else 0.0,
        "kink_L_D_km": LDk,
        "kink_L_D_over_wavelength": None if LDk is None else LDk / p.kink_wavelength_km(),
        "kink_amplitude_km_per_cut": {c: round(amp_k(c), 1) for c in cuts},
        "kink_velocity_kms_per_cut": {c: round(2 * np.pi * amp_k(c) / p.kink_P_s, 2) for c in cuts},
        "sausage_period_s": p.saus_P_s if p.sausage else None,
        "sausage_phase_speed_kms": p.saus_vphase_kms if p.sausage else None,
        "sausage_L_D_km": LDs,
        "sausage_width_pct_per_cut": {c: round(amp_s(c), 1) for c in cuts},
        "sausage_intensity_pct_at_x0": p.saus_intensity_pct if p.sausage else 0.0,
        "mean_fwhm_km": 2.355 * p.sigma0_pix * p.pix_km,
        "width_vs_intensity": "antiphase",
        "note": "Seeing broadens the profile, so the measured width change comes out a bit smaller.",
    }
    return cube, h, answers


def save(cube, h, answers, folder="data"):
    os.makedirs(folder, exist_ok=True)
    fits.writeto(os.path.join(folder, "fibril_cube.fits"), cube, h, overwrite=True)
    with open(os.path.join(folder, "ANSWERS_DO_NOT_DISTRIBUTE.json"), "w") as f:
        json.dump(answers, f, indent=2, default=str)
    return folder


def params(preset="moderate", seed=42, **changes):
    """From another script:  p = params("strong", kink_A_km=200)."""
    return replace(Params(seed=seed), **{**PRESETS[preset], **changes})


def _value(s):
    if s in ("None", "none"):
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        return float(s) if "." in s or "e" in s else int(s)
    except ValueError:
        return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="moderate", choices=PRESETS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folder", default="data")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("PRESETS:")
        for k, v in PRESETS.items():
            print(f"  {k:18s} {v}")
        print("\nPARAMETERS (defaults):")
        for f_ in fields(Params):
            print(f"  {f_.name:20s} = {f_.default}")
        raise SystemExit
    changes = dict(kv.split("=", 1) for kv in a.set)
    p = params(a.preset, a.seed, **{k: _value(v) for k, v in changes.items()})
    cube, h, answers = simulate(p)
    save(cube, h, answers, a.folder)
    LD = p.kink_LD()
    print(f"Cube {cube.shape} -> {a.folder}/fibril_cube.fits   (preset {a.preset})")
    print(f"  kink   : P = {p.kink_P_s:.0f} s, A0 = {p.kink_A_km:.0f} km, v_phase = {p.kink_vphase_kms:.0f} km/s,"
          f" L_D = {'no damping' if LD is None else f'{LD:.0f} km = {LD / p.kink_wavelength_km():.2f} lambda'}")
    print(f"  sausage: P = {p.saus_P_s:.0f} s, width +-{p.saus_width_pct:.0f} %, v_phase = "
          f"{p.saus_vphase_kms:.0f} km/s, L_D = {p.sausage_LD()}")
