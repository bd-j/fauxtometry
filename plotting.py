import glob
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from astropy.coordinates import SkyCoord


class catWCS:
    """Make a simple wcs from a catalog or from a center an pixel size
    """
    def __init__(self, cat, center=None, pixel_size=1.0):
        self.tangential(cat["ra"], cat["dec"], center=center)
        self.pixel_size = pixel_size

    def tangential(self, ra, dec, center=None):
        if center is None:
            mra = np.median(ra)
            mdec = np.median(dec)
            center = SkyCoord(mra, mdec, unit="deg")
        self.scene_frame = center.skyoffset_frame()
        self.center = center

    def sky_to_scene(self, ra, dec):
        c = SkyCoord(ra, dec, unit="deg")
        xy = c.transform_to(self.scene_frame)
        return xy.lon.arcsec, xy.lat.arcsec

    def world_to_pixel_values(self, ra, dec):
        x, y = self.sky_to_scene(ra, dec)
        return x / self.pixel_size, y / self.pixel_size


def bandmerge(hdus, bands=["F150W", 'F200W', "F444W"], n_ap=4):
    """Combine row-matched aperture photometry catalogs.
    """

    # --- format output ---
    coln = ["ra", "dec", "x", "y", "a", "b"]
    fluxn = [f"{b}_aper{i}" for b in bands for i in range(n_ap)]
    errn = [f"{c}_e" for c in fluxn]
    cols = [("id", int)] + [(c, float) for c in coln + fluxn + errn]
    dtype = np.dtype(cols)

    # create output and fill with common parameters
    cat = hdus[f"DET_{bands[0].upper()}"].data
    out = np.zeros(len(cat), dtype=dtype)
    out["id"] = np.arange(len(cat))
    for c in coln:
        out[c] = cat[c]

    # fill the aperture fluxes
    for b in bands:
        catn = f"DET_{b.upper()}"
        cat = hdus[catn].data
        for i in range(n_ap):
            out[f"{b}_aper{i}"] = cat[f"aper{i}"]
            out[f"{b}_aper{i}_e"] = cat[f"aper{i}_err"]

    return out


def collect(catnames):
    mcats, rcats = [], []
    for catn in catnames:
        with fits.open(catn) as hdul:
            mcat = hdul["INPUT"].data
            rcat = bandmerge(hdul)
            mcats.append(mcat)
            rcats.append(rcat)
    mcat = np.concatenate(mcats)
    rcat = np.concatenate(rcats)
    return mcat, rcat


if __name__ == "__main__":

    cats = glob.glob("catalogs/bluej/*completeness.fits")
    results = "results"
    mcat, rcat = collect(cats)
    wcs = catWCS(mcat)
    cp = np.array(wcs.sky_to_scene(mcat["ra"], mcat["dec"])).T

    ark_ra, ark_dec = 150.0432351757300, +02.1560036339700
    ca = wcs.sky_to_scene(ark_ra, ark_dec)
    radius = np.hypot(*(cp - ca).T)

    # --- mags input and output ---
    refband, iaper = "F200W", 1
    inmags = -2.5*np.log10(mcat[refband]/3631e9)
    outmags = -2.5 * np.log10(rcat[f"{refband}_aper{iaper}"]/3631e9)

    # --- color cut ---
    # color including aperture corrections.
    incolor = -2.5*np.log10(mcat["F200W"]/mcat["F444W"])
    color = -2.5 * np.log10((rcat[f"F200W_aper1"]*1.55) / ( rcat[f"F444W_aper1"]*3.3))
    cmd_color = -2.5 * np.log10((rcat[f"F150W_aper1"]*1.53) / ( rcat[f"F200W_aper1"]*1.55))

    # --- selection ---
    weight = (mcat["detected"]
              #& (rcat[f"{refband}_aper2"] / rcat[f"{refband}_aper2_e"] > 5)
              & np.isfinite(color)
              & ((-2.5 * np.log10(rcat[f"F200W_aper1"]*1.55/3631e9)) < 30)
              & (cmd_color < -0.00) & (cmd_color > -0.5)
              & (color < -1)
            )

    # --- define mag and radius bins ---
    mbins = np.arange(27.0, 31.1, 0.1)
    rbins = np.array([2.9, 3.7, 4.8, 6.1, 7.8, 10])
    rbins = 10**np.arange(np.log10(0.4), 1.05, 0.1)

    # --- plot vs radius in bins of mag ---
    pl.close("all")
    pl.ion()
    pl.rcParams["font.size"] = 16
    pl.rcParams["lines.linewidth"] = 2.5

    xx, xbins, xname = radius/60., rbins, "radius (')"
    ss, sbins, sname = inmags, np.arange(28, 31, 0.5), refband

    N = len(sbins) - 1
    ctable = np.zeros([N, len(rbins)-1])
    etable = np.zeros([N, len(rbins)-1])

    pl.rcParams["axes.prop_cycle"] = pl.cycler("color", pl.cm.viridis(np.linspace(0,1,N)))
    fig, ax = pl.subplots()
    for i, sbin in enumerate(sbins[:-1]):
        sel = (ss > sbins[i]) & (ss < sbins[i+1])
        print(sel.sum())
        x = xx[sel]
        w = weight[sel]
        hin, _ = np.histogram(x, bins=xbins)
        hout, _ = np.histogram(x,  weights=w, bins=xbins)
        comp = (hout/hin)
        err = np.sqrt(comp / hin * (1 - comp))
        ctable[i] = comp
        etable[i] = err
        xm = (xbins[:-1] + xbins[1:]) / 2
        print(sbin, comp[-9:].min(), comp[-9:].max())
        ax.errorbar(xm, comp/comp[4], yerr=err/comp[4], color="grey", linestyle="", marker="")
        ax.plot(xm, comp/comp[4], "-o", label=f"{sbins[i]:.1f} < {sname} < {sbins[i+1]:.1f}")

    ax.set_xlabel(xname)
    ax.set_ylabel("detected fraction")
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    fig.savefig(f"{results}/bluej_completness_vs_radius.png")

    sbm = (sbins[:-1] + sbins[1:]) / 2.
    with open(f"{results}/bluej_completness_vs_radius.dat", "w") as fout:
        magbins = " ".join([f"{refband}={mm:.2f}" for mm in sbm])
        fout.write(f"outer radius(')  {magbins} \n")
        for i, rbin in enumerate(rbins[1:]):
            cc = " ".join([f"{c:.3f} {e:.4f}" for c, e in zip(ctable[:, i], etable[:,i])])
            fout.write(f"{rbin:.2f} {cc}\n")

    if False:
        h2in, _, _ = np.histogram2d(radius, inmags, bins=[rbins, mbins])
        h2out, _, _ = np.histogram2d(radius, inmags, weights=weight,
                                 bins=[rbins, mbins])

    # --- F200W ----
    min_dist_ark = 2.0

    #total completeness
    xx, xbins, xname = inmags, mbins, refband
    sel = radius/60. > min_dist_ark
    x = xx[sel]
    w = weight[sel]
    hin, _ = np.histogram(x, bins=xbins)
    hout, _ = np.histogram(x,  weights=w, bins=xbins)
    comp = (hout/hin)
    err = np.sqrt(comp / hin * (1 - comp))  # asymptotic estimate of sqrt(var)
    xm = (xbins[:-1] + xbins[1:]) / 2
    cfig, ax = pl.subplots()
    ax.errorbar(xm, comp, yerr=err, color="grey", linestyle="", marker="")
    ax.plot(xm, comp, "-o")
    ax.set_xlabel(xname)
    ax.set_ylabel("selected fraction (detection + color)")
    ax.set_title(f"d_ark > {min_dist_ark} arcmin")
    cfig.savefig(f"{results}/bluej_completness_vs_F200W.png")

    # magnitude offsets
    delta = (outmags - inmags)
    sel = (np.isfinite(outmags)) & (mcat["detected"] > 0) & (radius > (2 * 60))
    dfig, ax = pl.subplots(figsize=(8, 5.8))
    cb = ax.hexbin(inmags[sel], delta[sel],
                   extent=(mbins.min(), mbins.max(), 0.11, 0.9), gridsize=[40, 20])
    ax.set_xlabel(f"{sname} TOTAL_input (mag)")
    ax.set_ylabel(f"APER{iaper} - TOTAL_input (mag)")

    mx = -1
    pct = np.zeros([len(mbins), 3])
    for i, m in enumerate(mbins[:mx]):
        ss = sel & (inmags > m) & (inmags <= mbins[i+1])
        if ss.sum() > 3:
            pct[i] = np.percentile(delta[ss], [50, 16, 85])

    ax.plot(mbins[:mx], pct[:mx, 0], "r--")
    ax.plot(mbins[:mx], pct[:mx, 1], "r:")
    ax.plot(mbins[:mx], pct[:mx, 2], "r:")

    ax.set_ylim(0.11, 0.9)

    dfig.savefig(f"{results}/delta_{refband}.png")

    # text table
    with open(f"{results}/bluej_completness_vs_{refband}.dat", "w") as fout:
        fout.write(f"# dist_ark > {min_dist_ark} arcmin\n")
        fout.write(f"#total_{refband}  completeness completeness_unc dm50 dm16 dm84\n")
        for i, m in enumerate(xm):
            fout.write(f"{m:.1f} {comp[i]:.3f} {err[i]:.4f} {pct[i, 0]:.3f} {pct[i, 1]:.3f} {pct[i, 2]:.3f}\n")


    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.BinTableHDU(mcat, name="INPUT"),
                         fits.BinTableHDU(rcat, name="OUTPUT")])
    hdul.writeto(f"{results}/completeness_all.fits", overwrite=True)

#    ax.imshow(h2out/h2in, extents)