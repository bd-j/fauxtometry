#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_test_image.py -
Use galsim to generate some images that can be used for training
and testing. Largely adopated from Ben Johnson's make_test_image.py.
"""

import os
from argparse import ArgumentParser

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import galsim


class Config:
    """
    Attributes
    ----------
    colors : dict
        Keyed by band name, this specifies the distribution of colors (in
        magnitudes) relative to a reference band.  The sense is that
        mean = band - ref, set mean=0 for the reference band

    mag_faint : float
        Faint limit of fake sources, AB mag

    mag_bright : float
        Bright limit of fake sources, AB mag

    npix_per_object: int
        side length of the area to simulate for each source, pixels.
    """
    colors = {}
    colors["F200W"] = dict(mean=0, sigma=0.01)
    colors["F150W"] =  dict(mean=-0.25, sigma=0.05)
    colors["F115W"] =  dict(mean=-0.5, sigma=0.1)
    colors["F444W"] =  dict(mean=1.5, sigma=0.1)

    mag_faint = 31.0
    mag_bright = 27.0

    npix_per_object = 64

    n_fake = 1000

    @property
    def bands(self):
        self._bands = list(self.colors.keys())
        self._bands.sort()
        return self._bands


# ----------------
# --- SIMULATE ---

def multiband_injection(imnames, tag, outdir=None,
                        n_fake=1000, bright=28, faint=31):
    config = Config()
    config.n_fake = n_fake
    config.mag_bright = bright
    config.mag_faint = faint

    image, hdr, conv = read_im(imnames[0])
    cat = fake_pointsource_catalog(image, config.n_fake, config)
    image.close()

    for imn in imnames:
        outn = imn.replace(".fits", f"{tag}.fits")
        if outdir:
            outn = os.path.join(outdir, os.path.basename(outn))
        image, hdr, conv = read_im(imn)
        print(f"{imn} image pixel scale is {hdr['PIXELSCL']:5f} arcsec/pixel")
        wpsf, nrc = make_webbpsf_image(hdr)
        gpsf = get_galsim_psf(psfobj=wpsf)
        sim = simulate_image(cat, hdr, gpsf, config.npix_per_object)
        assert sim.sum() > 0
        print(f'{outn}: image/catalog={sim.sum() / cat[hdr["FILTER"]].sum():.3f}')
        write_im(outn, sim / conv, wpsf, cat, real=image)
        image.close()


def simulate_image(cat, hdr, psf, n_side, beep=100):
    '''
    Generate image with galfit.
    '''

    sim = np.zeros([hdr["NAXIS2"], hdr["NAXIS1"]], dtype=np.float32)
    pixel_scale = hdr["PIXELSCL"]
    band = hdr["FILTER"]
    gsp = galsim.GSParams(maximum_fft_size=10240)

    for i, row in enumerate(cat):
        if np.mod(i, beep) == 0:
            print(f"on star {i} of {len(cat)}")
        xfull, yfull = row["x"], row["y"]
        flux = row[band]
        gimage = galsim.ImageF(n_side, n_side, scale=pixel_scale)
        x0, y0 = int(xfull - 0.5 * n_side), int(yfull - 0.5 * n_side)
        src = galsim.DeltaFunction(flux=flux)
        src_conv = galsim.Convolve([psf, src], gsparams=gsp)
        #egal = gal.shear(q=row["q"], beta=row["pa"] * galsim.radians)
        # place the source and draw it
        x = (xfull - x0)
        y = (yfull - y0)
        dx, dy = x - (n_side - 1) / 2., y - (n_side - 1) / 2.
        if np.hypot(dx, dy) > 1e-2:
            offset = dx, dy
        else:
            offset = 0.0, 0.0
        src_conv.drawImage(gimage, offset=offset, add_to_image=True)

        sim[y0:y0+n_side, x0:x0+n_side] = gimage.array[:]

    return sim


# ---------------
# --- CATALOG ---

def draw_xy(image, n_fake):

    err = image["ERR"].data
    yv, xv = np.where(np.isfinite(err) & (err > 0))
    ind = np.random.choice(len(xv), size=n_fake)
    xi, yi = xv[ind], yv[ind]
    # subpixel offset
    x = xi + np.random.uniform(0, 1, size=n_fake)
    y = yi + np.random.uniform(0, 1, size=n_fake)

    return x, y


def fake_pointsource_catalog(image, n_fake, config):
    '''
    Draw galaxy parameter
    '''

    cols = np.unique(["id", "ra", "dec", "x", "y"] + config.bands)
    cat_dtype = np.dtype([(c, np.float64) for c in cols])
    cat = np.zeros(n_fake, dtype=cat_dtype)
    cat["id"] = np.arange(n_fake) + 1

    cat["x"], cat["y"] = draw_xy(image, n_fake)
    hdr = image[0].header.copy()
    hdr.update(image['SCI'].header)
    wcs = WCS(hdr)
    cat["ra"], cat["dec"] = wcs.pixel_to_world_values(cat["x"], cat["y"])

    # log-uniform reference band fluxes
    rmags = np.random.uniform(config.mag_bright, config.mag_faint, size=n_fake)

    for b in config.bands:
        mag = rmags + np.random.normal(loc=config.colors[b]["mean"],
                                       scale=config.colors[b]["sigma"],
                                       size=n_fake)
        cat[b] = 3631e9*10**(-0.4*mag)

    return cat


# -----------------
# --- PSF MODEL ---

def get_info(hdr):
    inst = hdr['INSTRUME']
    det = hdr['DETECTOR']
    filtern = hdr['FILTER']
    date = hdr['DATE-BEG']
    if det == "MULTIPLE":
        if hdr["CHANNEL"] == "SHORT":
            det = "NRCA1"
        else:
            det = "NRCA5"

    return inst, det, filtern, date


def make_webbpsf_image(hdr, oversamp=4,
                       fov=10.015, x=1024, y=1024):
    '''
    Models PSF for NIRCam (oversampled by 4)
        input: filter [str]
        output: psf_obj [FITS HDUList], psf_data [np.array]
    '''
    inst, det, filtern, date = get_info(hdr)
    nwave = dict(M=9, W=21)

    import webbpsf
    det = det.replace("LONG", "5") #omg
    nrc = webbpsf.NIRCam()
    nrc.detector = det

    nrc.filter = filtern
    nrc.load_wss_opd_by_date(date, plot=False, choice="before")

    # --- mess with webbpsf here ---
    #nrc.options['parity'] = 'odd'
    #nrc.options["jitter_sigma"] = 0.007
    #cds = webbpsf.constants.INSTRUMENT_DETECTOR_CHARGE_DIFFUSION_DEFAULT_PARAMETERS
    #nrc.options['charge_diffusion_sigma'] = cds[f"{inst}_{channel[0]}W"]
    nrc.options['add_ipc'] = False
    nrc.detector_position = (int(x), int(y)) # use x, y, order, not pythonic y,x

    psf_obj = nrc.calc_psf(detector_oversample=oversamp,
                           fft_oversample=4,
                           fov_arcsec=fov,
                           nlambda=nwave[filtern[-1]])
    return psf_obj["OVERDIST"], nrc


def get_galsim_psf(psfobj=None, sigma_psf=1.0,
                   pixel_scale=0.03, psfmixture=None):
    """
    Parameters
    ----------
    pixel_scale : float
        arcsec per science detector pxel
    psfimage : string
        name of fits file containing PSF image, if any
    sigma_psf : float
        dispersion of gaussian PSF in pixels
    psfmixture :
        Not implemented
    """
    if psfobj:
        hdu = psfobj
        pixel_scale = hdu.header.get("PIXELSCL", pixel_scale)
        print(f'using PSF pixel scale {pixel_scale:5f} arcsec/pixel')
        psfim = hdu.data.astype(np.float64)
        pim = galsim.Image(np.ascontiguousarray(psfim), scale=pixel_scale)
        gpsf = galsim.InterpolatedImage(pim)
    elif psfmixture:
        raise NotImplementedError
    else:
        gpsf = galsim.Gaussian(flux=1., sigma=sigma_psf * pixel_scale)

    return gpsf


# ---------------
# --- UTILS -----

def read_im(filename):
    image = fits.open(filename)
    hdr = image[0].header.copy()
    hdr.update(image['SCI'].header)
    wcs = WCS(hdr)
    hdr["PIXELSCL"] = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix*3600)))
    conv = 1e15 / 4.25e10 * hdr["PIXELSCL"]**2 # conversion from MJy/sr to nJy/pix
    image[0].header["TO_NJYPX"] = conv
    hdr["TO_NJYPX"] = conv
    return image, hdr, conv


def write_im(filename, mock, psf, cat, real=None):
    '''
    Write image to fits file.
    '''
    hdr, sci_hdr = None, None
    if real is not None:
        hdr = real[0].header
        sci_hdr = real["SCI"].header

    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr),
                         fits.ImageHDU(mock, header=sci_hdr, name="MOCKIM"),
                         psf,
                         fits.BinTableHDU(cat, name="MOCKCAT")])
    if real is not None:
        sci = fits.ImageHDU((real["SCI"].data + mock).astype(np.float32),
                            header=real["SCI"].header, name="SCI")
        err = fits.ImageHDU((real["ERR"].data).astype(np.float32),
                            header=real["ERR"].header, name="ERR")
        hdul.append(sci)
        hdul.append(err)

    hdul.writeto(filename, overwrite=True)
    hdul.close()


def test_shape(xfull=101.5, yfull=1024.0, flux=1.0, shape_full=[2350, 1350]):
    # xfull=101.5; yfull=1024.0; flux=1.0; shape_full=[2350, 1350] (NAXIS2, NAXIS1)
    """
    Simple test with Gaussian PSF
    1. convention is that integer pixel values referes to the center of the pixel
    2. convention is that edge of pixel is half-integer
    3. convention is that lower-left pixel is x=0, y=0 and, e.g. runs from -0.5 to +0.5
    4. convention is that output dimensions are (y, x) (i.e. NAXIS2, NAXIS1)
    """

    n_side = 16
    pixel_scale = 1.0
    gsp = galsim.GSParams(maximum_fft_size=10240)
    sim = np.zeros(shape_full, dtype=np.float32)
    psf = get_galsim_psf(sigma_psf=1.0, pixel_scale=pixel_scale)

    image = galsim.ImageF(n_side, n_side, scale=pixel_scale)
    x0, y0 = int(xfull - 0.5 * n_side), int(yfull - 0.5 * n_side)
    src = galsim.DeltaFunction(flux=flux)
    src_conv = galsim.Convolve([psf, src], gsparams=gsp)
    #egal = gal.shear(q=row["q"], beta=row["pa"] * galsim.radians)
    # place the source and draw it
    x = (xfull - x0)
    y = (yfull - y0)
    dx, dy = x - (n_side - 1) / 2., y - (n_side - 1) / 2.
    if np.hypot(dx, dy) > 1e-2:
        offset = dx, dy
    else:
        offset = 0.0, 0.0
    src_conv.drawImage(image, offset=offset, add_to_image=True)

    sim[y0:y0+n_side, x0:x0+n_side] = image.array[:]
    return sim


def test(plot=False):
    xfull = 101.5
    yfull = 1024.0
    sim = test_shape(xfull=xfull, yfull=yfull)
    assert np.isclose(sim.sum(), 1.0)
    # half-integer is on the pixel edge
    assert sim[int(yfull), int(np.floor(xfull))] == sim[int(yfull), int(np.floor(xfull))+1]
    # integer is center, pixels on either side are fainter
    assert sim[int(yfull)-1, int(xfull)] < sim[int(yfull), int(xfull)]
    assert sim[int(yfull)+1, int(xfull)] < sim[int(yfull), int(xfull)]

    if plot:
        import matplotlib.pyplot as pl
        fig, ax = pl.subplots()
        pad = 10
        yr = int(yfull)-pad, int(yfull)+pad
        xr = int(xfull)-pad, int(xfull)+pad
        ax.imshow(sim[yr[0]:yr[1], xr[0]:xr[1]],
                  origin="lower") #extents=omg stop
        ax.set_xlabel("X")
        ax.set_ylabel("Y")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--image_names", type=str, nargs="+",
                        default=["../image/jw018100/v0.7_mosaic/mosaic_F115W.fits"])
    parser.add_argument("--mock_dir", type=str, default="./")
    parser.add_argument("--tag", type=str, default="ast-00")
    parser.add_argument("--n_fake", type=int, default=1000)
    parser.add_argument("--display", type=int, default=0)
    args = parser.parse_args()

    test(plot=args.display)
    multiband_injection(args.image_names, args.tag, n_fake=args.n_fake,
                        outdir=args.mock_dir)

