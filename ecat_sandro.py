#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.lib.recfunctions import append_fields
from argparse import ArgumentParser

from astropy.io import fits
from astropy.wcs import WCS

import sep


# This should match the defaults in detection.py
class SEPParameters:
    def __init__(self, thresh=3.0, deblend_cont=0.0001, minarea=3, filter_type='matched',
                 deblend_nthresh=32, clean=True, clean_param=1.0):
        self.thresh          = thresh          #threshold in sigma
        self.deblend_cont    = deblend_cont    #deblending contrast
        self.minarea         = minarea         #minimum pixel area for a source
        self.filter_type     = filter_type     #detection filter type
        self.deblend_nthresh = deblend_nthresh #deblending pixel threshold
        self.clean           = clean           #clean?
        self.clean_param     = clean_param     #clean parameter
        self.int_nan         = np.nan#-999999         #integer representation of nan


class SepImage:

    def __init__(self, imn, xslice=slice(None), yslice=slice(None)):

        data = fits.getdata(imn, "SCI")
        err = fits.getdata(imn, "ERR")
        #err.byteswap(inplace=True).newbyteorder()
        #print(xslice, yslice)

        hdr = fits.getheader(imn, 1)
        data = data[yslice, xslice]
        err = err[yslice, xslice]
        data = data.byteswap().newbyteorder()
        if xslice.start is not None:
            hdr["CRPIX1"] = hdr["CRPIX1"] - xslice.start
        if yslice.start is not None:
            hdr["CRPIX2"] = hdr["CRPIX2"] - yslice.start
        hdr["NAXIS1"] = data.shape[1]
        hdr["NAXIS2"] = data.shape[0]

        self.hdr = hdr
        self.wcs = WCS(self.hdr)
        conv = 1e15 / 4.25e10 * self.pixscale**2
        self.data = np.ascontiguousarray(data * conv)
        self.err = np.ascontiguousarray(err * conv)
        self.mask = ~np.isfinite(self.err) | (self.err <= 0)


    @property
    def pixscale(self):
        pixarea = np.abs(np.linalg.det(self.wcs.pixel_scale_matrix*3600))
        return np.sqrt(pixarea)


def sdetect(im, sp):

    objects, seg = sep.extract(im.bsub, sp.thresh, err=im.err, mask=im.mask,
                               minarea=sp.minarea, filter_type=sp.filter_type,
                               deblend_nthresh=sp.deblend_nthresh, deblend_cont=sp.deblend_cont,
                               clean=sp.clean, clean_param=sp.clean_param, segmentation_map=True)

    #im.segmentation = seg_map
    ra, dec = im.wcs.pixel_to_world_values(objects['x'], objects['y'])
    xwin, ywin, flag = sep.winpos(im.bsub, objects["x"], objects["y"], 1)
    RA, DEC = im.wcs.pixel_to_world_values(xwin, ywin)

    cols = ["id", "ra", "dec", "xwin", "ywin", "RA", "DEC"]
    values = [np.arange(len(objects)), ra, dec, xwin, ywin, RA, DEC]
    detected = append_fields(objects, cols, values)

    return detected, seg


def sphotometer(detections, image, apers, buffer=0):
    """Photometer sources in an image using SEP

    Parameters
    ----------
    detections : structured array (or dictionary of arrays)
        The detections; must include the columns 'ra' and 'dec'

    image : an Image() instance
        The image one which to perform photometry

    aper : float
        Circular aperture radius in pixels

    buffer : int
        Only photometr detections further than `buffer` pixels from the edge

    Returns
    -------
    catalog : structured ndarray
        Structured array containing source parameters, including aperture fluxes
    """
    n = len(detections)

    # --- Pixel locations ---
    # x, y = image.wcs.world_to_pixel_values(detections["ra"], detections["dec"])
    x, y = image.wcs.world_to_pixel_values(detections["RA"], detections["DEC"])
    ok = ((x > buffer) & (x < (image.hdr["NAXIS1"] - buffer)) &
          (y > buffer) & (y < (image.hdr["NAXIS2"] - buffer)))

    cols = ["x", "y", "ra", "dec", "a", "b"]
    phot = np.zeros(n, dtype=np.dtype([(c, float) for c in cols]))
    phot["x"] = x
    phot["y"] = y
    # phot["ra"] = detections["ra"]
    # phot["dec"] = detections["dec"]
    phot["ra"] = detections["RA"]
    phot["dec"] = detections["DEC"]
    phot["a"] = detections["a"]
    phot["b"] = detections["b"]

    # --- Circular Apertures ---
    for i, aper in enumerate(apers):
        print(i, aper)
        flux, err, flag = sep.sum_circle(image.bsub, x[ok], y[ok], aper,
                                        mask=image.mask, err=image.err, subpix=5)
        cols = [f"aper{i}", f"aper{i}_err", f"aper{i}_flag"]
        values = [np.zeros(n), np.zeros(n), np.zeros(n)]
        phot = append_fields(phot, cols, values)
        phot[f"aper{i}"][ok] = flux
        phot[f"aper{i}_err"][ok] = err
        phot[f"aper{i}_flag"][ok] = flag

    return phot


def subtract_background(im):
    im.bkg = sep.Background(im.data)
    im.bsub = im.data - im.bkg


def make_scatalog(im, sp, apers=[2, 4]):

    detected, seg = sdetect(im, sp)
    ra, dec = im.wcs.pixel_to_world_values(detected["xwin"], detected["ywin"])
    detected = append_fields(detected, ["RA", "DEC"], [ra, dec])
    photometry = sphotometer(detected, im, apers)

    return detected, photometry



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--mosaic_dir", type=str, default="/n/holystore01/LABS/conroy_lab/Lab/BlueJay/nircam_data_reduction/v0.7_mosaic/")
    parser.add_argument("--bands", type=str, nargs='+', default=["F200W", "F150W", "F115W", "F090W", "F277W", "F356W", "F444W"])
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    apers = 1, 2, 4, 8, 16
    image_names = [f"{args.mosaic_dir}/mosaic_{band.upper()}{args.tag}.fits"
                   for band in args.bands]


    detected = []
    size_cut = 2000
    im_size = [23500, 13500]
    full_seg = np.zeros(np.array(im_size), dtype=np.float32)

    image_name = image_names[0]
    segname = image_name.replace(".fits", "_seg.fits")
    detname = image_name.replace(".fits", "_det.fits")

    print(f"detecting on {image_names[0]} and {image_names[1]}")
    for i in range(int(np.ceil(im_size[1]/size_cut))):
        for j in range(int(np.ceil(im_size[0]/size_cut))):
            #print(i, j)
            x_slice = slice(i*size_cut, np.min([(i+1)*size_cut, im_size[1]]))
            y_slice = slice(j*size_cut, np.min([(j+1)*size_cut, im_size[0]]))
            im = SepImage(image_names[0], xslice=x_slice, yslice=y_slice)
            subtract_background(im)
            im2 = SepImage(image_names[1], xslice=x_slice, yslice=y_slice)
            subtract_background(im2)
            # im3 = SepImage(image_names[2], xslice=x_slice, yslice=y_slice)
            # subtract_background(im3)
            # combine images
            im.bsub = 1./2. * (im.bsub + im2.bsub)
            im.err = 1./2. * np.sqrt(im.err**2 + im2.err**2)
            im.mask = im.mask | im2.mask

            # set SEP params and do dections
            sp = SEPParameters()
            sp.thresh = 2.0
            sp.minarea = 2.0

            try:
                # do detection
                det, seg = sdetect(im, sp)
                # add to full seg
                idx0 = (seg == 0.0)
                seg += np.max(full_seg).astype(int)
                seg[idx0] = 0
                full_seg[y_slice, x_slice] += seg
                detected.append(det)
            except:
                print(i, j)
                try:
                    print("increase threshold for extraction: 3!")
                    sp.thresh = 3.0
                    sp.minarea = 3.0
                    det, seg = sdetect(im, sp)
                    idx0 = (seg == 0.0)
                    seg += np.max(full_seg).astype(int)
                    seg[idx0] = 0
                    full_seg[y_slice, x_slice] += seg
                    detected.append(det)
                except:
                    try:
                        print("increase threshold for extraction: 4!")
                        sp.thresh = 4.0
                        sp.minarea = 3.0
                        det, seg = sdetect(im, sp)
                        idx0 = (seg == 0.0)
                        seg += np.max(full_seg).astype(int)
                        seg[idx0] = 0
                        full_seg[y_slice, x_slice] += seg
                        detected.append(det)
                    except:
                        try:
                            print("increase threshold for extraction: 5 (3)!")
                            sp.thresh = 5.0
                            sp.minarea = 3.0
                            det, seg = sdetect(im, sp)
                            idx0 = (seg == 0.0)
                            seg += np.max(full_seg).astype(int)
                            seg[idx0] = 0
                            full_seg[y_slice, x_slice] += seg
                            detected.append(det)
                        except:
                            try:
                                print("increase threshold for extraction: 5 (5)!")
                                sp.thresh = 5.0
                                sp.minarea = 5.0
                                det, seg = sdetect(im, sp)
                                idx0 = (seg == 0.0)
                                seg += np.max(full_seg).astype(int)
                                seg[idx0] = 0
                                full_seg[y_slice, x_slice] += seg
                                detected.append(det)
                            except:
                                try:
                                    print("increase threshold for extraction: 20!")
                                    sp.thresh = 20.0
                                    sp.minarea = 5.0
                                    det, seg = sdetect(im, sp)
                                    idx0 = (seg == 0.0)
                                    seg += np.max(full_seg).astype(int)
                                    seg[idx0] = 0
                                    full_seg[y_slice, x_slice] += seg
                                    detected.append(det)
                                except:
                                    print("no extraction!")
                                    pass
    detected = np.concatenate(detected)

    fits.writeto(detname, detected, overwrite=True)
    fits.writeto(segname, full_seg, overwrite=True)

    for image_name in image_names:
        print(f"photometering {image_name}")
        im = SepImage(image_name)
        subtract_background(im)
        catname = image_name.replace(".fits", "_phot.fits")
        photometry = sphotometer(detected, im, apers)
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             fits.BinTableHDU(photometry)])
        for hdu in hdul:
            for i, ap in apers:
                hdu.header[f"AP{i}PIXR"] = ap
        hdul.writeto(catname, overwrite=True)



