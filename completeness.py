from argparse import ArgumentParser
import numpy as np
from numpy.lib.recfunctions import append_fields
from astropy.io import fits
from astropy.coordinates import SkyCoord
from scipy.spatial import KDTree

# read in the catalog from the simulated image

# read in the detection catalog

# cross match, and filter (compact, similarish flux, etc..)

# add a "detected" field, and construct an extension withthe measured properties


class catWCS:
    """Make a simple wcs from a catalog or from a center an pixel size
    """
    def __init__(self, cat, center=None, pixel_size=1.0):
        self.tangential(cat["RA"], cat["DEC"], center=center)
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


def crossmatch(Xr, Xi, threshold=4):
    """
    Paramters
    ---------
    Xr : shape (nr, 2)
        coordinates in reference catalog

    Xi : shape (ni, 2)
        coordinates in target catalog

    threshold : float
        Max distance to call a match

    Returns
    -------
    xr : shape (nm, 2)
        Reference coordinates of the matched objects, equal to Xr[ref_ind]

    dX : shape (nm, 2)
        Distances for the matched objects

    ref_ind : int array of shape (nm,)
        Indices into Xr for the matches

    i_ind : int array of shape (nm,)
        Indices into Xi for the matches
    """
    kdt = KDTree(Xr)
    dist, inds = kdt.query(Xi)
    success = dist < threshold

    ref_ind = inds[success]
    im_ind = np.arange(len(Xi))[success]

    #in ref image pixels
    dX = Xi[im_ind] - Xr[ref_ind]

    return Xr[ref_ind], dX, ref_ind, im_ind


def completeness_catalog(mock, det, threshold=1.5):
    Xr = np.array([mock["x"], mock["y"]]).T
    Xi = np.array([det["x"], det["y"]]).T

    X, dX, rinds, iinds = crossmatch(Xr, Xi, threshold=threshold)

    # make matched catalogs with same rows as mock
    detected, dist = np.zeros(len(mock), dtype=int), np.zeros([len(mock), 2])
    detected[rinds] = 1
    dist[rinds, :] = dX
    recovered = np.zeros(len(mock), dtype=det.dtype)
    recovered[rinds] = det[iinds]

    # Filter matches based on other criteria?
    # leave that to further pos processing....

    dummy = np.zeros(len(mock))
    complete = append_fields(mock, ["detected", "match_dist"],
                             [detected, dummy], dtypes=[int, (float, (2,))], usemask=False)
    complete["dist"] = dist

    return complete, recovered


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--mock_dir", type=str, default="/n/holystore01/LABS/conroy_lab/Lab/BlueJay/image/jw018100/v0.7_mosaic/")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--band", type=str, default=["F200W", "F150W", "F115W", "F090W", "F277W", "F356W", "F444W"])
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="match distance threshold in pixels")
    args = parser.parse_args()

    mock_name = f"{args.mock_dir}/mosaic_{args.band.upper()}{args.tag}.fits"
    det_name = mock_name.replace(".fits", "_phot.fits")
    comp_name = mock_name.replace(".fits", "_completeness.fits")

    mock = fits.getdata(mock_name, "MOCKCAT")
    det = fits.getdata(det_name)

    cat, props = completeness_catalog(mock, det)

    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.BinTableHDU(cat, name="INPUT"),
                         fits.BinTableHDU(props, name="MEASURED")])
    for hdu in hdul:
        hdu.header["MOCKN"] = mock_name
        hdu.header["DETN"] = det_name
    hdul["MEASURED"].header["FILTER"] = args.band
    hdul.writeto(comp_name, overwrite=True)
