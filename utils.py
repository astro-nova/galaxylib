import os
from astropy.wcs import WCS
import h5py as hdf
import warnings
import numpy as np

def attr_to_wcs(attrs):
    """ Converts attribute info of HDF5 node to a WCS object
    using WCS keywords in the node"""
    return WCS(dict(attrs))

def coord_to_pix(ra, dec, wcs):
    """ Returns integer pixel coordinates of (ra, dec)"""
    x, y = wcs.all_world2pix(ra, dec, 0)
    x, y = int(x+0.5), int(y+0.5)
    return x, y

def assert_properties(properties, hdf):
    for p in properties:
            assert p in hdf.attrs, f"Missing {p} property. Use galaxy.add_metadata to add to this node"

def assert_data(types, hdf):
    for type in types:
            assert type in hdf, f"Missing {type}. Use galaxy.add_array to add to this node"

def clear_overwrite(path, node, overwrite=False):
    if path in node:
        if overwrite: del node[path]
        else:
            warnings.warn(f"{path} already exists in node, skipping. " +
                "To overwrite, set overwrite=True")
            return False
    return True

def check_float(num):
    try:
        out = float(num)
    except:
        return False
    return True

def get_magnitude(flux, dataset, zp, exptime):
    '''
    DESCRIPTION: calculates the magnitude based on flux and possibly other parameters
    INPUTS:
      flux:     the flux that is to be converted into magnitude
      dataset:  flux/mag conversion depends on the dataset; choose sdss | dupont
      img_info: pandas row or dict containing necessary info (exptime, zp, ...)
    OUTPUTS:
      mag:      magnitude
    '''
    if (dataset == "dupont") or (dataset == "panstarrs"):
        mag = -2.5 * np.log10(flux / exptime) + zp
    elif (dataset == "des") or (dataset == "hst"):
        mag = -2.5 * np.log10(flux) + zp
    elif dataset == "sdss":
        mag = -2.5 * np.log10(flux*1e-9)
    else:
        raise Exception("Magnitude/flux conversion for dataset %s not set" % dataset)
    return mag
