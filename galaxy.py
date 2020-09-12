"""
Python package for storing and analysing
multi-wavelength and cross-dataset astronomical data.

This module impletemts base Galaxy class and
connects to an HDF5 objects storing the Galaxy data
in a hierarchical structure.

Structure:
-- global metadata (id, size, ra, dec)
-- dataset (hst, sdss, ...)
    -- filter (F814W, g, ...)
        -- metadata (zp, exptime, pxscale)
        -- img (NumPy image array)
        -- raw: raw data from the telescope,
                processed to match the Galaxy data format
                mask, unc, psf
        -- products: data products from initial processing
                mask, segmap
        -- measurements: final analysis products
                galfit, statmorph catalogs
"""

import h5py as hdf
import warnings
from astropy.io  import fits
from astropy.wcs import WCS
from .utils import *


class Galaxy:

    _important_props = {"size", "ra", "dec"}
    properties = {}


    def __init__(self, name, filepath, **kwargs):
        """ Initialize the Galaxy object, identified by name, and its HDF5
        file located at filepath. If the HDF5 file doesn't exist, create one.
        If filepath is None, filepath="name.h5"
        """

        filepath = f"{filepath}/{name}.h5"
        self.name = name
        self.data = hdf.File(filepath, "a")

        # Init metadata
        self._init_metadata(kwargs)



    def _init_metadata(self, kwargs):
        """ Assign meta properties (galaxy ra, dec, size) and check
        if all properties are assigned """

        # Assign additional properties to the HDF5
        if kwargs is not None:
            for key, val in kwargs.items():
                self.data.attrs[key] = val

        # Assign all galaxy metadata from HDF5 to the class
        for key, val in self.data.attrs.items():
            self.properties[key] = val

        # Check that all important properties are given
        missing          = self._important_props - set(self.properties.keys())
        assert len(missing) == 0, f"Missing properties: {missing}"


    ############################################################################
    ######################## Adding data to HDF5 File## ########################
    ############################################################################


    def add_array(self, dataset, filt, name, data, overwrite=False):
        path = f"{dataset}/{filt}/{name}"
        self._add_array(path, data, overwrite)

    def _add_array(self, path, data, overwrite=False):
        """ Store an array data in the HDF5 file at path.
        If overwrite=False and path already exists, do nothing."""

        # if path in self.data:
        #     if overwrite: del self.data[path]
        #     else:
        #         warnings.warn(f"{key} already exists in file, skipping. " +
        #             "To overwrite, set overwrite=True")
        #         return
        if not clear_overwrite(path, self.data, overwrite): return
        self.data[path] = data

    def add_data(self, dataset, filt, image, wcs, meta_dict,
                 raw_mask=None, uncertainty=None, psf=None, mask=None, segmap=None,
                 psf_nsamp=None, diff_kernel=None, overwrite=False):
        """
        Saves the data for dataset, filter in the HDF5.
        Input:
            dataset:    Image dataset: e.g. hst, sdss, ...
            filt:       Image filter: e.g. f814w, g, w1, ...
            meta_dict:  Metadata dictionary for this image:
                        E.g. zp, exptime, pxscale...
            image:      NumPy array with image data
            optional:   raw_mask, uncertainty, psf, mask, segmap data arrays
            overwrite:  Skip existing data arrays unless True
        TODO:
            allow input to be a fits file with image and header
            allow input to be a CCDData
            allow wcs input to be WCS, header or dict
        """

        # Paths of different data arrays in the HDF5 file
        path       = f"{dataset}/{filt}"
        data_types = {
            "image"     :   {"data" : image,        "path" : f"{path}/img"},
            "raw_mask"  :   {"data" : raw_mask,     "path" : f"{path}/raw_mask"},
            "uncertainty" : {"data" : uncertainty,  "path" : f"{path}/unc"},
            "psf"       :   {"data" : psf,          "path" : f"{path}/psf"},
            "mask"      :   {"data" : mask,         "path" : f"{path}/mask"},
            "segmap"    :   {"data" : segmap,       "path" : f"{path}/segmap"},
            "kernel"    :   {"data" : diff_kernel,  "path" : f"{path}/diff_kernel"},
            "psf_nsamp" :   {"data" : psf_nsamp,    "path" : f"{path}/psf_nsamp"}
        }

        # Check if this node (dataset+filter) already exists and create if not
        node_exists = path in self.data
        if not node_exists: self.data.create_group(path)


        # Assign meta data & WCS to the dataset/filter directory
        if not node_exists or overwrite:
            # Write meta data
            for key, val in meta_dict.items():
                self.data[path].attrs[key] = val
            # Write wcs
            for key, val in wcs.to_header().items():
                self.data[path].attrs[key] = val

        # Check that the WCS has a PC matrix (sometimes has CD).
        # TO DO: MAKE SURE CD IS NOT UNITARY (MULTIPLY BY PXSCALE)
        try:    wcs.wcs.pc
        except: wcs.wcs.pc = wcs.wcs.cd
        # Calculate pixel scale if not passed in the meta_dict
        if "pxscale" not in meta_dict.items():
            pc       = wcs.wcs.pc
            pxscale  = 3600 * np.sqrt( pc[0,0]**2 + pc[0,1]**2)
            self.data[path].attrs["pxscale"] = pxscale

        # Write data to HDF5
        for key, val in data_types.items():
            if val["data"] is None: continue
            self._add_array(val["path"], val["data"], overwrite=overwrite)


    def add_metadata(self, path, **kwargs):
        """ Add metadata to a node located at path """
        for key, val in kwargs.items():
            self.data[path].attrs[key] = val

    def close(self):
        """ Close the HDF5 file, changes are saved"""
        self.data.close()

    ############################################################################
    ########################## Post-processing data ############################
    ############################################################################
