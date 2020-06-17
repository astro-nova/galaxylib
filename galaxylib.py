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



class Galaxy:

    _important_props = {"size", "ra", "dec"}
    properties = {}


    def __init__(self, name, filepath=None, **kwargs):
        """ Initialize the Galaxy object, identified by name, and its HDF5
        file located at filepath. If the HDF5 file doesn't exist, create one.
        If filepath is None, filepath="name.h5"
        """

        filepath  = name + ".h5" if filepath is None else filepath
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




    def add_data(self, dataset, filt, meta_dict, image,
                 raw_mask=None, uncertainty=None, psf=None, mask=None, segmap=None,
                 overwrite=False):
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
        """

        # Paths of different data arrays in the HDF5 file
        path       = f"{dataset}/{filt}"
        data_types = {
            "image"     :   {"data" : image,        "path" : f"{path}/img"},
            "raw_mask"  :   {"data" : raw_mask,     "path" : f"{path}/raw/mask"},
            "uncertainty" : {"data" : uncertainty,  "path" : f"{path}/raw/unc"},
            "psf"       :   {"data" : psf,          "path" : f"{path}/raw/psf"},
            "mask"      :   {"data" : mask,         "path" : f"{path}/products/mask"},
            "segmap"    :   {"data" : segmap,       "path" : f"{path}/products/segmap"},
        }

        # Check if this node already exists
        node_exists = path in self.data

        # Assign meta data to the dataset/filter directory
        if not node_exists:
            self.data.create_group(path)
        if not node_exists or overwrite:
            for key, val in meta_dict.items():
                self.data[path].attrs[key] = val

        # Write data to HDF5
        for key, val in data_types.items():

            if val["data"] is None: continue
            if val["path"] in self.data:
                if overwrite:
                    del self.data[val["path"]]
                else:
                    warnings.warn(f"{key} already exists in file, skipping. " +
                        "To overwrite, set overwrite=True")
                    continue
            self.data[val["path"]] = val["data"]


    def add_metadata(self, path, **kwargs):
        """ Add metadata to a node located at path """
        for key, val in kwargs.items():
            self.data[path].attrs[key] = val



    def close(self):
        """ Close the HDF5 file, changes are saved"""
        self.data.close()

    def hello(self):
        print("Hello World! I'm a galaxy")
