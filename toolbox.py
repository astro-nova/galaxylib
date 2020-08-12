

import photutils as phot
import numpy as np
import os
import warnings
from .utils import *
from .galaxy import Galaxy
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import statmorph
from statmorph.utils.image_diagnostics import make_figure as sm_make_figure

def _segmap_base(data, numpix, mask=None, nsigma=2, contrast=0.4,
                 nlevels=5, kernel=None):
    """Returns a generic segmentation map of the input image
    INPUTS:
      data:     image data
      numpix:   minimum region size in pixels (default: 10)
      mask:     mask for the image (default: None)
      snr:      signal-to-noise threshold for detecting objects (default: 2)
      contrast: contrast ratio used in deblending (default: 0.4)
      nlevels:  number of lebels to split image to when deblending (default: 5)
      kernel:   kernel to use for image smoothing (default: None)
    """

    # Convert mask to boolean
    if mask is not None and (mask.dtype != "bool"):
        mask = np.array(mask, dtype="bool")

    threshold = phot.detect_threshold(data, nsigma=nsigma, mask=mask)
    segmap    = phot.detect_sources(data, threshold, numpix,
                                    filter_kernel=kernel,mask=mask)
    segmap    = phot.deblend_sources(data, segmap, npixels=numpix, filter_kernel=kernel,
                                     nlevels=nlevels, contrast=contrast)
    return segmap


def _morphology_dict( source_morph, dataset, zp, exptime, pxscale,
                            flux_type="flux_ellip", rad_type="rhalf_ellip"):
    """Saves the morphology parameters calculated by statmorph.
    API reference: https://statmorph.readthedocs.io/en/latest/api.html
    """

    mag   = get_magnitude(source_morph[flux_type], dataset, zp, exptime)
    rad   = source_morph[rad_type] * pxscale
    depth = get_magnitude(source_morph["sky_sigma"], dataset, zp, exptime)

    output = {
        "x"                     : source_morph.xc_centroid,
        "y"                     : source_morph.yc_centroid,
        "mag"                   : mag,
        "radius"                : rad,
        "depth"                 : depth,
        "asymmetry"             : source_morph.asymmetry,
        "concentration"         : source_morph.concentration,           # Part of CAS (Conselice04?)
        "deviation"             : source_morph.deviation,               # Part of MID (Freeman13, Peth16 )
        "ellipticity_asymmetry" : source_morph.ellipticity_asymmetry,   # Ellip. rel to min. asym. point
        "ellipticity_centroid"  : source_morph.ellipticity_centroid,    # Ellip. rel to centroid
        "elongation_asymmetry"  : source_morph.elongation_asymmetry,    # Elong. rel to min. asym. point
        "elongation_centroid"   : source_morph.elongation_centroid,     # Elong. rel to centorid
        "flag"                  : source_morph.flag,                    # 1 if failed to estimate
        "flag_sersic"           : source_morph.flag_sersic,             # 1 if sersic fit failed
        "flux_circ"             : source_morph.flux_circ,               # Flux in 2xPetrosian radius
        "flux_ellip"            : source_morph.flux_ellip,              # Flux in 2xPetrosian ellip. radius
        "gini"                  : source_morph.gini,                    # Lotz04
        "gini_m20_bulge"        : source_morph.gini_m20_bulge,          # Rodriguez-Gomez19
        "gini_m20_merger"       : source_morph.gini_m20_merger,         # Rodriguez-Gomez19
        "intensity"             : source_morph.intensity,               # Part of MID (Freeman13, Peth16)
        "m20"                   : source_morph.m20,                     # Lotz04
        "multimode"             : source_morph.multimode,               # Part of MID (Freeman13, Peth16)
        "orientation_asymmetry" : source_morph.orientation_asymmetry,   # Orientation rel. to min asym. point
        "orientation_centroid"  : source_morph.orientation_centroid,    # Orientation rel. to centroid
        "outer_asymmetry"       : source_morph.outer_asymmetry,         # Wen14
        "r20"                   : source_morph.r20          * pxscale,  # 20% light within 1.5Rpetro
        "r50"                   : source_morph.r50          * pxscale,  # 50% light within 1.5Rpetro
        "r80"                   : source_morph.r80          * pxscale,  # 80% light within 1.5Rpetro
        "rhalf_circ"            : source_morph.rhalf_circ   * pxscale,  # 50% light; circ ap; min asym; total at rmax
        "rhalf_ellip"           : source_morph.rhalf_ellip  * pxscale,  # 50% light; ell. ap; min asym; total at rmax
        "rmax_circ"             : source_morph.rmax_circ    * pxscale,  # From min asym to edge, Pawlik16
        "rmax_ellip"            : source_morph.rmax_ellip   * pxscale,  # Semimajor ax. from min asym to edge
        "rpetro_circ"           : source_morph.rpetro_circ  * pxscale,  # Petrosian; wrt min asym point
        "rpetro_ellip"          : source_morph.rpetro_ellip * pxscale,  # Petrosian ellip; wrt min asym point
        "sersic_amplitude"      : source_morph.sersic_amplitude,        # Amplitude of sersic fit at rhalf
        "sersic_ellip"          : source_morph.sersic_ellip,            # Ellipticity of sersic fit
        "sersic_n"              : source_morph.sersic_n,                # Sersic index
        "sersic_rhalf"          : source_morph.sersic_rhalf * pxscale,  # Sersic 1/2light radius
        "sersic_theta"          : source_morph.sersic_theta,            # Orientation of sersic fit
        "shape_asymmetry"       : source_morph.shape_asymmetry,         # Pawlik16
        "sky_mean"              : source_morph.sky_mean,
        "sky_median"            : source_morph.sky_median,
        "sky_sigma"             : source_morph.sky_sigma,
        "smoothness"            : source_morph.smoothness,            # Part of CAS (Conselice04?)
        "sn_per_pixel"          : source_morph.sn_per_pixel,
        "xc_asymmetry"          : source_morph.xc_asymmetry,          # Asym. center (x)
        "yc_asymmetry"          : source_morph.yc_asymmetry           # Asym. center (y)
    }

    return output



def _galfit_make_temp_files(node, dataset, temp_path="."):
    """Creates temporary files given the node in the file format required by galfit.
    Requires: img, psf, mask, dataset name, gain
    """

    ## Image file for GALFIT
    image  = node["img"][()]
    header = fits.Header()
    header["EXPTIME"]   = node.attrs["exptime"]
    header["GAIN"]      = node.attrs["gain"]
    header["NCOMBINE"]  = 1

    # HST stores data in counts / second
    if dataset == "hst": image[()] = image * node.attrs["exptime"]
    imfile = fits.PrimaryHDU(image[()], header=header)
    imfile.writeto(f"{temp_path}/image.fits", overwrite=True)

    ## Other files for GALFIT
    ## todo: uncertainty. For now, GALFIT calculates uncertainty
    filetypes = ["psf", "mask"]
    for type in filetypes:
        data  = node[type]
        dfile = fits.PrimaryHDU(data[()])
        dfile.writeto(f"{temp_path}/{type}.fits", overwrite=True)

def _galfit_get_value(input):
    string = [s.strip('*') for s in input.split()]
    value  = [s for s in string if check_float(s)][0]
    return float(value)


def _galfit_get_flag(header, component=1):
    flag = False
    cols = ["N","XC", "YC", "RE", "AR", "PA"]
    for col in cols:
        string = header[f"{component:d}_{col}"]
        if "*" in string: flag=True
    return flag

def _galfit_param_dict(header, pxscale):
    output  = {
            "sersic_n"  : _galfit_get_value(header["1_N"] ),
            "xc"        : _galfit_get_value(header["1_XC"]),
            "yc"        : _galfit_get_value(header["1_YC"]),
            "mag"       : _galfit_get_value(header["1_MAG"]),
            "rad"       : _galfit_get_value(header["1_RE"])*pxscale,
            "e"         : _galfit_get_value(header["1_AR"]),
            "theta"     : _galfit_get_value(header["1_PA"]),
            "flag"      : _galfit_get_flag(header, 1)
        }
    return output



class GalaxyAnalyzer:

    _important_props = {"pxscale", "zp", "exptime", "psf_fwhm"}

    def __init__(self, galaxy, survey, filt):
        """Starts the Toolbox instance for post-processing of galaxy data.
        INPUTS:
          hdf_node: dataset/filter node containing image, mask etc. data
                    for a galaxy. The node must also contain image metadata:
                    zp, exptime, pxscale; psf/fwhm
          ra, dec, size:     meta data for the galaxy
        """

        node      = galaxy.data[f"{survey}/{filt}"]
        self.data = node
        self.ra   = galaxy.properties["ra"]
        self.dec  = galaxy.properties["dec"]
        self.size = galaxy.properties["size"]
        self.wcs  = attr_to_wcs(self.data.attrs)
        self.survey = survey
        self.filt   = filt

        # Check if the required metadata is given in the header
        assert "img" in self.data, "Dataset img must be in the HDF5 node"
        assert_properties(self._important_props, self.data)

        # Calculate sky value for later if not already calculated
        try:
            assert_properties(["sky_mean", "sky_med", "sky_rms"])
        except:
            mask = None if "raw_mask" not in node else node["raw_mask"]
            mean, med, std = sigma_clipped_stats(node["img"], mask=mask)
            node.attrs["sky_mean"] = mean
            node.attrs["sky_med"]  = med
            node.attrs["sky_rms"]  = std

    ############################################################################
    ################ IMAGE SEGMENTATION (UNDER CONSTRUCTION) ###################
    ############################################################################

    def make_segmap(self, overwrite=False):
        """Returns a segmap with just the galaxy object, and all other
        objects are assigned to a mask.
        INPUTS:
        """

        # Check if segmap already exists in the node
        if not clear_overwrite("segmap", self.data, overwrite): return
        clear_overwrite("mask", self.data, True)

        # Initialize data and metadata from the HDF5
        data    = self.data["img"][()]
        pxscale = self.data.attrs["pxscale"]
        fwhm    = self.data.attrs["psf_fwhm"]
        mask    = self.data["raw_mask"] if "raw_mask" in self.data else None

        # Create the smoothing kernel based on PSF fwhm
        fwhm   = fwhm / pxscale
        kernel = Gaussian2DKernel(fwhm)
        kernel.normalize()

        ### Method: create 2 segmaps: hot and cold
        ### Hot: smoothed, large snr, small size - to identify all objects
        ### 1. Hot segmap: small area (0.1 arcsec^2), large SNR
        area       = int(0.1 / pxscale**2)
        hot_segmap = _segmap_base(  data, area, mask=mask, nsigma=10,
                                    kernel=kernel, contrast=0.9, nlevels=128)

        ### 2. "Filter": increase importance of hot regions
        _, _, sd  = sigma_clipped_stats(data, mask=mask)
        filt_data = data.copy()
        filt_data[hot_segmap.data > 0] += 1e3*sd

        ### 3. Cold segmap: large area (0.5 arcsec^2) , small SNR
        area       = int(0.5/pxscale**2)
        segmap     = _segmap_base(  filt_data, area, mask=mask, nsigma=0.5,
                                    kernel=kernel, contrast=0.01, nlevels=32)

        ### Determine galaxy center in the image using ra, dec
        x, y = coord_to_pix(self.ra, self.dec, self.wcs)

        ### Keep only the central pixel segment, add rest to mask
        segment  = segmap.data[y, x]
        mask_new = np.zeros(data.shape) if mask is None else mask.copy()
        mask_new[(segmap.data != segment) & (segmap.data != 0)] = 1
        segmap.keep_label(segment, relabel=True)

        ### Assign to the node's segmap
        self.data["segmap"] = segmap
        self.data["mask"]   = mask_new

    ############################################################################
    ####################### Morphology (UNDER CONSTRUCTION) ####################
    ############################################################################
    '''
    Calculate morphology 2 ways: statmorph and galfit
    '''

    ##### Statmorph
    def run_statmorph(self, flux_type="flux_ellip", rad_type="rhalf_ellip", overwrite=False):
        '''
        Requires: gain, mask, psf, segmap
        '''
        assert_properties(["gain", "zp", "exptime"], self.data)
        assert_data(["img", "segmap", "mask", "psf"], self.data)
        if not clear_overwrite("statmorph", self.data, overwrite): return

        morphology = statmorph.source_morphology(
            image  = self.data["img"][()]-self.data.attrs["sky_mean"],
            segmap = self.data["segmap"][()],
            mask   = np.array(self.data["mask"][()], dtype=bool),
            psf    = self.data["psf"][()],
            gain   = self.data.attrs["gain"])[0]

        output = _morphology_dict(morphology, self.survey, self.data.attrs["zp"],
                    self.data.attrs["exptime"], self.data.attrs["pxscale"],
                    flux_type = flux_type, rad_type = rad_type)

        #### Save the diagnostic figure as RGB array
        diag_fig = sm_make_figure(morphology)
        canvas   = FigureCanvas(diag_fig)
        canvas.draw()
        diag_fig_arr = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        diag_fig_arr = diag_fig_arr.reshape(diag_fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        #### Store in the node
        self.data["statmorph"] = diag_fig_arr
        for key, value in output.items():
            self.data["statmorph"].attrs[key] = value



    '''
    GALFIT runs with two temporary files: a temporary CCD (normalized to exptime
    of 1 etc), and a parameter file. Create these temp files in a directory
    given by path (or, by default, root of the kernel). Delete when galfit runs.
    Conversion depends on the survey, since data is stored in different units.
    '''
    def run_galfit(self, temp_path='.', constraints_path=None,
                    numiters=100, overwrite=False ):
        """
        Run single-component galfit; requires a previous statmorph run
        to estimate the starting parameters
        """

        assert_properties(["exptime", "gain", "pxscale"], self.data)
        assert_data(["img", "psf", "mask", "statmorph"], self.data)
        if not clear_overwrite("galfit", self.data, overwrite): return

        ## Save data files
        _galfit_make_temp_files(self.data, self.survey, temp_path)

        ## Constraint file: unnless passed, use the default one
        if constraints_path is None:
            constraints_path = "galfit_constraints.txt" ### how to point to root directory of the library???

        ## GALFIT parameters
        ymax, xmax  = self.data["img"].shape  # Image size
        boxsize     = int(0.1 * np.min([xmax, ymax])) # Convolution box
        morphs      = dict(self.data["statmorph"].attrs)
        pxscale     = self.data.attrs["pxscale"]
        sky_mean    = self.data.attrs["sky_mean"]

        ## Parameter file for GALFIT
        params = {
              "A)" : f"{temp_path}/image.fits", # Input image
              "B)" : f"{temp_path}/output.fits", # Output image filename
              "C)" : f"none",#%s/galfit/%s/temp_data/sigma.fits" % (path, filt),# % (path, filt), #"none", # Uncertainty (weight) image -- 1 sigma deviation including counting and sky RMS
              "D)" : f"{temp_path}/psf.fits",#"%s" % psf,
              "F)" : f"{temp_path}/mask.fits", # Mask
              "G)" : f"{constraints_path}", # Parameter coupling: constraints
              "H)" : f"0 {xmax:d} 0 {ymax:d}", # Fitting region -- entire image
              "I)" : f"{boxsize} {boxsize}", # Convolution box size
              "J)" : f"{self.data.attrs['zp']:2.2f}", # ZP to convert fluxes into magnitudes
              "K)" : f"{pxscale} {pxscale}", # Pixel scale
              "O)" : f"regular", # Interactive window - off
              "P)" : f"0"} # Options (normal run)

        # Sersic function
        sersic_params = {
            "0)"  : "sersic", # Type of fit
             "1)"  : f"{morphs['x']:0.0f} {morphs['y']:0.0f} 1 1", # x and y positions of the galaxy
             "3)"  : f"{morphs['mag']:0.2f} 1", # Integrated magnitude of a galaxy
             "4)"  : f"{morphs['sersic_rhalf']/pxscale:0.2f} 1", # Scale length in pixels
             "5)"  : f"{morphs['sersic_n']:0.2f} 1", # Sersic index, n
             "9)"  : f"{morphs['sersic_ellip']:0.2f} 1", # Ellipticity: 1 = circle, < 1 = ellipse
             "10)" : f"{morphs['sersic_theta']*180/np.pi:0.2f} 1", # Position angle
             "Z)" : "0" # Subtract the model in the final image? 1=no, 0=yes to get residual
        }

        # Sersic function
        sky_params = {
             "0)"  : "sky", # Type of fit
             "1)"  : f"{sky_mean} 1 ", # Sky background
             "2)"  : "0 1", # x gradient dF/dx
             "3)"  : "0 1", # y gradient dF/dy
             "Z)" : "0" # Subtract the model in the final image? 1=no, 0=yes to get residual
        }

        # Write the galfit paramfile in the temporary file
        paramfile = open(f"{temp_path}/input.txt", "w")
        for key, value in params.items():
            paramfile.write(f"{key} {value}\n")
        paramfile.write("\n")
        for key, value in sersic_params.items():
            paramfile.write(f"{key} {value}\n")
        paramfile.write("\n")
        for key, value in sky_params.items():
            paramfile.write(f"{key} {value}\n")
        paramfile.close()

        # Run GALFIT
        cmd  = f"galfit -imax {numiters} {temp_path}/input.txt"
        os.system(cmd)

        # Save output.fits to node
        outfile = fits.open(f"{temp_path}/output.fits")
        self.data["galfit/fit"]         = outfile[2].data
        self.data["galfit/residual"]    = outfile[3].data
        fit_params = _galfit_param_dict(outfile[2].header, pxscale)
        for key, value in fit_params.items():
            self.data["galfit"].attrs[key] = value

        # Delete temp files
        files = ["image.fits", "psf.fits", "mask.fits", "input.txt", "output.fits"]
        for file in files:
            cmd = f"rm {temp_path}/{file}"
            os.system(cmd)


    ############################################################################
    ################################# Morph. Analysis ##########################
    ############################################################################

    def get_rff(self):

        assert_properties(["pxscale"], self.data)
        assert_data(["img", "mask", "galfit"], self.data)
