from matplotlib import pyplot as plt
from .utils import *
from .galaxy import Galaxy
import photutils as phot
from astropy.visualization import AsymmetricPercentileInterval, AsinhStretch
from reproject import reproject_adaptive, reproject_interp

def _rotate_wcs(wcs, ra, dec, pxscale):
    """ Rotates the WCS so that it is north-aligned around.
    INPUTS:
        wcs:      World Coordinate System of the data
        ra, dec:  Coordinates around which to rotate (best if image center)
        pxscale:  pixel scale of the image
    RETURNS:
        rotated wcs
    """

    # Convert the central pixel into ra, dec using WCS
    crcoords = [ra, dec]
    crpix    = wcs.wcs_world2pix(ra, dec, 0)

    # Construct the new WCS that is north-aligned
    new_wcs  = wcs.deepcopy()
    new_wcs.wcs.crpix = crpix
    new_wcs.wcs.crval = crcoords
    new_wcs.wcs.pc    = (pxscale/3600) * np.array([[-1,0], [0,1]])
    return new_wcs



def _resize(data, wcs, ra, dec, pxscale, rpet, scale=2):
    """ Resizes the image so that based on Petrosian radius of the galaxy.
    INPUTS:
        data:    input NumPy data array
        wcs:     World Coordinate System of the data
        xc, yc:  pixel coordinates of the galaxy center
        pxscale: pixel scale of the image
        rpet:    galaxy Petrosian radius size in arcseconds
        scale:   desired image size in Petrosian radii, default: 2
    RETURNS:
        resized image and WCS
    """
    xc, yc  = wcs.wcs_world2pix(ra, dec, 0)
    xc, yc  = int(xc), int(yc)
    size    = scale * rpet / pxscale
    size    = int(size+0.5)

    # Check if size is not too large
    size_img = np.min([size, xc, yc, data.shape[1]-xc, data.shape[0]-xc])
    if size_img != size: warnings.warn("Hitting the edge, reducing image Expect error on RGB plot; adjust the size to be smaller!")

    # Get the cutout of the data and WCS
    slices = (slice(yc-size, yc+size), slice(xc-size, xc+size))
    wcs_small = wcs.slice(slices)
    return data[slices], wcs_small

# def _align(data, wcs, ra, dec, pxscale, rpet, scale=2, boolean=False):
#     """ Create a North-aligned cutout of the galaxy based on its radius
#     INPUTS:
#         data:    input NumPy data array
#         wcs:     Wolrd Coordinate System of the data
#         xc, yc:  pixel coordinates of the galaxy center
#         pxscale: pixel scale of the image
#         rpet:    galaxy Petrosian radius in arcseconds
#         scale:   desired image size in Petrosian radii, default: 2
#         boolean: is the data boolean (e.g. mask)? Default: False
#     RETURNS:
#         aligned image, new wcs
#     """
#     rot_img, rot_wcs = _rotate(data, wcs, ra, dec, pxscale, boolean)
#     res_img, res_wcs = _resize(rot_img, rot_wcs, ra, dec, pxscale, rpet, scale)
#     return res_img, res_wcs




class GalaxyPlotter:
    """ Helper class to make various plots of the Galaxy object."""

    # Important properties needed for plotting
    # _important_props = {"pxscale", "zp"}

    def __init__(self, galaxy, dataset):
        """Starts the Plotter instance for plotting the galaxy data.
        INPUTS:
          galaxy:  a Galaxy class object containing image data
          survey:  survey identified for the data (e.g. "hst", "manga", "sdss", ...)
        """


        # Initialize the data from the Galaxy class for a given survey.
        node      = galaxy.data[f"{dataset}"]
        self.data = node
        self.ra   = galaxy.properties["ra"]
        self.dec  = galaxy.properties["dec"]
        self.size = galaxy.properties["size"]
        # self.xc   = galaxy.data["hst/i/galfit"].attrs["xc"]
        # self.yc   = galaxy.data["hst/i/galfit"].attrs["yc"]
        # self.wcs  = attr_to_wcs(self.data.attrs)
        self.dataset = dataset
        # self.filt   = filt

        # Check if the required metadata is given in the header
        # assert "img" in self.data, "Dataset img must be in the HDF5 node"
        # assert_properties(self._important_props, self.data)



    def _plot_normalized(self, data, ax, wcs, upper_scale=99.8, lower_scale=1, asinh_factor=0.7):
        """ Makes a normalized plot of a galaxy with an Asinh stretch.
        INPUTS:
            data:        NumPy data array to plot
            upper_scale: The upper limit to use in normalization if not in HDF5 already
            lower_scale: Lower limit to use in normalization
            ax:          axis to plot on. If None, create a new axis
        """
        # If the upper scale is defined, use that; otherwise, 99.7%
        try:    upper_scale = self.data.attrs["uscale"]
        except: pass
        stretch = AsinhStretch(asinh_factor)
        norm    = AsymmetricPercentileInterval(lower_scale, upper_scale)

        ax.imshow(stretch(norm(data)), cmap="gray", transform=ax.get_transform(wcs))
        ax.axis('off')


    def plot_grayscale(self, filt, ax=None,
                    north=True, radius=None, ra=None, dec=None, rscale=2,
                    upper_scale=99.7, lower_scale=1, asinh_factor=0.7):
        """ Given a filter, makes a normalized, asinh-stretched plot of a galaxy.
        INPUTS:
            filt:        filter to use in the plot
            ax:          axis to plot on. If None, create a new plot
            north:       align the data North? Default: True
            radius:      radius of the galaxy. If None, use "size" from the Galaxy object
            xc, yc:      pixel coordinates of image center. If None, use ra, dec of Galaxy
            rscale:      image scale in terms of galaxy radii
            upper_scale, lower_scale: scales used in normalization. By default, upper_scale
                         is given in metadata or 99.7; lower_scale is 1
        """

        # Load the image
        node = self.data[filt]
        img  = node["img"][()]
        wcs  = attr_to_wcs(node.attrs)

        # Find the center if xc, yc are not given
        if ra is None or dec is None: ra, dec = self.ra, self.dec
        # Find radius if not given
        if radius is None: radius = self.size

        pxscale  = node.attrs["pxscale"]
        img, wcs = _resize(img, wcs, ra, dec, pxscale, radius, scale=rscale)
        plot_wcs = _rotate_wcs(wcs, ra, dec, pxscale) if north else wcs

        # If no axis is given create a figure
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax  = plt.subplot(projection=plot_wcs)
        self._plot_normalized(img, ax, wcs, upper_scale, lower_scale, asinh_factor)




    def plot_rgb(self, filters, ax=None, ra=None, dec=None, radius=None,
                    north=True, rscale=2,
                    upper_scale=99.7, lower_scale=1, asinh_factor=0.7):
        """ Given a filter, makes a normalized, asinh-stretched plot of a galaxy.
        INPUTS:
            filters:     list of filter names [red, green, blue] e.g. ["i", "r", "g"]
            ax:          axis to plot on. If None, create a new plot
            north:       align the data North? Default: True
            rpet:        radius of the galaxy. If None, use "size" from the Galaxy object
            xc, yc:      pixel coordinates of image center. If None, use ra, dec of Galaxy
            rpet_scale:  image scale in terms of galaxy radii
            upper_scale, lower_scale: scales used in normalization. By default, upper_scale
                         is given in metadata or 99.7; lower_scale is 1
        """

        # Load the images for each filter
        data = {"r" : {}, "g" : {}, "b" : {}}
        for filt, key in zip(filters, data):
            node = self.data[filt]
            data[key]["img"] = node["img"][()]
            data[key]["wcs"] = attr_to_wcs(node.attrs)
            data[key]["pxscale"] = node.attrs["pxscale"]

        # If ra, dec and rpet are not passed, populate them using galaxy properties
        if ra is None or dec is None or radius is None:
            ra, dec, radius = self.ra, self.dec, self.size

        # Resize all images
        for k, v in data.items():
            v["img"], v["wcs"] = _resize(v["img"], v["wcs"], ra, dec,
                                         v["pxscale"], radius, scale=rscale)


        # Create a WCS to reproject the images to using blue WCS (north-aligned?)
        rgb_wcs = data["b"]["wcs"].deepcopy()
        if north:
            rgb_pxscale   = data["b"]["pxscale"]
            rgb_wcs       = _rotate_wcs(rgb_wcs, ra, dec, rgb_pxscale)

        # Define stretch and norm
        try:    upper_scale = self.data.attrs["uscale"]
        except: pass
        stretch = AsinhStretch(asinh_factor)
        norm    = AsymmetricPercentileInterval(lower_scale, upper_scale)
        norm_r  = AsymmetricPercentileInterval(lower_scale, upper_scale-0.1)

        # Reproject all data onto this new WCS, apply norm and stretch
        for k, v in data.items():
            img = reproject_interp((v["img"], v["wcs"]), rgb_wcs, shape_out=data["b"]["img"].shape)[0]#data["b"]["img"].shape)[0]
            img = stretch(norm_r(img)) if k == "r" else stretch(norm(img))
            v["plot_img"] = img

        # Make a data cube
        rgb = np.dstack([val["plot_img"] for key, val in data.items()])

        # If ax=None, create figure
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax  = plt.axes(); ax.axis('off')

        ax.imshow(rgb, origin="lower")




    def rff_plot(self, rpet_scale=2, rhalf_scale=1 ):
        assert_data(["galfit/fit", "galfit/residual"], self.data)
        assert_properties(["rff", "rff_in", "rff_out"], self.data["galfit"])

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        self._plot_normalized(self.data["img"], axs[0])
        self._plot_normalized(self.data["galfit/fit"], axs[1])
        self._plot_normalized(self.data["galfit/residual"], axs[2])

        # Define values needed for calculation
        pxscale = self.data.attrs["pxscale"]
        rpet    = self.data["statmorph"].attrs["rpetro_circ"] / pxscale
        rhalf   = self.data["galfit"].attrs["rad"] / pxscale
        xc      = self.data["galfit"].attrs["xc"]
        yc      = self.data["galfit"].attrs["yc"]
        e       = self.data["galfit"].attrs["e"]
        pa      = self.data["galfit"].attrs["theta"]
        th      = (pa-90)*np.pi/180
        rff     = self.data["galfit"].attrs["rff"]
        rff_in  = self.data["galfit"].attrs["rff_in"]
        rff_out = self.data["galfit"].attrs["rff_out"]
        n       = self.data["galfit"].attrs["sersic_n"]

        ap_pet1 = phot.CircularAperture( (xc, yc), rpet_scale*rpet)
        # ap_pet2 = phot.CircularAperture( (xc, yc), rpet)
        ap_ser  = phot.EllipticalAperture((xc, yc), rhalf_scale*rhalf,
                    rhalf_scale*rhalf*e, theta=th)

        for i in [1, 2]:
            ap_pet1.plot(axes=axs[i], color="y", ls="-")
            # ap_pet2.plot(axes=axs[i], color="orange", ls="--")
            ap_ser.plot( axes=axs[i], color="red", ls="-")


        axs[2].annotate(f"RFF = {rff:2.3f}",
                        xy=(0.05, 0.03),
                        xycoords="axes fraction",
                        ha="left", va="bottom", c="w", size=14, fontweight=800)
        axs[2].annotate(f"Inner RFF = {rff_in:2.3f}",
                        xy=(0.05, 0.97),
                        xycoords="axes fraction",
                        ha="left", va="top", c="w", size=14, fontweight=800)
        axs[2].annotate(f"Outer RFF = {rff_out:2.3f}",
                        xy=(0.05, 0.90),
                        xycoords="axes fraction",
                        ha="left", va="top", c="w", size=14, fontweight=800)
        axs[1].annotate(f"n = {n:2.1f}",
                        xy=(0.05, 0.03),
                        xycoords="axes fraction",
                        ha="left", va="bottom", c="w", size=14, fontweight=800)

        plt.subplots_adjust(wspace=0.02)
        return fig, axs
