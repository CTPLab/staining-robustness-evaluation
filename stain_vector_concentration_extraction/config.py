# -*- coding: utf-8 -*-
"""
    ALGORITHM PARAMETERS
    __author__ = "Maxime Lafarge, maxime.lafarge[at]unibas[dot]ch"
    __creation__ = "2025"
"""

patch_size = 512  # -- Maximum size allowed for the input image patch
patch_subsampling_factor = 4  # -- Subsampling factor for faster processing

background_cutoff_percentile = 50  # -- Upper pixel intensity cutoff value
foreground_cutoff = 20  # -- Lower pixel intensity cutoff value

scatterplot_smoothing_kernel = 8  # -- Scatter plot smoothing: gaussian kernel size
scatterplot_smoothing_stdev = 1  # -- Scatter plot smoothing: gaussian kernel standard deviation


# -- Percentile value in [0, 100]: controls the noise-sensitivity of the staining plane search
# -- The larger this number is, a smaller number of points are selected to search for a staining plane
angular_density_percentile_cutoff = 80

# -- Percentile value in [0, 100]: controls the noise-sensitivity of the the stain vector search
# -- The lower this number is, narrower the angle of stain vector is.
# -- When the angle is small the umixing effect is stronger and the colors get more separated, but this can cause artifacts.
angular_percentile = 95  # try 90 to enhance unmixing effect, or 98 to reduce unmixing artifacts
# -- Additive angular shift: enhance or reduce the unmixing effect using a fixed anglular value (in radians)
# -- This has an additive/compensation effect on top of the "angular_percentile" variable and can cause artifacts
# -- negative value => reduces color unmixing; positive value => increases color unmixing
angular_shift = 0.0  # try -0.1 to reduce unmixing artifacts, or +0.1 to enhance unmixing effect

## SurGen-specific parameters
thresholds = {
    "frac_sat": 0.6,  # -- Minimum fraction of saturated pixels
    "entropy": 6.4,  # -- Minimum entropy value
    "colorfulness": 0.008,  # -- Minimum colorfulness value
    "lap_var": 100.0,  # -- Minimum variance of Laplacian (blurriness measure)
    "frac_redyellow": 0.2,  # -- Maximum fraction of red/yellow pixels
    "frac_green": 0.01,  # -- Maximum fraction of green pixels
}
