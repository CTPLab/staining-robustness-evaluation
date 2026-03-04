# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]unibas[dot]ch"
    __creation__ = "2025"
"""
import sys

sys.dont_write_bytecode = True

import numpy as np
import scipy.signal

try:
    from utils import utils_global as _ub
except ImportError:
    import utils_global as _ub


def staining_unmix(
    image_patch: np.ndarray,
    patch_subsampling_factor: float,
    background_cutoff_percentile: int,
    foreground_cutoff: int,
    scatterplot_smoothing_kernel: int,
    scatterplot_smoothing_stdev: int,
    angular_density_percentile_cutoff: float,
    angular_percentile: float,
    angular_shift: float,
):
    """INPUT:
        - image patch, numpy format, shape [height, width, channels(rgb)]
    OUTPUT:
        - report_dictionary
        - maximum intensity vector
        - stain matrix [3,3]
        - stain maximum concentration vector [2]
    """
    height_base, width_base = image_patch.shape[:2]

    """ 0) OUTPUT DICTIONARY
    """
    report_dict = {}
    report_dict["original image"] = image_patch

    """ 1) MAX INTENSITY COMPUTATION
    """
    max_intensity_vect = np.amax(image_patch, axis=(0, 1))
    max_intensity_vect = max_intensity_vect.astype(np.float32)

    # -- Reporting
    rect_bg = np.zeros([height_base, 32, 3])
    rect_bg = rect_bg + max_intensity_vect
    rect_bg = _ub.draw_frame(rect_bg, 4)
    report_dict["BG"] = rect_bg  # -- background color

    """ 2) SUBSAMPLE AND FLATTEN IMAGE
    """
    _slice = slice(0, None, patch_subsampling_factor)
    image_sub = image_patch[_slice, _slice, :]
    pixel_count = image_sub.shape[0] * image_sub.shape[1]
    image_flat = np.reshape(image_sub, [pixel_count, -1])

    """ 3) INTENSITY-BASED FILTERING
        >> Exclude pixels with too-high intensity (ambiguous unmixing)
        >> Exclude pixels with too-low intensity (ambiguous unmixing)
    """
    # -- Maximum intensity
    image_flat_min = np.amin(image_flat, axis=1)  # -- RGB projection, changed to amin (prev amax)
    background_cutoff = min(np.percentile(image_flat_min, background_cutoff_percentile), 240)
    print(f"Background_cutoff to {background_cutoff_percentile}th percentile or max 240: {background_cutoff}")
    image_flat_keep = np.where(image_flat_min < background_cutoff)
    keep_indices = image_flat_keep[0]

    # -- 2-percent safety rule
    if len(keep_indices) < 0.02 * pixel_count:
        print(f">> Insufficient valid pixel count (n={len(keep_indices)})")
        return None

    print(
        f"High intensity pixels discarded: {len(np.where(image_flat_min >= background_cutoff)[0])}, {len(np.where(image_flat_min >= background_cutoff)[0])/pixel_count*100:.2f}%"
    )
    image_flat = image_flat[keep_indices, :]

    # -- Minimum intensity
    image_flat_max = np.amax(image_flat, axis=1)
    image_flat_keep = np.where(image_flat_max > foreground_cutoff)
    keep_indices = image_flat_keep[0]

    print(
        f"Low intensity pixels discarded: {len(np.where(image_flat_max <= foreground_cutoff)[0])}, {len(np.where(image_flat_max <= foreground_cutoff)[0])/pixel_count*100:.2f}%"
    )
    image_flat = image_flat[keep_indices, :]

    # -- Safety rule
    if len(keep_indices) < 0.02 * pixel_count:
        print(f">> Insufficient valid pixel count (n={len(keep_indices)})")
        return None

    """ 4) OPTICAL DENSITY TRANSFORM
    """
    image_flat = np.maximum(1, image_flat)
    OD_flat = image_flat / max_intensity_vect
    OD_flat = -1.0 * np.log(OD_flat)

    """ 5) COMPUTE AN ANGULAR TRANSFORM OF THE IMAGE
        - ANGLE of OD-VECT with PROJECTION on RG-PLANE
        - ANGLE of OD-VECT with PROJECTION on RB-PLANE
    """
    angle_mat = np.zeros(
        [OD_flat.shape[0], 2]
    )  # -- Every pixel gets its coordinates transformed in a spherical coordinate system
    for _i in range(OD_flat.shape[0]):  # -- Scanning of the pixels
        _v = OD_flat[_i, :]
        _n = np.sqrt(np.sum(np.square(_v))) + 1e-5  # -- L2-norm computation
        _vn = _v / _n  # -- Normalized vector

        # -- BASE ANGLE CALCULATION (ANGLE w/ R-axis)
        RG_proj = np.copy(_vn)
        RG_proj[2] = 0
        _n = np.sqrt(np.sum(np.square(RG_proj)))  # -- L2-norm computation
        RG_proj = RG_proj / _n  # -- Normalized vector
        angle_base = np.acos(RG_proj[0])  # -- ANGLE w/ R-axis

        # -- LIFT ANGLE CALCULATION (ANGLE w/ B-axis)
        vect_dot_product = np.sum(_vn * RG_proj)
        angle_lift = np.acos(vect_dot_product)  # -- ANGLE w/ B-axis

        # -- STORE RESULT
        angle_mat[_i, :] = [angle_base, angle_lift]

    """ 6) GENERATE A 2-D SCATTER PLOT OF THE TRANSFORMED POINTS
    """
    y_bins, x_bins = 256, 256
    scatter_c = np.zeros([y_bins, x_bins])
    x_min, y_min = 0, 0
    x_max, y_max = (np.pi / 2,) * 2
    angle_coordinates_mat = np.zeros([angle_mat.shape[0], 2], dtype=np.int32)
    for idx_c in range(angle_mat.shape[0]):
        y_c, x_c = angle_mat[idx_c, :]

        # -- Rescaling to fit points in the scatter plot
        y_b = int((y_bins - 1) * (y_c - y_min) / (y_max - y_min))
        x_b = int((x_bins - 1) * (x_c - x_min) / (x_max - x_min))
        angle_coordinates_mat[idx_c, :] = [y_b, x_b]  # -- Track angular coordinates

        # -- Draw point in the scatter plot
        scatter_c[y_b, x_b] += 1  # -- Increment

    # -- Smoothing of the scatter plot
    _kernlen = scatterplot_smoothing_kernel
    _stdev = scatterplot_smoothing_stdev
    gkern1d = scipy.signal.windows.gaussian(_kernlen, std=_stdev)
    gkern1d = gkern1d / np.sum(gkern1d)
    gkern1d = gkern1d.astype(np.float32)

    # -- Vertical smoothing
    _ker_v = gkern1d[None, :]
    scatter_c = scipy.signal.convolve2d(scatter_c, _ker_v, mode="same", boundary="symm")

    # -- Horizontal smoothing
    _ker_h = gkern1d[:, None]
    scatter_c = scipy.signal.convolve2d(scatter_c, _ker_h, mode="same", boundary="symm")

    # -- Reporting
    _visu = 1 - scatter_c[..., None] / np.amax(scatter_c)  # -- Normalized in [0, 1]
    _visu = np.repeat(_visu, 3, axis=2)  # -- Channel axis
    _visu = 255 * _visu
    _visu = _ub.draw_frame(_visu, 1)
    report_dict["angular scatter plot"] = _visu

    """ 7) APPLY A DENSITY PERCENTILE THRESHOLDING
    """
    _scatter_values = scatter_c[np.where(scatter_c > 0)]
    _scatter_cutoff = np.percentile(_scatter_values, q=angular_density_percentile_cutoff)

    # -- Reporting
    _visu = scatter_c > _scatter_cutoff
    _visu = 1 - _visu[..., None]
    _visu = np.repeat(_visu, 3, axis=2)
    _visu = 255 * _visu
    _visu = _ub.draw_frame(_visu, 1)
    report_dict[f"angular scatter plot (thesholded; p={angular_density_percentile_cutoff})"] = _visu

    # -- Reporting (colored scatter plot)
    _scatter_bin = scatter_c > _scatter_cutoff  # -- Binary scatter plot
    _visu = 255 * np.ones([y_bins, x_bins, 3])  # -- White canvas
    for _y in range(y_bins):
        for _x in range(x_bins):
            if _scatter_bin[_y, _x]:
                # -- Rescaling [0, pi/2]
                y_b = y_max * _y / (y_bins - 1)
                x_b = x_max * _x / (x_bins - 1)

                # -- Retrieve color from angles
                _v = [np.cos(y_b), np.sin(y_b), np.sin(x_b)]
                _v = np.array(_v)
                _v = _v / _ub.vnorm(_v)
                _v = 1.5 * _v  # -- Amplify
                color_c = 255 * np.exp(-1 * _v)  # -- OD.inversion

                color_c = np.maximum(0, color_c)
                color_c = np.minimum(255, color_c)
                _visu[_y, _x, :] = color_c
    _visu = _ub.draw_frame(_visu, 1)
    report_dict["colored angular scatter plot"] = _visu

    """ 8) FILTER POINTS BASED ON THEIR ANGULAR DENSITY
    """
    densityScores_list = np.array([scatter_c[y, x] for y, x in angle_coordinates_mat])
    densityFilter = densityScores_list >= _scatter_cutoff
    OD_filtered = OD_flat[densityFilter, :]  # -- Application of the filter

    """ 9) EIGEN ANALYSIS OF THE FILTERED POINTS
    """
    C = np.matmul(OD_filtered.T, OD_filtered)
    C /= OD_filtered.shape[0]
    eigVals, eigMat = np.linalg.eig(C)

    eig_sorted = np.argsort(eigVals)[::-1]
    eigVals = eigVals[eig_sorted]
    eigMat = np.stack([eigMat[:, idx] for idx in eig_sorted], axis=1)

    # -- Flip negative vectors
    for col_idx in range(eigMat.shape[1]):
        if np.amax(eigMat[:, col_idx]) < 0:
            eigMat[:, col_idx] *= -1

    """ 10) IDENTIFY THE STAIN VECTORS
    """
    mat_proj_12 = np.stack([eigMat[:, 0], eigMat[:, 1]], axis=1)
    OD_proj = np.matmul(OD_flat, mat_proj_12)

    OD_angles = np.arctan2(OD_proj[:, 0], OD_proj[:, 1])

    angle_min = np.percentile(OD_angles, q=100 - angular_percentile)
    angle_max = np.percentile(OD_angles, q=angular_percentile)

    angle_min = angle_min + angular_shift
    angle_max = angle_max - angular_shift

    vect_HEa = np.cos(angle_min) * mat_proj_12[:, 1] + np.sin(angle_min) * mat_proj_12[:, 0]
    vect_HEb = np.cos(angle_max) * mat_proj_12[:, 1] + np.sin(angle_max) * mat_proj_12[:, 0]

    # -- HEa -> Hematoxylin (blocks RED channel the most)
    # -- HEb -> Eosin
    if vect_HEb[0] > vect_HEa[0]:
        vect_HEt = vect_HEb
        vect_HEb = vect_HEa
        vect_HEa = vect_HEt

    """ 11) IDENTIFY THE RESIDUAL VECTOR (orthogonal vector to HE-plane)
    """
    vect_res = [0] * 3
    vect_res[0] = vect_HEa[1] * vect_HEb[2] - vect_HEa[2] * vect_HEb[1]
    vect_res[1] = -1 * (vect_HEa[0] * vect_HEb[2] - vect_HEa[2] * vect_HEb[0])
    vect_res[2] = vect_HEa[0] * vect_HEb[1] - vect_HEa[1] * vect_HEb[0]
    vect_res = np.array(vect_res)

    """ 12) CALCULATE PROJECTION MATRIX (= STAIN MATRIX)
    """
    mat_proj_final = np.stack([vect_HEa, vect_HEb, vect_res], axis=1)
    mat_proj_inverse = np.linalg.inv(mat_proj_final)

    """ 13) VISUALIZE INDIVIDUAL STAIN VECTORS AS A GRADIENT
    """
    step_s = 16
    rect_h = height_base
    rect_step_nb = rect_h // step_s

    for k, v in {"H": vect_HEa, "E": vect_HEb}.items():
        rect_c = np.zeros([rect_h, 32, 3])
        for y in range(rect_step_nb):
            av = y / (rect_step_nb - 1)
            color_c = av * np.log(255) * (v / np.amax(v))
            color_c = 255 * np.exp(-1 * color_c)
            slice_c = slice(y * step_s, (1 + y) * step_s)
            rect_c[slice_c, ...] = color_c
        rect_c = _ub.draw_frame(rect_c, 4)
        report_dict[f"{k}"] = rect_c

    """ 14) APPLY THE UNMIXING TRANSFROM AND COMPUTE THE MAXIMUM STAIN CONCENTRATIONS
    """
    image_OD = np.maximum(1, image_patch)
    image_OD = image_OD / max_intensity_vect
    image_OD = -1.0 * np.log(image_OD)

    # -- Inverse projection
    image_proj = np.matmul(image_OD, mat_proj_inverse.T)

    # -- Track max values
    H_max_val = np.amax(image_proj[..., 0])
    E_max_val = np.amax(image_proj[..., 1])
    HE_max_val = np.array([H_max_val, E_max_val])
    print(f">> Proj max intensities = {H_max_val, E_max_val}")

    # -- Forward projection
    image_backward = np.matmul(image_proj, mat_proj_final.T)

    # -- Compute reconstruction
    image_rec = max_intensity_vect * np.exp(-1 * image_backward)
    image_rec = np.maximum(0, image_rec)
    image_rec = np.minimum(255, image_rec)
    image_rec = image_rec.astype(np.uint8)

    """ 15) CALCULATE RECONSTRUCTION ERROR
    """
    MAE_score = np.abs(image_patch - image_rec)
    MAE_score = np.mean(MAE_score)
    MAE_score = int(MAE_score * 1000) / 1000

    # -- Reporting
    report_dict[f"image reconstruction (MAE={MAE_score})"] = image_rec

    """ 16) VISUALIZATION: H-only, E-only
    """
    image_OD = np.maximum(1, image_patch)
    image_OD = image_OD / max_intensity_vect
    image_OD = -1.0 * np.log(image_OD)
    image_proj = np.matmul(image_OD, mat_proj_inverse.T)  # -- Inverse projection

    for a, b in {0: "H", 1: "E"}.items():
        _image_processed = np.copy(image_proj)
        for i in [i for i in range(3) if i != a]:
            _image_processed[..., i] *= 0  # -- Attenuation of non-selected channels

        # -- Forward projection
        _image_processed = np.matmul(_image_processed, mat_proj_final.T)

        image_rec = max_intensity_vect * np.exp(-1 * _image_processed)
        image_rec = np.maximum(0, image_rec)
        image_rec = np.minimum(255, image_rec)
        image_rec = image_rec.astype(np.uint8)

        ## image_rec = image_patch #-- debug
        image_rec = _ub.draw_frame(image_rec, 2)
        report_dict[f"reconstruction: {b}-only"] = image_rec

    """ OUTPUT
    """
    return MAE_score, report_dict, max_intensity_vect, HE_max_val, mat_proj_final
