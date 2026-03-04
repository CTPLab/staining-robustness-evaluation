"""
    Extract H&E stain vectors and intensities from WSIs [v1]

    The script performs the following steps:
    * Load WSIs and their corresponding tissue masks, OR generate tissue masks if not provided
    * Randomly sample tiles from the tissue regions
    * Apply quality thresholds (e.g. saturation, entropy, colorfulness) to filter out low-quality tiles
    * Perform stain unmixing on the selected tiles
    * Export the extracted stain vectors and intensities
    * Generate visual analysis reports for each tile, including the unmixing results and quality metrics

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

import os
import random
import shutil
from glob import glob
from typing import Dict, List, Tuple

import config as _c
import cv2
import numpy as np
import openslide
import pandas as pd
import PIL
import torch
import utils_global as _ub
from PIL import Image
from shapely.geometry import Polygon
from skimage.color import rgb2hsv
from skimage.measure import shannon_entropy
from tqdm import tqdm
from utils.tile_utils import create_tissue_tiles, extract_tile_from_slide
from utils.tissue_detector import TissueDetector
from utils.utils_stainUnmix import staining_unmix


def to_spherical(v: np.ndarray) -> np.ndarray:
    theta = np.arctan2(v[1], v[0])
    phi = np.arctan2(v[2], np.sqrt(v[0] ** 2 + v[1] ** 2))
    return np.array([theta, phi])


def get_intensity(image: np.ndarray, stain_source: np.ndarray, thresh: float = 0.1) -> Dict[str, float]:
    max_intensity_vect = np.amax(image, axis=(0, 1))
    max_intensity_vect = max_intensity_vect.astype(np.float32)
    max_intensity_vect = np.maximum(max_intensity_vect, 1)
    mat_proj_inverse = np.linalg.inv(stain_source)
    image_OD = np.maximum(1, image)
    image_OD = image_OD / max_intensity_vect
    image_OD = -1.0 * np.log(image_OD)
    image_proj = np.matmul(image_OD, mat_proj_inverse.T)

    # -- Track median (above min threshold) of H and E channels
    channels = {0: "H", 1: "E"}
    stats = {}
    masks = {}
    for ch, label in channels.items():
        mask = image_proj[..., ch] > thresh
        masks[label] = mask
        values = image_proj[mask, ch]
        if values.size == 0:
            stats[f"{label}_median"] = np.nan
            stats[f"{label}_90th"] = np.nan
            stats[f"{label}_95th"] = np.nan
            stats[f"{label}_99th"] = np.nan
            stats[f"{label}_max"] = np.nan
            continue
        stats[f"{label}_median"] = float(np.median(values))
        stats[f"{label}_90th"] = float(np.percentile(values, 90))
        stats[f"{label}_95th"] = float(np.percentile(values, 95))
        stats[f"{label}_99th"] = float(np.percentile(values, 95))
        stats[f"{label}_max"] = float(np.max(values))
    return stats


def colorfulness_variance_filtered(hsv: np.ndarray) -> float:
    """Return colorfulness score and reason if excluded due to dominant unwanted hues."""
    h, s = hsv[..., 0] * 360.0, hsv[..., 1]
    mask = s > 0.1
    if not np.any(mask):
        return 0.0

    h_masked = h[mask]
    h_rad = np.deg2rad(h_masked)
    R = np.sqrt(np.mean(np.cos(h_rad)) ** 2 + np.mean(np.sin(h_rad)) ** 2)
    hue_var = 1 - R
    mean_sat = np.mean(s[mask])
    score = float(hue_var * mean_sat)
    return score


def keep_tile(tile: np.ndarray, thresholds: Dict[str, float], tile_name: str = "") -> bool:
    hsv = rgb2hsv(tile)
    sat, val, h = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    mask = sat > 0.1
    frac_sat = float(np.mean(sat > 0.2))
    if frac_sat < thresholds["frac_sat"]:
        print(f"Ignore {tile_name}: low saturation ({frac_sat:.2f} < {thresholds['frac_sat']:.2f})")
        return False
    entropy = float(shannon_entropy(tile))
    colorfulness = colorfulness_variance_filtered(hsv)
    gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if np.any(mask):
        h_masked, s_masked, v_masked = h[mask], sat[mask], val[mask]
        # Bright red, yellow, and brown/orange tones
        red_mask = (h_masked < 1 / 12) | (h_masked >= 11 / 12)
        yellow_mask = (h_masked >= 1 / 12) & (h_masked < 1 / 6)
        brown_orange_mask = (h_masked >= 1 / 12) & (h_masked < 1 / 3)  # orange → brown range
        # Allow both bright and darker saturated colors
        color_mask = red_mask | yellow_mask | brown_orange_mask
        sat_val_mask = (s_masked > 0.4) & (v_masked > 0.3)  # keep dark brown, exclude dull
        frac_redyellow = float(np.mean(color_mask & sat_val_mask))
        frac_green = float(np.mean((h_masked >= 1 / 4) & (h_masked < 5 / 12)))
    else:
        frac_redyellow = frac_green = 0.0

    print(
        f"Tile {tile_name}: frac_sat={frac_sat:.2f}, entropy={entropy:.2f}, colorfulness={colorfulness:.5f}, lap_var={lap_var:.2f}, frac_redyellow={frac_redyellow:.2f}, frac_green={frac_green:.2f}"
    )
    if frac_sat < thresholds["frac_sat"]:
        print(f"Ignore {tile_name}: low saturation ({frac_sat:.2f} < {thresholds['frac_sat']:.2f})")
        return False
    if entropy < thresholds["entropy"]:
        print(f"Ignore {tile_name}: low entropy ({entropy:.2f} < {thresholds['entropy']:.2f})")
        return False
    if colorfulness < thresholds["colorfulness"]:
        print(f"Ignore {tile_name}: low colorfulness ({colorfulness:.5f} < {thresholds['colorfulness']:.5f})")
        return False
    if lap_var < thresholds["lap_var"]:
        print(f"Ignore {tile_name}: blurry tile ({lap_var:.2f} < {thresholds['lap_var']:.2f})")
        return False
    if frac_redyellow > thresholds["frac_redyellow"]:
        print(
            f"Ignore {tile_name}: dominated by red/yellow ({frac_redyellow:.2f} > {thresholds['frac_redyellow']:.2f})"
        )
        return False
    if frac_green > thresholds["frac_green"]:
        print(f"Ignore {tile_name}: dominated by green ({frac_green:.2f} > {thresholds['frac_green']:.2f})")
        return False
    else:
        return True


def get_random_valid_tiles(
    wsi: openslide.OpenSlide,
    filtered_tiles: List[Polygon],
    out_px: int,
    num_tiles: int,
    thresholds: Dict[str, float],
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Return exactly num_tiles random valid tiles and their bounding box coordinates."""
    valid_tiles, indices = [], list(range(len(filtered_tiles)))
    random.shuffle(indices)
    for idx in indices:
        if len(valid_tiles) >= num_tiles:
            break
        minx, miny, maxx, maxy = filtered_tiles[idx].bounds
        tile = extract_tile_from_slide(wsi, filtered_tiles[idx], out_px, use_lower_level=True)
        if keep_tile(tile, thresholds, f"minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}"):
            print(f"{len(valid_tiles)+1}/{num_tiles} valid tiles found")
            valid_tiles.append((tile, (int(minx), int(miny), int(maxx), int(maxy))))
    return valid_tiles


if __name__ == "__main__":
    ### CONFIGURATION ###
    num_tiles = 10  # <--- Number of valid tiles to extract per WSI
    redo: bool = False  # <--- Force re-extraction of tiles
    tissue_masks_dir = None  # <--- Path to precomputed tissue masks, script generates masks if not provided
    csv = "../SURGEN.csv"  # <--- Path to SurGen CSV file
    output_data_dir = f"./stain_vectors"  # <--- Path to store extracted stain vectors and intensities
    output_report_dir = f"./logs"  # <--- Path to store analysis report images for each tile
    output_intensity_dir = f"./intensities"  # <--- Path to store extracted stain intensities
    tile_size = 224  # um
    out_px = 448  # px
    ######################

    random.seed(42)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_report_dir, exist_ok=True)
    os.makedirs(output_intensity_dir, exist_ok=True)
    df = pd.read_csv(csv)
    df = df[(~df["qc_excluded"]) & (~df["MSI"].isna())]
    df["cohort"] = (
        df["slide_id"].str.extract(r"(SR386|SR1482)", expand=False)
        if "SURGEN" in csv
        else df["patient_id"].str.split("-").str[1]
    )

    thresholds = _c.thresholds
    with tqdm(enumerate(zip(df["slide_id"], df["slide_path"], df["cohort"])), total=len(df)) as pbar:
        for idx, (slide_id, slide_path, cohort) in pbar:
            if os.path.exists(f"{output_data_dir}/{slide_id}.npz"):
                if not redo:
                    num_tiles_slide = len(glob(f"{output_data_dir}/{slide_id}/*.npz"))
                    if num_tiles_slide == 10:
                        continue
                    else:
                        print(f"Only {num_tiles_slide}/10 tiles found, repeat: {slide_id}")

                shutil.rmtree(f"{output_data_dir}/{slide_id}")
                shutil.rmtree(f"{output_report_dir}/{slide_id}")
                os.remove(f"{output_data_dir}/{slide_id}.npz")

            pbar.set_postfix(slide_id=slide_id)

            wsi = openslide.open_slide(slide_path)
            try:
                orig_mpp = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
            except KeyError:
                raise Exception(
                    f"Could not find the {openslide.PROPERTY_NAME_MPP_X} in the slide {slide_id}, all keys: {wsi.properties.keys()}"
                )

            if tissue_masks_dir is None:
                tissue_detector = TissueDetector(
                    tissue_method="fcnn",
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
                mask = tissue_detector.detect_tissue(wsi, slide_id)
                os.makedirs(f"./tissue_masks", exist_ok=True)
                PIL.Image.fromarray((mask * 255).astype(np.uint8)).save(f"./tissue_masks/{slide_id}.png")
                print(f"Saved tissue mask to ./tissue_masks/{slide_id}.png")
            else:
                mask = np.array(Image.open(f"{tissue_masks_dir}/{slide_id}.png"))
            filtered_tiles = create_tissue_tiles(wsi, mask, tile_size, slide_id)
            print("mask.shape", mask.shape)
            export_dicts = []
            tiles_with_coords = get_random_valid_tiles(wsi, filtered_tiles, out_px, num_tiles, thresholds=thresholds)
            wsi.close()

            if len(tiles_with_coords) == 0:
                print(f"No valid tiles found for slide {slide_id}, skipe slide {slide_id} for now")
                continue

            for image, (x_min, y_min, x_max, y_max) in tiles_with_coords:
                MAE_score, report_dict, max_intensity_vect, HE_max_val, stainMatrix = staining_unmix(
                    image,
                    _c.patch_subsampling_factor,
                    _c.background_cutoff_percentile,
                    _c.foreground_cutoff,
                    _c.scatterplot_smoothing_kernel,
                    _c.scatterplot_smoothing_stdev,
                    _c.angular_density_percentile_cutoff,
                    _c.angular_percentile,
                    _c.angular_shift,
                )
                print(f">> Image sucessfully unmixed (reconstruction.MAE={MAE_score})")
                stainMatrix_inv = None
                if np.any(stainMatrix):
                    stainMatrix_inv = np.linalg.inv(stainMatrix)

                export_dict = {
                    "maxIntensityVect": max_intensity_vect.astype(np.float32),
                    "maxStainIntensityVect": HE_max_val.astype(np.float32),
                    "stainMatrix": stainMatrix.astype(np.float32),
                    "stainMatrixInv": stainMatrix_inv.astype(np.float32),
                }

                report_image_list = []
                for k, v in report_dict.items():
                    _text = _ub.text2np(k, 14)
                    img_c = _ub.concat_vertical([v, _text])
                    report_image_list.append(img_c)

                H, E = stainMatrix[:, 0], stainMatrix[:, 1]
                h_sph, e_sph = to_spherical(H), to_spherical(E)
                h_sph, e_sph = to_spherical(H), to_spherical(E)
                if h_sph[1] > e_sph[1] and h_sph[0] > e_sph[0]:
                    png_fp = f"{output_report_dir}/{slide_id}/{slide_id}_xmin={x_min}_ymin={y_min}_xmax={x_max}_ymax={y_max}_analysisReport_ignored.png"
                    npz_fp = f"{output_data_dir}/{slide_id}/{slide_id}_xmin={x_min}_ymin={y_min}_xmax={x_max}_ymax={y_max}_ignored.npz"
                    print(
                        f">> Hematoxylin and Eosin appear swapped, ignore tile for mean calculation, (theta_H, phi_H)={h_sph}, (theta_E, phi_E)={e_sph}."
                    )
                else:
                    png_fp = f"{output_report_dir}/{slide_id}/{slide_id}_xmin={x_min}_ymin={y_min}_xmax={x_max}_ymax={y_max}_analysisReport.png"
                    npz_fp = f"{output_data_dir}/{slide_id}/{slide_id}_xmin={x_min}_ymin={y_min}_xmax={x_max}_ymax={y_max}.npz"
                    print(f">> Compute intensity, (theta_H, phi_H)={h_sph}, (theta_E, phi_E)={e_sph}.")
                    c_stat = get_intensity(image, stainMatrix)
                    print(f">> 95th percentile intensity values: H={c_stat['H_95th']}, E={c_stat['E_95th']}")
                    export_dict = {**export_dict, **c_stat}
                    export_dicts.append(export_dict)

                if not os.path.isdir(f"{output_report_dir}/{slide_id}"):
                    os.makedirs(f"{output_report_dir}/{slide_id}", exist_ok=True)
                image_out = _ub.concat_horizontal(report_image_list)
                image_out = image_out.astype(np.uint8)
                PIL.Image.fromarray(image_out).save(png_fp)
                print(f">> Analysis report exported at: {png_fp}")

                if not os.path.isdir(f"{output_data_dir}/{slide_id}"):
                    os.makedirs(f"{output_data_dir}/{slide_id}", exist_ok=True)
                np.savez(npz_fp, **export_dict)
                print(f">> Stain data exported at: {npz_fp}")

            average_export_dict = {}
            for key in export_dict.keys():
                values = [d[key] for d in export_dicts]
                arr = np.array(values, dtype=np.float32)
                average_export_dict[key] = np.median(arr, axis=0)

            if not os.path.isdir(output_data_dir):
                os.makedirs(output_data_dir, exist_ok=True)
            path_target = f"{output_data_dir}/{slide_id}.npz"
            np.savez(path_target, **average_export_dict)
            print(f">> Stain data exported at: {path_target}")
