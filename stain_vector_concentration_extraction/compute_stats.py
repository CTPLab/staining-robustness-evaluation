"""
    Compute image quality metrics for each tile in the PLISM-wsi dataset:
    - Fraction of saturated pixels
    - Shannon entropy- Mean pixel intensity
    - Colorfulness score (based on hue variance and saturation)
    - Variance of Laplacian (blurriness measure)
    - Average hue and fractions of specific hue ranges (red/yellow, green, blue)
    - Circular variance of hue (color diversity measure)

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

from typing import Dict

import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv
from skimage.measure import shannon_entropy
from tqdm import tqdm


def colorfulness_variance(hsv: np.ndarray) -> float:
    """Return colorfulness score"""
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


def compute_tile_metrics(tile: np.ndarray) -> Dict[str, float]:
    """Compute image quality metrics for a given tile."""
    hsv = rgb2hsv(tile)
    sat, val, h = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    mask = sat > 0.1
    if np.any(mask):
        h_masked = h[mask]
        avg_hue = float(np.mean(h_masked))
        frac_redyellow = float(
            np.mean(((h_masked < 1 / 12) | (h_masked >= 11 / 12)) | ((h_masked >= 1 / 12) & (h_masked < 1 / 6)))
        )
        frac_green = float(np.mean((h_masked >= 1 / 4) & (h_masked < 5 / 12)))
        frac_blue = float(np.mean((h_masked >= 5 / 12) & (h_masked < 7 / 12)))
        hue_circular_var = float(
            1 - np.sqrt(np.mean(np.cos(2 * np.pi * h_masked)) ** 2 + np.mean(np.sin(2 * np.pi * h_masked)) ** 2)
        )
    else:
        avg_hue = frac_redyellow = frac_green = frac_blue = hue_circular_var = np.nan
    frac_sat = float(np.mean(sat > 0.2))
    mean_val = float(np.mean(val))
    entropy = float(shannon_entropy(tile))
    colorfulness = colorfulness_variance(hsv)
    gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "frac_sat": frac_sat,
        "entropy": entropy,
        "mean_val": mean_val,
        "colorfulness": float(colorfulness),
        "lap_var": lap_var,
        "avg_hue": avg_hue,
        "frac_redyellow": frac_redyellow,
        "frac_green": frac_green,
        "frac_blue": frac_blue,
        "hue_circular_var": hue_circular_var,
    }


if __name__ == "__main__":
    ### CONFIGURATION ###
    data_dir = ""  # <--- Path to PLISM-wsi dataset
    stats_dir = ""  # <--- Path to store computed img metrics
    #####################

    df = pd.read_csv(f"{data_dir}/PLISM_wsi_en.csv")
    for stain, stain_df in df.groupby("stain"):
        for device, subset_df in stain_df.groupby("device"):
            all_metrics = []
            print(stain, device)
            for _, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
                img_name = row["path"].split("/")[-1]
                fp = f"{data_dir}/{row['stain']}_{row['device']}/{img_name}"
                tile = cv2.imread(fp)[:, :, ::-1] / 255.0
                metrics = compute_tile_metrics(tile)
                metrics.update({"fp": fp})
                all_metrics.append(metrics)
            df_metrics = pd.DataFrame(all_metrics)
            csv_path = f"{stats_dir}/metrics_{stain}_{device}.csv"
            df_metrics.to_csv(csv_path, index=False)
            print(f"Saved metrics to {csv_path}")
            print(df_metrics.describe())
