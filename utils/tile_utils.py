"""
    Utility functions for tile generation and extraction from whole slide images (WSIs).

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import openslide
from PIL import Image
from shapely.geometry import Polygon


def get_mpp(slide: openslide.OpenSlide, slide_id: str) -> float:
    if openslide.PROPERTY_NAME_MPP_X in slide.properties.keys():
        if float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) < 10:
            mpp_x = slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = slide.properties[openslide.PROPERTY_NAME_MPP_Y]
        else:  # HISTAI WSIs contain mpp=1000, https://huggingface.co/datasets/histai/HISTAI-metadata/discussions/8
            if m := re.search(r"(?:^|_)x(\d{2})(?:_|$)", slide_id):
                magnification = int(m.group(1))
                assert 1 <= magnification <= 40, f"Unexpected magnification: x{magnification} in '{slide_id}'"
                mpp_x = mpp_y = 10 / magnification
            else:
                mpp_x = mpp_y = 0.5  # 20x
    elif slide.properties.get("openslide.comment") or slide.properties.get("tiff.ImageDescription"):
        xml_str = slide.properties.get("openslide.comment") or slide.properties.get("tiff.ImageDescription")
        root = ET.fromstring(xml_str)
        pixels = root.find(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels")
        mpp_x = float(pixels.attrib["PhysicalSizeX"])
        mpp_y = float(pixels.attrib["PhysicalSizeY"])
    else:
        print(
            f"No MPP in slide properties or comment/ Image Description, slide property keys: {slide.properties.keys()}"
        )
        raise ValueError
    if mpp_x != mpp_y:
        print(f"Mismatch in mpp x and mpp y: mpp_x={mpp_x}, mpp_y={mpp_y}, using mpp x")
    return float(mpp_x)


def create_tissue_tiles(
    wsi: openslide.OpenSlide,
    tissue_mask: np.ndarray,
    tile_size_microns: int,
    slide_id: str,
    offsets_micron: int = None,
) -> List[Polygon]:
    print(f"tile size is {tile_size_microns} µm")
    mpp_scale_factor = get_mpp(wsi, slide_id)
    tile_size_pix = round(tile_size_microns / mpp_scale_factor)

    wsi_w, wsi_h = wsi.level_dimensions[0]
    mask_h, mask_w = tissue_mask.shape[:2]
    scale_x = wsi_w / mask_w
    scale_y = wsi_h / mask_h
    print(f"Using mask scale factors x={scale_x:.3f}, y={scale_y:.3f}")
    print(f"WSI dimensions (level 0): width={wsi_w}, height={wsi_h}, Mask dimensions: width={mask_w}, height={mask_h}")

    ys, xs = np.nonzero(tissue_mask)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    print(f"Tissue mask bounding box (mask pixel coordinates): minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")

    # convert to WSI pixel coordinates
    minx = int(minx * scale_x)
    maxx = int(maxx * scale_x)
    miny = int(miny * scale_y)
    maxy = int(maxy * scale_y)
    print(f"Tissue mask bounding box (WSI pixel coordinates): minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")

    tissue_margin_pix = tile_size_pix * 2
    min_offset_x = minx - tissue_margin_pix
    min_offset_y = miny - tissue_margin_pix
    offsets = [(min_offset_x, min_offset_y)]

    if offsets_micron is not None:
        offset_pix = [round(o / mpp_scale_factor) for o in offsets_micron]
        offsets = [(o + min_offset_x, o + min_offset_y) for o in offset_pix]

    filtered_tiles = generate_tiles(
        tile_size_pix,
        tile_size_pix,
        maxx + tissue_margin_pix,
        maxy + tissue_margin_pix,
        offsets=offsets,
        mask=tissue_mask,  # pass original mask
        mask_scale=(scale_x, scale_y),
    )
    print(f"Finished generating N={len(filtered_tiles)} tiles.")
    return filtered_tiles


def generate_tiles(
    tile_width_pix: int,
    tile_height_pix: int,
    img_width: int,
    img_height: int,
    offsets: List[Tuple[int, int]] = [(0, 0)],
    mask: np.ndarray = None,
    mask_scale: Tuple[float, float] = (1.0, 1.0),
) -> List[Polygon]:
    tiles = []
    if mask is not None:
        scale_x = 1 / mask_scale[0]
        scale_y = 1 / mask_scale[1]
    else:
        scale_x = scale_y = 1.0

    for xmin, ymin in offsets:
        for x in range(
            int(np.floor(xmin)),
            int(np.ceil(img_width + tile_width_pix)),
            tile_width_pix,
        ):
            for y in range(
                int(np.floor(ymin)),
                int(np.ceil(img_height + tile_height_pix)),
                tile_height_pix,
            ):
                if mask is not None:
                    mx0 = int(x * scale_x)
                    my0 = int(y * scale_y)
                    mx1 = int((x + tile_width_pix) * scale_x)
                    my1 = int((y + tile_height_pix) * scale_y)
                    tile_mask = mask[my0:my1, mx0:mx1]
                    if not tile_mask.any():
                        # tile at x={x}, y={y}: no tissue in mask")
                        continue
                tiles.append(
                    Polygon(
                        [
                            (x, y),
                            (x + tile_width_pix, y),
                            (x + tile_width_pix, y - tile_height_pix),
                            (x, y - tile_height_pix),
                        ]
                    )
                )
    return tiles


def extract_tile_from_slide(
    slide: openslide.OpenSlide,
    rect: Polygon,
    out_size: int,
    use_lower_level: bool = False,
) -> Image:
    minx, miny, maxx, maxy = rect.bounds
    # Note that the y-axis is flipped in the slide: the top of the shapely polygon is y = ymax,
    # but in the slide it is y = 0. Hence: miny instead of maxy.
    top_left_coords = (int(minx), int(miny))  # at level 0
    if use_lower_level:
        ideal_f = int(maxx - minx) / out_size
        level = slide.get_best_level_for_downsample(ideal_f)
        f = slide.level_downsamples[level]
        w, h = round((maxx - minx) / f), round((maxy - miny) / f)
    else:
        w, h = int(maxx - minx), int(maxy - miny)

    tile = slide.read_region(top_left_coords, level, (w, h))
    tile = tile.convert("RGB").resize((out_size, out_size))
    return np.array(tile)
