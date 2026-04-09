"""
    TissueDetector: A class to perform tissue detection on whole slide images (WSIs) using either Otsu's method or a pretrained fully convolutional neural network (FCNN).

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import openslide
import skimage.transform
import torch
from scipy import ndimage
from skimage.filters import threshold_otsu

from models.nature_net import NatureNet, compute_upsampling_and_padding


def compute_diagonal(bbox: List[int]) -> float:
    """Compute the diagonal length of a bounding box."""
    minr, minc, maxr, maxc = bbox
    return np.hypot(maxr - minr, maxc - minc)


def filter_regions(
    mask: np.ndarray, diagonal_threshold: float = 0.0, full_connectivity: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Remove connected regions smaller than the diagonal threshold."""
    structure = ndimage.generate_binary_structure(2, 2 if full_connectivity else 1)
    labeled, _ = ndimage.label(mask, structure=structure)
    objects = ndimage.find_objects(labeled)

    keep_mask = np.zeros_like(mask, dtype=bool)
    kept_labels = []

    for i, sl in enumerate(objects):
        region_mask = labeled[sl] == i + 1
        bbox = [sl[0].start, sl[1].start, sl[0].stop, sl[1].stop]
        if compute_diagonal(bbox) >= diagonal_threshold:
            keep_mask[sl] |= region_mask
            kept_labels.append(i + 1)

    return keep_mask, labeled, kept_labels


def fill_holes(
    mask: np.ndarray,
    diagonal_threshold: float = 0.0,
    full_connectivity: bool = False,
    fill_value: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Fill holes in a mask that are smaller than the diagonal threshold."""
    inv_mask = ~mask
    structure = ndimage.generate_binary_structure(2, 2 if full_connectivity else 1)
    labeled, _ = ndimage.label(inv_mask, structure=structure)
    objects = ndimage.find_objects(labeled)

    filled_mask = mask.copy()
    filled_labels = []

    for i, sl in enumerate(objects):
        hole_mask = labeled[sl] == i + 1
        bbox = [sl[0].start, sl[1].start, sl[0].stop, sl[1].stop]
        if compute_diagonal(bbox) <= diagonal_threshold:
            filled_mask[sl][hole_mask] = fill_value
            filled_labels.append(i + 1)

    return filled_mask, labeled, filled_labels


def TissueSgm(pretrained_path: str = "./MODELS/TISSUE_SGM"):
    model = NatureNet()
    e = model.load_state_dict(
        torch.load(
            os.path.join(pretrained_path, "tissue_sgm_pretrained.pt"),
            map_location="cpu",
        ),
        strict=True,
    )
    print(e)
    return model


class TissueDetector:
    """Performs tissue detection."""

    def __init__(self, tissue_method: str, device: torch.device = None):
        """
        tissue_method: str, either "otsu" or "fcnn"
        device: torch.device, only relevant if tissue_method is "fcnn"
        """
        self.tissue_method = tissue_method
        self.device = device
        if self.tissue_method == "fcnn":
            self.fcnn_model = TissueSgm()
            self.fcnn_model.to(device)
            self.fcnn_model.eval()

    def detect_tissue(self, slide: openslide.OpenSlide, slide_id: str, mpp: float = 8) -> np.ndarray:
        """Detect tissue, generate a tissue mask for the WSI."""
        orig_mpp = self._get_mpp(slide, slide_id)
        level = slide.get_best_level_for_downsample(mpp / orig_mpp)
        print(
            f"Running tissue detection at level={level}, orig_mpp={orig_mpp}, f={slide.level_downsamples[level]}, shape={slide.level_dimensions[level]}"
        )
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])

        if self.tissue_method == "otsu":
            img = np.array(img.convert("L"))
            val = threshold_otsu(img)
            mask = img < val
        elif self.tissue_method == "fcnn":
            img = np.array(img.convert("RGB"))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255
            padding, upscale, output_shape = compute_upsampling_and_padding(self.fcnn_model, img_tensor.shape)
            pred = self.fcnn_model(img_tensor.to(self.device))
            pred = pred.cpu().detach().numpy()
            rescaled_pred = np.array(
                [
                    skimage.transform.rescale(
                        pred[c, :, :],
                        (upscale[0], upscale[1]),
                        preserve_range=True,
                        order=True,
                        mode="edge",
                    )
                    for c in range(pred.shape[0])
                ],
                dtype="float32",
            )
            rescaled_padded_result = np.pad(
                rescaled_pred,
                ((0, 0), (padding[2], padding[3]), (padding[0], padding[1])),
                "constant",
            )

            assert (
                img_tensor.shape[1:] == rescaled_padded_result.shape[1:]
            ), f"Mismatch in input img and output pred shape: input_shape={img_tensor.shape[1:]}, output_shape={rescaled_padded_result.shape[1:]}"
            mask = (
                rescaled_padded_result[-1, :, :] > 0.8
            )  # threshold from https://github.com/DIAGNijmegen/pathology-tissue-background-segmentation-processor/blob/007b9b4afa8eaf37aba7c14a27386eb94ad6069c/process.json#L12
            mask, _, _ = filter_regions(mask.astype(np.uint8), diagonal_threshold=100, full_connectivity=True)
            mask = ndimage.binary_dilation(mask, iterations=3)
            mask, _, _ = fill_holes(
                mask.astype(np.uint8),
                diagonal_threshold=250,
                full_connectivity=True,
                fill_value=1,
            )

        else:
            raise NotImplementedError
        return mask

    def _get_mpp(self, slide: openslide.OpenSlide, slide_id: str) -> float:
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
