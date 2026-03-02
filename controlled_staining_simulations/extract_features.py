"""
    Extract features from augmented slides with controlled stain variations for robustness evaluation

    The script performs the following steps:
    * read pre-computed tile coordinates OR run tissue detection + tiling for each slide
    * apply controlled stain augmentations to the tiles based on specified target stain and intensity settings
    * extract features from the augmented tiles using specified foundation models
    * save to npz files

    Please pre-download:
    * PLISM reference stain library: https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/plism-wsi_stain_references
    * SurGen pre-extracted stain properties: https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/surgen_stain_properties
    * Any foundation model you want to use: Univ2, HOptimus1, Virchow2, CTransPath, RetCCL,... 

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"

"""

import argparse
import os
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from utils.gpu_monitor import GPUStatsLogger
from utils.load_encoder import ENCODER_LOADERS, ENCODER_NORMALIZATIONS, INPUT_FEATURE_SIZE
from utils.tile_utils import create_tissue_tiles, extract_tile_from_slide
from utils.tissue_detector import TissueDetector


def collate_features(
    batch: List[Tuple[torch.Tensor, torch.Tensor, np.ndarray, Tuple[float, float, float, float]]]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    imgs, aug_imgs, orig_imgs, coords = zip(*batch)
    img = torch.stack(imgs, dim=0)
    aug_img = torch.stack(aug_imgs, dim=0).permute(0, 3, 1, 2)
    orig_imgs = np.stack(orig_imgs, axis=0)
    coords = torch.from_numpy(np.array(coords, dtype=np.float32))
    return img, aug_img, orig_imgs, coords


def crop_rect_from_slide(
    slide: openslide.OpenSlide,
    rect: np.ndarray,
    out_size: int,
    use_lower_level: bool = False,
) -> np.ndarray:
    minx, miny, maxx, maxy = rect
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


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> List[torch.Tensor]:
    embeds = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [embeds, label]


def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")


def augment_tile(
    image: np.ndarray,
    stain_source: np.ndarray,
    stain_target: np.ndarray,
    trgt_intensity: Tuple[float, float] = None,
    src_intensity: Tuple[float, float] = None,
    residual_reduction_f: float = 0.01,
) -> np.ndarray:
    maxIntensityVect = np.amax(image, axis=(0, 1))
    # avoid zero division
    if np.all(maxIntensityVect == 0):
        return image
    else:
        # replace all zero with 1 to avoid zero division
        maxIntensityVect = np.maximum(maxIntensityVect, 1)
    stainMatrixInv = np.linalg.inv(stain_source)
    image_norm = np.clip(image.astype(np.float32) / maxIntensityVect, 1e-6, 1.0)  # range (eps,1]
    image_OD = -np.log(image_norm)
    image_proj = np.matmul(image_OD, stainMatrixInv.T)
    if (
        src_intensity is not None
        and trgt_intensity is not None
        and (np.array(trgt_intensity) / np.array(src_intensity) < 1).any()
    ):
        failure_mask = image_proj[..., 0] > np.percentile(image_proj[..., 0].ravel(), 90)
        image_proj[failure_mask] = np.maximum(image_proj[failure_mask], 0)

    if trgt_intensity is not None:
        for idx, (src_c, trgt_c) in enumerate(zip(src_intensity, trgt_intensity)):
            image_proj[..., idx] = (trgt_c / src_c) * image_proj[..., idx]
    mask = image_proj[..., 2] > 0.1
    image_proj[mask, 2] = image_proj[mask, 2] * residual_reduction_f

    stain_target[:, 2] = stain_source[:, 2]
    image_backward = np.matmul(image_proj, stain_target.T)
    image_rec = maxIntensityVect * np.exp(-1 * image_backward)
    image_rec = np.maximum(0, image_rec)
    image_rec = np.minimum(255, image_rec)
    image_rec = image_rec.astype(np.uint8)
    return image_rec


class BagOfTiles(Dataset):
    def __init__(
        self,
        wsi: openslide.OpenSlide,
        coords: List[Tuple[int, int, int, int]],
        resize_to: int,
        stain_matrix: np.ndarray,
        target_stain_matrix: np.ndarray,
        trgt_intensity: Tuple[float, float] = None,
        src_intensity: Tuple[float, float] = None,
        use_lower_level: bool = True,
    ) -> None:
        self.wsi = wsi
        self.coords = coords
        self._out_size = resize_to
        self._use_lower_level = use_lower_level
        self._stain_matrix = stain_matrix
        self._target_stain_matrix = target_stain_matrix
        self._trgt_intensity = trgt_intensity
        self._src_intensity = src_intensity
        print(f"User lower slide levels if available: {use_lower_level}")
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float, float, float]]:
        coord = self.coords[idx]
        img = crop_rect_from_slide(self.wsi, coord, self._out_size, self._use_lower_level)
        augmented_img = augment_tile(
            np.array(img),
            self._stain_matrix,
            self._target_stain_matrix,
            self._trgt_intensity,
            self._src_intensity,
        )
        norm_img = self.transform(augmented_img)
        return (
            norm_img,
            torch.tensor(augmented_img, dtype=torch.uint8),
            np.array(img),
            coord,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--foundation_models",
        type=lambda s: list(map(str, s.split(","))),
        required=True,
    )
    parser.add_argument("--stain_dir", type=str, default="../surgen_stain_properties/stain_vectors")
    parser.add_argument(
        "--ref_stain_dir",
        type=str,
        default="../plism-wsi_stain_references/stain_vectors",
    )
    parser.add_argument("--um_size", type=int, default=224)
    parser.add_argument("--px_size", type=int, default=224)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to output visualization of augmented slides and tiles",
    )
    parser.add_argument(
        "--target_stain_name",
        type=str,
        default=None,
        help="Name of target stain condition",
    )
    parser.add_argument(
        "--target_intensity",
        type=str,
        default=None,
        help="Name of target intensity condition",
    )
    min_height, min_width = 1000, 1000  # for visualization
    args = parser.parse_args()
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    um_size, px_size = args.um_size, args.px_size
    foundation_models = args.foundation_models
    target_stain_name = None if args.target_stain_name == "None" else args.target_stain_name
    target_intensity = None if args.target_intensity == "None" else args.target_intensity

    df = pd.read_csv(args.csv)
    # Ignore nan entries for the specific task
    df = df[~df[args.task].isna()].reset_index(drop=True)
    # Only consider single slide per patient
    df = df[df["primary_patient_slide"]].reset_index(drop=True)
    # Only consider qc passed
    df = df[~df["qc_excluded"]].reset_index(drop=True)

    # Settings for stain augmentation and intensity
    if target_intensity is not None:
        dataset_intensities = pd.read_csv(f"{Path(args.stain_dir).parent}/intensity/intensity_stats_tiles.csv")
        dataset_intensities = dataset_intensities.groupby("slide_id")[["H_95th", "E_95th"]].median()
        target_intensity: Tuple[str, str] = tuple(target_intensity.split("_"))
        target_intensities = pd.read_csv(
            f"{Path(args.ref_stain_dir).parent}/intensities/median_intensities_per_condition.csv"
        )
        trg_c_values = target_intensities.loc[
            (target_intensities["stain"] == target_intensity[0])
            & (target_intensities["device"] == target_intensity[1]),
            ["H_95th", "E_95th"],
        ].values[0]

        print(f"Using intensity setting={target_intensity}: H {trg_c_values[0]}, E {trg_c_values[1]}")
    else:
        print(f"No intensity settings provided, not conducting intensity variation")

    if target_stain_name is not None:
        stain_npzs = sorted(glob(f"{args.ref_stain_dir}/*.npz"))
        target_stains = [
            (
                Path(stain_npz).stem.replace("_median", ""),
                np.load(stain_npz)["stainMatrix"],
            )
            for stain_npz in stain_npzs
        ]
        target_stain = [t for t in target_stains if t[0] == target_stain_name][0][1]
        print(f"Using target stain {target_stain_name} for augmentation")
    else:
        target_stain = None
        print(f"No target stain specified, not using stain augmentation")
    output_dir = f"{args.output_dir}/intensity={args.target_intensity}_stain={target_stain_name}"
    if args.visualize:
        os.makedirs(f"{output_dir}/augmented_slides", exist_ok=True)
    print("######################################################################")
    print(f"=> Output dir: {output_dir}")
    for fm in foundation_models:
        os.makedirs(f"{output_dir}/{fm}_features_{um_size}um_{px_size}px_fcnn", exist_ok=True)
    log_csv_path = f"{output_dir}/gpu_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    gpu_logger = GPUStatsLogger(csv_path=log_csv_path, poll_interval=0.1, gpu_ids=[args.gpu_id])
    tissue_detector = TissueDetector(tissue_method="fcnn", device=device)

    try:
        gpu_logger.start()
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        assert device.type == "cuda" and device.index == args.gpu_id, f"Device is not cuda:{args.gpu_id}"
        encoders, aggregators, means, stds = {}, {}, {}, {}
        for fm in foundation_models:
            means[fm] = torch.tensor(ENCODER_NORMALIZATIONS[fm]["mean"], device=device, dtype=torch.float32).view(
                1, -1, 1, 1
            )
            stds[fm] = torch.tensor(ENCODER_NORMALIZATIONS[fm]["std"], device=device, dtype=torch.float32).view(
                1, -1, 1, 1
            )
            encoders[fm] = ENCODER_LOADERS[fm]()
            encoders[fm].to(device)
            encoders[fm].eval()

        for i, row in tqdm(df.iterrows(), total=len(df)):
            slide_id = row["slide_id"]
            # check if already processed for all fms
            if all(
                os.path.exists(f"{output_dir}/{fm}_features_{um_size}um_{px_size}px_fcnn/{slide_id}.npz")
                for fm in foundation_models
            ):
                print(f"Skipping already processed slide {slide_id}")
                continue

            if args.visualize:
                os.makedirs(f"{output_dir}/augmented_tiles/{slide_id}", exist_ok=True)
            print(f"Processing slide {slide_id}")
            wsi = openslide.open_slide(row["slide_path"])

            # Load stain matrix and intensity setting
            stain_matrix = np.load(f"{args.stain_dir}/{slide_id}.npz")["stainMatrix"]
            try:
                orig_mpp = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
            except KeyError:
                raise Exception(
                    f"Could not find the {openslide.PROPERTY_NAME_MPP_X} in the slide {slide_id}, all keys: {wsi.properties.keys()}"
                )

            print("Detect tissue in slide")
            mask = tissue_detector.detect_tissue(wsi, slide_id)
            print(f"Tile slide into {um_size}um X {um_size}um tiles")
            coords = create_tissue_tiles(
                wsi,
                mask,
                tile_size_microns=um_size,
                slide_id=slide_id,
                offsets_micron=0,
            )
            coords = np.array([c.bounds for c in coords])
            coord_tile_size_px = int(coords[0, 2] - coords[0, 0])
            print(
                f"Generated tile coords {coords.shape} tile size={coord_tile_size_px}px, {coord_tile_size_px*orig_mpp:.1f}um"
            )

            # Output visualization
            if args.visualize:
                num_vis_tiles = 10
                tile_size_pix = round(um_size / orig_mpp)
                assert (
                    tile_size_pix == coord_tile_size_px
                ), f"tile_size_pix={tile_size_pix}, actual tile size={coord_tile_size_px}"
                tissue_margin_pix = tile_size_pix * 2
                roi_minx, roi_miny, roi_maxx, roi_maxy = (
                    np.min(coords[:, :2], axis=0).tolist() + np.max(coords[:, 2:], axis=0).tolist()
                )
                roi_minx, roi_miny, roi_maxx, roi_maxy = (
                    roi_minx - tissue_margin_pix,
                    roi_miny - tissue_margin_pix,
                    roi_maxx + tissue_margin_pix,
                    roi_maxy + tissue_margin_pix,
                )
                roi_h, roi_w = roi_maxy - roi_miny, roi_maxx - roi_minx
                scale_factor = min_width / min(roi_w, roi_h)
                # print(roi_minx, roi_miny, roi_maxx, roi_maxy, roi_h, roi_w, scale_factor)
                overview_w, overview_h = round(roi_w * scale_factor), round(roi_h * scale_factor)
                overview = np.zeros((overview_h, overview_w, 3), dtype=np.uint8)
                vis_tile_size = round(coord_tile_size_px * scale_factor)

            # Data loading pipeline
            dataset = BagOfTiles(
                wsi,
                coords,
                resize_to=px_size,
                stain_matrix=stain_matrix,
                target_stain_matrix=(target_stain if target_stain is not None else stain_matrix),
                trgt_intensity=((trg_c_values[0], trg_c_values[1]) if target_intensity is not None else None),
                src_intensity=(tuple(dataset_intensities.loc[slide_id]) if target_intensity is not None else None),
                use_lower_level=True,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                collate_fn=collate_features,
                num_workers=2,
                pin_memory=False,
            )

            all_features = {}
            for foundation_model in foundation_models:
                all_features[foundation_model] = torch.empty(
                    (len(dataset), INPUT_FEATURE_SIZE[foundation_model]),
                    dtype=torch.float32,
                ).to("cuda")
            if args.visualize:
                num_vis_tiles_batch = max(1, round(num_vis_tiles / len(loader)))
            with torch.no_grad():
                for idx, (batch, augmented, orig_imgs, b_coords) in tqdm(enumerate(loader), total=len(loader)):
                    batch = batch.to("cuda", non_blocking=True)
                    for fm in all_features.keys():
                        norm_batch = (batch - means[fm]) / stds[fm]
                        features = encoders[fm](norm_batch)
                        all_features[fm][
                            idx * args.batch_size : min((idx + 1) * args.batch_size, len(dataset))
                        ] = features

                    if args.visualize:
                        # Visualization
                        b_coords = b_coords - torch.tensor([roi_minx, roi_miny, roi_minx, roi_miny])
                        scaled_coords = torch.round(b_coords * scale_factor).to(torch.int64)
                        tile_spans = scaled_coords[:, 2:] - scaled_coords[:, :2]  # (N, 2): [w, h]
                        max_tile_h = tile_spans[:, 1].max().item()
                        max_tile_w = tile_spans[:, 0].max().item()
                        aug_resized = (
                            F.interpolate(
                                augmented.float(),
                                size=(max_tile_h, max_tile_w),
                                mode="bilinear",
                                align_corners=False,
                            )
                            .byte()
                            .numpy()
                        )

                        for (x_min, y_min, x_max, y_max), tile in zip(scaled_coords, aug_resized):
                            tile = tile[:, : (y_max - y_min), : (x_max - x_min)]  # crop dynamically
                            overview[y_min:y_max, x_min:x_max, :] = tile.transpose(1, 2, 0)

                        vis_idxs = np.random.choice(len(batch), num_vis_tiles_batch)
                        for vis_idx, coord, tile, orig_img in zip(
                            vis_idxs,
                            b_coords[vis_idxs],
                            augmented[vis_idxs],
                            orig_imgs[vis_idxs],
                        ):
                            minx, miny, maxx, maxy = coord
                            Image.fromarray(orig_img).save(
                                f"{output_dir}/augmented_tiles/{slide_id}/orig_minx={minx}_miny={miny}_maxx={maxx}_maxy={maxy}.png"
                            )
                            Image.fromarray(tile.permute(1, 2, 0).numpy()).save(
                                f"{output_dir}/augmented_tiles/{slide_id}/aug_minx={minx}_miny={miny}_maxx={maxx}_maxy={maxy}.png"
                            )

                if args.visualize:
                    Image.fromarray(overview).save(f"{output_dir}/augmented_slides/{slide_id}.png")
                print(
                    f"Extracted features for {all_features[fm].shape[0]} tiles of slide {slide_id}, stained to {target_stain if target_stain is not None else stain_matrix}"
                )
                for fm in foundation_models:
                    np.savez(
                        f"{output_dir}/{fm}_features_{um_size}um_{px_size}px_fcnn/{slide_id}.npz",
                        embeds=all_features[fm].cpu().numpy(),
                        coords=coords,
                    )

                # Store results
            wsi.close()
            del all_features

        print("Finished feature extraction.")
        gpu_logger.stop_and_log_total()
    finally:
        gpu_logger.stop_and_log_interrupted()
