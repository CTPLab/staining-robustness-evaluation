"""
    Infere simulated MSI classification models under controlled staining variations, and evaluate their performance.
    Runs inference for all n=300 models at once and saves result to csvs.

    Prerequistites:
    1) Download trained AMBIL models + simulation configurations from our HuggingFace repo: https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/MSI_classification_models
    2) Extract foundation model features for all slides using extract_features.py

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

import argparse
import logging
import os
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from models.abmil import ClassificationAttentionNet
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from utils.gpu_monitor import GPUStatsLogger
from utils.load_encoder import INPUT_FEATURE_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def bootstrap_metrics_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    th: float = 0.5,
) -> Dict[str, Tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    metrics = {"auc": [], "precision": [], "recall": [], "f1": []}
    n = len(labels)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        metrics["auc"].append(roc_auc_score(labels[idx], preds[idx]))
        metrics["precision"].append(precision_score(labels[idx], preds[idx] > th))
        metrics["recall"].append(recall_score(labels[idx], preds[idx] > th))
        metrics["f1"].append(f1_score(labels[idx], preds[idx] > th))

    alpha = (1 - ci) / 2
    result = {}
    for k, v in metrics.items():
        v_sorted = np.sort(v)
        lower = np.percentile(v_sorted, 100 * alpha)
        mean = np.mean(v_sorted)
        upper = np.percentile(v_sorted, 100 * (1 - alpha))
        result[k] = (mean, lower, upper)
    return result


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> List[torch.Tensor]:
    embeds = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [embeds, label]


def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {n_trainable_params} parameters")


def compute_cm_metrics(cm: np.ndarray) -> Tuple[float, float, float]:
    def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(numerator / denominator)

    precision = safe_divide(cm[1, 1], cm[:, 1].sum())
    recall = safe_divide(cm[1, 1], cm[1, :].sum())
    f1 = safe_divide(cm[1, 1], (0.5 * (cm[0, 1] + cm[1, 0]) + cm[1, 1]))
    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, default="../SURGEN.csv")
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Directory containing extracted features. Generate with extract_features.py",
    )
    parser.add_argument(
        "--sim_settings_csv",
        type=str,
        default="../MSI_classification_models/fixed_simulation_hps_n=300.csv",
    )
    parser.add_argument(
        "--sim_pretrained_dir",
        type=str,
        default="../MSI_classification_models/trained_models",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--task", type=str, default="MSI")

    ci = 0.95  # 95% confidence interval for bootstrapping
    args = parser.parse_args()
    cohort = Path(args.csv).stem
    logging.info("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    df = pd.read_csv(args.csv)
    if "intensity" in args.features_dir:
        intensity, stain = (lambda s: re.search(r"intensity=(.*?)_stain=(.*)", s).groups())(
            Path(args.features_dir).stem
        )
    else:
        intensity, stain = None, None

    # Ignore nan entries for the specific task
    df = df[~df[args.task].isna()].reset_index(drop=True)
    # Only consider single slide per patient
    df = df[df["primary_patient_slide"]].reset_index(drop=True)
    # Only consider qc passed
    df = df[~df["qc_excluded"]].reset_index(drop=True)

    n_classes = 2
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda" and device.index == args.gpu_id, f"Device is not cuda:{args.gpu_id}"
    sim_settings = pd.read_csv(args.sim_settings_csv)
    log_csv_path = f"{args.output_dir}/gpu_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    gpu_logger = GPUStatsLogger(csv_path=log_csv_path, poll_interval=0.1, gpu_ids=[0])

    try:
        gpu_logger.start()
        logging.info(f"Using features from setting: intensity={intensity}, stain={stain}")
        setting_name = "reference" if stain is None and intensity is None else f"intensity={intensity}_stain={stain}"
        test_results = []
        for idx, hp in sim_settings.iterrows():
            gpu_logger.start_section(f"inference_setting={idx}")
            model_path = glob(f"{args.sim_pretrained_dir}/*exp_idx={idx}-*.pt")
            assert len(model_path) == 1, f"More than one pretrained model match found: {model_path}"
            pretrained_path = model_path[0]
            logging.info(f"Pretrained model path: {pretrained_path}")

            output_dir = f"{args.output_dir}/{setting_name}"
            os.makedirs(output_dir, exist_ok=True)
            logging.info("######################################################################")
            logging.info(f"=> Output dir: {output_dir}")

            # Load model
            aggregator = ClassificationAttentionNet(
                input_feature_size=INPUT_FEATURE_SIZE[hp["foundation_model"]],
                precompression_layer=hp["precompression_layer"],
                feature_size_comp=hp["feature_size_comp"],
                feature_size_attn=hp["feature_size_attn"],
                feature_size_comp_post=hp["feature_size_comp_post"],
                dropout=True,
                p_dropout_fc=hp["p_dropout_fc"],
                p_dropout_atn=hp["p_dropout_atn"],
                n_classes=n_classes,
            )

            aggregator.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))
            aggregator.eval()
            aggregator.to(device)

            loss_fn_eval = nn.CrossEntropyLoss()
            eval_loss = 0
            labels, probs, logits = [], [], []
            fm_feature_dir = glob(f"{args.features_dir}/{hp['foundation_model']}_*")
            assert (
                len(fm_feature_dir) == 1
            ), f"Expected exactly one feature directory for foundation model {hp['foundation_model']}, but found: {fm_feature_dir}"
            fm_feature_dir = fm_feature_dir[0]
            for i, row in df.iterrows():
                slide_id = row["slide_id"]
                features = torch.tensor(
                    np.load(f"{fm_feature_dir}/{slide_id}.npz")["embeds"],
                    dtype=torch.float32,
                ).to(device)
                label = torch.tensor([row[args.task]], dtype=torch.long).to(device)
                with torch.no_grad():
                    logit, Y_prob, Y_hat, A_raw, m = aggregator(features.to(device))
                    loss = loss_fn_eval(logit, label)
                    logit = logit[:, 1].squeeze(0).cpu().item()
                    Y_prob = Y_prob[:, 1].squeeze(0).cpu().item()
                    label = label.cpu().item()
                    labels.append(label)
                    probs.append(Y_prob)
                    logits.append(logit)
                    eval_loss += loss.item()

            torch.cuda.empty_cache()

            eval_loss /= len(df)
            labels, probs, logits = np.array(labels), np.array(probs), np.array(logits)
            auc = roc_auc_score(labels, probs)
            logging.info(f"########### Exp no. {idx} ####################")
            logging.info(f"Eval loss: {eval_loss}, AUC: {auc:.4f}")
            metrics_ci = bootstrap_metrics_ci(labels, probs, ci=ci)
            for metric, (mean, lower, upper) in metrics_ci.items():
                logging.info(f"{metric}: mean={mean:.4f}, {ci*100:.1f}% CI=({lower:.4f}-{upper:.4f})")

            preds_df = pd.DataFrame(probs, columns=["prob"])
            preds_df["logit"] = logits
            preds_df[args.task] = df[args.task].tolist()
            preds_df["slide_id"] = df["slide_id"].tolist()

            test_results.append(
                {
                    "exp_idx": idx,
                    "test_mean_auc": metrics_ci["auc"][0],
                    "test_lower_ci": metrics_ci["auc"][1],
                    "test_upper_ci": metrics_ci["auc"][2],
                    "test_auc": auc,
                    "test_loss": eval_loss,
                    **hp,
                }
            )
            # Save into a csv file.
            logging.info(f"Saving predictions...")
            preds_df.to_csv(f"{output_dir}/predictions_exp_id={idx}_{cohort}_N{len(df)}.csv")
            gpu_logger.stop_section(f"inference_setting={idx}")

        test_df = pd.DataFrame(test_results)
        test_df.to_csv(f"{args.output_dir}/test_results_{setting_name}.csv", index=False)

        gpu_logger.stop_and_log_total()
    finally:
        gpu_logger.stop_and_log_interrupted()
