"""
    Infer publicly available MSI classification models under controlled staining variations, and evaluate their performance.
    
    Script can be utilized for any other task, dataset and model, with the following limitations:
    - Currently only supports binary classification, expected model output is single logit (Batch_size,).
    - To utilize your own models, please add your own model loading logic.

    Prerequistites:
    1) Extract features for all slides using extract_features.py
    2) Download pretrained models (e.g. Wagner2023, Niehues2023)

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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from models.niehues2023 import Niehues2023
from models.wagner2023 import Wagner2023
from utils.gpu_monitor import GPUStatsLogger
from utils.train_eval_utils import bootstrap_metrics_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="SURGEN.csv")
    parser.add_argument(
        "--features_dir",
        "--feature_dir",
        dest="features_dir",
        type=str,
        required=True,
        help="High-level directory containing extracted features. Generate with extract_features.py",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--task", type=str, default="MSI")

    args = parser.parse_args()
    ci = 0.95  # 95% confidence interval for bootstrapping
    model_name = f"agg={Path(args.pretrained_model).stem}"
    assert Path(
        args.csv
    ).exists(), f"CSV file {args.csv} does not exist, please verify the path or ensure you are in the staining-robustness-evaluation directory when wanting to use SURGEN.csv."

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
    log_csv_path = f"{args.output_dir}/gpu_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    gpu_logger = GPUStatsLogger(csv_path=log_csv_path, poll_interval=0.1, gpu_ids=[args.gpu_id])

    try:
        gpu_logger.start()
        logging.info(f"Using features from setting: intensity={intensity}, stain={stain}")
        setting_name = "reference" if stain is None and intensity is None else f"intensity={intensity}_stain={stain}"
        gpu_logger.start_section(f"inference_setting={model_name}")
        pretrained_path = args.pretrained_model
        logging.info(f"Pretrained model path: {pretrained_path}")

        output_dir = f"{args.output_dir}/{setting_name}"
        os.makedirs(output_dir, exist_ok=True)
        logging.info("######################################################################")
        logging.info(f"=> Output dir: {output_dir}")

        # Load model
        if "wagner2023" in pretrained_path.lower():
            foundation_model = "ctranspath"
            model_family = "wagner2023"
            aggregator = Wagner2023(pretrained_path=pretrained_path)
        elif "niehues2023" in pretrained_path.lower():
            foundation_model = "retccl"
            model_family = "niehues2023"
            aggregator = Niehues2023(pretrained_path=pretrained_path)
        ## FIXME: Add your own model loading logic here
        else:
            raise ValueError(f"Unknown model name in pretrained path: {pretrained_path}")
        loss_fn_eval = nn.BCEWithLogitsLoss()

        aggregator.eval()
        aggregator.to(device)

        eval_loss = 0
        labels, probs, logits = [], [], []
        fm_feature_dir = glob(f"{args.features_dir}/{foundation_model}_*")
        assert (
            len(fm_feature_dir) == 1
        ), f"Expected exactly one feature directory for foundation model {foundation_model}, but found: {fm_feature_dir}"
        fm_feature_dir = fm_feature_dir[0]
        for i, row in df.iterrows():
            slide_id = row["slide_id"]
            features = (
                torch.tensor(
                    np.load(f"{fm_feature_dir}/{slide_id}.npz")["embeds"],
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .to(device)
            )
            label = int(row[args.task])
            with torch.no_grad():
                logit = aggregator(features)
                target = torch.tensor([label], dtype=torch.float32, device=device)
                loss = loss_fn_eval(logit, target)

                prob = torch.sigmoid(logit)

                labels.append(label)
                probs.append(prob.item())
                logits.append(logit.item())
                eval_loss += loss.item()

        torch.cuda.empty_cache()

        eval_loss /= len(df)
        labels, probs, logits = np.array(labels), np.array(probs), np.array(logits)
        auc = roc_auc_score(labels, probs)
        logging.info(f"########### Exp agg={model_name} ####################")
        logging.info(f"Eval loss: {eval_loss}, AUC: {auc:.4f}")
        metrics_ci = bootstrap_metrics_ci(labels, probs, ci=ci)
        for metric, (mean, lower, upper) in metrics_ci.items():
            logging.info(f"{metric}: mean={mean:.4f}, {ci*100:.1f}% CI=({lower:.4f}-{upper:.4f})")

        preds_df = pd.DataFrame(probs, columns=["prob"])
        preds_df["logit"] = logits
        preds_df[args.task] = df[args.task].tolist()
        preds_df["slide_id"] = df["slide_id"].tolist()

        test_results = {
            "exp_idx": model_name,
            "test_mean_auc": metrics_ci["auc"][0],
            "test_lower_ci": metrics_ci["auc"][1],
            "test_upper_ci": metrics_ci["auc"][2],
            "test_auc": auc,
            "test_loss": eval_loss,
        }

        # Save into a csv file.
        logging.info(f"Saving predictions...")
        preds_df.to_csv(f"{output_dir}/predictions_agg={model_name}_{cohort}_N{len(df)}.csv")
        gpu_logger.stop_section(f"inference_setting={model_name}")

        if Path(f"{args.output_dir}/test_results_{setting_name}.csv").exists():
            test_df = pd.read_csv(f"{args.output_dir}/test_results_{setting_name}.csv")
        else:
            test_df = pd.DataFrame()
        test_df = pd.concat([test_df, pd.DataFrame([test_results])], ignore_index=True)
        test_df.to_csv(f"{args.output_dir}/test_results_{setting_name}.csv", index=False)

        gpu_logger.stop_and_log_total()
    finally:
        gpu_logger.stop_and_log_interrupted()
