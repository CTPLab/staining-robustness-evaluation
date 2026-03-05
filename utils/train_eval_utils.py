"""
    Utility functions for training and evaluation of MSI classification models:
    * Confusion matrix visualization
    * Reproducibility settings
    * Performance metric calculations

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def create_confusion_matrix_figure(confusion_matrix_data: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Creates a matplotlib figure with a confusion matrix"""
    confusion_matrix_rows = [[round(float(value), 3) for value in row] for row in confusion_matrix_data]
    ax = sns.heatmap(
        confusion_matrix_rows,
        annot=True,
        fmt="g",
        xticklabels=["WT", "MUT"],
        yticklabels=["WT", "MUT"],
        square=True,
        cbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    return ax.get_figure()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_model(model):
    logging.info(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {n_trainable_params} parameters")


def print_cuda_memory(rank: int):
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    print(f"[Rank {rank}] -- Memory Allocated: {allocated / (1024 ** 3):.2f} GB")
    print(f"[Rank {rank}] -- Memory Cached: {cached / (1024 ** 3):.2f} GB")
    print(f"[Rank {rank}] -- Max Memory Allocated: {max_allocated / (1024 ** 3):.2f} GB")


def get_sampling_weights(df: pd.DataFrame, task: str) -> List[float]:
    df = df[df["primary_patient_slide"]].reset_index(drop=True)
    assert (
        not df["patient_id"].duplicated().any()
    ), f"{task} column contains multiple True primary_patient_slide per patient!"
    assert df[task].notna().all(), f"{task} column contains nan values!"
    inverse_weight_mapping = dict(len(df) / df[task].value_counts())
    logging.info(f"Initializing weighted random sampler with the following class weights: {inverse_weight_mapping}.")
    weights = df[task].replace(inverse_weight_mapping).tolist()
    return weights


def compute_cm_metrics(
    cm: np.ndarray, return_specificity: bool = False
) -> Union[Tuple[float, float, float], Tuple[float, float, float, float]]:
    def safe_divide(numerator: float, denominator: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(numerator / denominator) if denominator != 0 else 0.0

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(tp, tp + 0.5 * (fp + fn))

    if return_specificity:
        specificity = safe_divide(tn, tn + fp)
        return precision, recall, f1, specificity
    else:
        return precision, recall, f1


def bootstrap_metrics_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    th: float = 0.5,
) -> Dict[str, Tuple[float, float, float]]:
    """Computes bootstrap mean and confidence intervals for performance metrics"""

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
