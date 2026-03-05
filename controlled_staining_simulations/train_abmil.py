"""
    Train n=300 ABMIL models based on sampled hyperparams.

    * Runs all 300 trainings with parallel processes
    * Please download hyperparameter configurations and train/val splits from Hugging Face: 
    https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/abmil_simulation_hyperparams

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

import copy
import logging
import multiprocessing as mp
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models.abmil import ClassificationAttentionNet
from utils.gpu_monitor import GPUStatsLogger
from utils.load_encoder import INPUT_FEATURE_SIZE
from utils.train_utils import (
    compute_cm_metrics,
    create_confusion_matrix_figure,
    get_sampling_weights,
    print_model,
    seed_worker,
    set_seed,
)


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> List[torch.Tensor]:
    embeds = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [embeds, label]


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(Path(log_file).stem)  # unique logger name per file
    logger.setLevel(logging.INFO)

    # Remove existing handlers on this logger
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))

    logger.addHandler(fh)

    return logger


class MILDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        label_col: str,
        random_slide_selection: bool = False,
    ) -> None:
        self.df = df
        self.patients = list(df["patient_id"].unique())
        self.data_dir = Path(data_dir)
        self.label_col = label_col
        self.random_slide_selection = random_slide_selection

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_id = self.patients[idx]
        if self.random_slide_selection:
            rows = self.df[self.df["patient_id"] == patient_id]
            # Randomly choose a slide
            i = random.randint(0, len(rows) - 1)
            row = rows.iloc[i]
        else:
            rows = self.df[(self.df["patient_id"] == patient_id) & self.df["primary_patient_slide"]]
            assert (
                len(rows) == 1
            ), f"Multiple slides available per patient, expected only primary slide, patient_id={patient_id}, slide_ids={rows['slide_id'].tolist()}"
            row = rows.iloc[0]
        feats = np.load(f"{self.data_dir}/{row['slide_id']}.npz")["embeds"]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(row[self.label_col], dtype=torch.int64)


class MILTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        early_stopping_patience: int,
        grad_accum_steps: int,
        output_dir: Path,
        logger: logging.Logger,
        n_classes: int = 2,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.patience = early_stopping_patience
        self.grad_accum_steps = grad_accum_steps
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.logger = logger

    def train(self, epochs: int, gpu_logger: GPUStatsLogger) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
        best_metrics = {}
        best_model = None
        best_val_auroc = float("-inf")  # best_val_loss = float("inf")
        patience_counter = 0
        self.optimizer.zero_grad()
        self.model.train()

        for epoch in range(epochs):
            gpu_logger.start_section(f"train_epoch={epoch}")
            total_loss = 0.0
            train_cm = np.zeros((self.n_classes, self.n_classes))

            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                y_logits, y_prob, y_hat, _, _ = self.model(x)
                loss = self.criterion(y_logits, y)
                (loss / self.grad_accum_steps).backward()

                if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Collect metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                train_cm += confusion_matrix(
                    y_true=y.detach().cpu().numpy(),
                    y_pred=y_hat.detach().cpu().numpy(),
                    labels=list(range(self.n_classes)),
                )

            gpu_logger.stop_section(f"train_epoch={epoch}")
            gpu_logger.start_section(f"val_epoch={epoch}")
            val_loss, val_auc, val_cm, val_df = self.evaluate(epoch)
            gpu_logger.stop_section(f"val_epoch={epoch}")

            # Collect metrics and log to tensorboard
            avg_train_loss = total_loss / len(self.train_loader)
            val_precision, val_recall, val_f1 = compute_cm_metrics(val_cm)
            train_precision, train_recall, train_f1 = compute_cm_metrics(train_cm)
            self.writer.add_figure(
                "Train Confusion Matrix",
                create_confusion_matrix_figure(train_cm),
                epoch,
            )
            self.writer.add_figure("Val Confusion Matrix", create_confusion_matrix_figure(val_cm), epoch)
            self.writer.add_scalars("Precision", {"train": train_precision, "val": val_precision}, epoch)
            self.writer.add_scalars("Recall", {"train": train_recall, "val": val_recall}, epoch)
            self.writer.add_scalars("F1 Score", {"train": train_f1, "val": val_f1}, epoch)
            self.writer.add_scalars("AUROC", {"val": val_auc}, epoch)
            self.writer.add_scalars("Epoch Loss", {"train": avg_train_loss, "val": val_loss}, epoch)
            if self.scheduler is not None:
                self.writer.add_scalar("Learning rate", self.scheduler._get_closed_form_lr()[0], epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            if val_auc > best_val_auroc:
                self.logger.info(
                    f"Epoch {epoch} validation auroc increased ({best_val_auroc:.6f} --> {val_auc:.6f}), new best model at epoch={epoch}, val_loss={val_loss}, val f1={val_f1:.4f}, val precision={val_precision:.4f}, val recall={val_recall:.4f}, val cm={val_cm}, "
                )
                best_val_auroc = val_auc
                best_metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                }
                patience_counter = 0
                best_model = copy.deepcopy(self.model.state_dict())
                best_val_df = val_df
            else:
                patience_counter += 1
                self.logger.info(
                    f"Validation auroc does not increase : Early stopping counter {patience_counter}/{self.patience}"
                )
                if patience_counter >= self.patience:
                    self.logger.info(
                        f"Experiment is stopped early at epoch {epoch}, as val auroc did not improve for {self.patience} epochs!"
                    )
                    break
        return best_model, best_metrics, best_val_df

    def evaluate(self, epoch: int) -> tuple[float, float, np.ndarray, pd.DataFrame]:
        self.model.eval()
        total_loss = 0.0
        val_cm = np.zeros((self.n_classes, self.n_classes))
        preds = np.zeros(len(self.val_loader))
        labels = np.zeros(len(self.val_loader))
        logits = np.zeros(len(self.val_loader))
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.device), y.to(self.device)
                Y_logits, Y_prob, Y_hat, _, _ = self.model(x)
                loss = self.criterion(Y_logits, y)
                total_loss += loss.item()

                # Collect metrics and log to tensorboard
                val_cm += confusion_matrix(
                    y_true=y.detach().cpu().numpy(),
                    y_pred=Y_hat.detach().cpu().numpy(),
                    labels=list(range(self.n_classes)),
                )
                preds[i] = Y_prob[:, 1].squeeze().item()
                logits[i] = Y_logits[:, 1].squeeze().item()
                labels[i] = y.squeeze().item()

        val_auc = roc_auc_score(labels, preds)
        val_df = pd.DataFrame(
            {
                "patient_id": self.val_loader.dataset.patients,
                "pred_prob": preds,
                "logit": logits,
                "gt_label": labels,
            }
        )
        # val_df.to_csv(f"{self.output_dir}/val_auc={val_auc}_epoch={epoch}.csv", index=False)
        return float(total_loss / len(self.val_loader)), float(val_auc), val_cm, val_df

    def save_results(
        self,
        output_dir: str,
        best_model: Dict[str, Any],
        best_metrics: Dict[str, Any],
        best_val_df: pd.DataFrame,
        run_history: Dict[str, Any],
    ) -> None:
        torch.save(
            best_model,
            f"{output_dir}/model_epoch={best_metrics['epoch']}_val_loss={best_metrics['val_loss']:.4f}_val_auc={best_metrics['val_auc']:.4f}.pt",
        )
        self.logger.info(
            f"Saved best model to {output_dir}/model_epoch={best_metrics['epoch']}_val_loss={best_metrics['val_loss']:.4f}_val_auc={best_metrics['val_auc']:.4f}.pt"
        )

        df = pd.DataFrame([run_history])
        csv_path = os.path.join(os.path.dirname(output_dir), f'runs_history_{run_history["task"]}.csv')
        df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
        best_val_df.to_csv(
            f"{output_dir}/model_epoch={best_metrics['epoch']}_val_loss={best_metrics['val_loss']:.4f}_val_auc={best_metrics['val_auc']:.4f}.csv",
            index=False,
        )
        self.logger.info(f"Saved experiment results to {csv_path}.")


def run_experiment(
    exp_idx: int,
    task: str,
    um_size: str,
    px_size: str,
    hp: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_dir: str,
    output_dir: str,
) -> None:
    set_seed(hp["seed"])  # crucial to set seed here!
    git_sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_run_id = f"{git_sha}-exp_idx={exp_idx}-{hp['foundation_model']}-{timestamp}"
    output_dir = f"{output_dir}/{train_run_id}"
    os.makedirs(output_dir, exist_ok=True)
    log_csv_path = f"{output_dir}/gpu_logs_{timestamp}.csv"
    log_file: str = f"{output_dir}/train_{timestamp}.log"
    logger = setup_logger(log_file)

    logger.info("######################################################################")
    logger.info(f"=> Experiment idx {exp_idx}")
    logger.info(f"=> Git SHA {train_run_id}")
    logger.info(f"=> Output dir {output_dir}")
    logger.info(f"=> Data dir {data_dir}")
    gpu_logger = GPUStatsLogger(csv_path=log_csv_path, poll_interval=0.1, gpu_ids=[0])

    try:
        gpu_logger.start()
        n_classes = train_df[task].nunique()
        logger.info("Hyperparameters:")
        for k, v in hp.items():
            logger.info(f"  {k}: {v}")
        writer = SummaryWriter(log_dir=f"{output_dir}/tf_logs")
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        assert device.type == "cuda" and device.index == 0, f"Device is not cuda:0"
        input_feature_size = (
            INPUT_FEATURE_SIZE[hp["foundation_model"]]
            if not type(INPUT_FEATURE_SIZE[hp["foundation_model"]]) == dict
            else INPUT_FEATURE_SIZE[hp["foundation_model"]][output_dir.split("/")[5]]
        )

        model = ClassificationAttentionNet(
            input_feature_size=input_feature_size,
            precompression_layer=hp["precompression_layer"],
            feature_size_comp=hp["feature_size_comp"],
            feature_size_attn=hp["feature_size_attn"],
            feature_size_comp_post=hp["feature_size_comp_post"],
            dropout=True,
            p_dropout_fc=hp["p_dropout_fc"],
            p_dropout_atn=hp["p_dropout_atn"],
            n_classes=n_classes,
        )
        model.to(device)
        logger.info("MIL model created")
        print_model(model)

        train_ds = MILDataset(train_df, data_dir, label_col=task, random_slide_selection=True)
        val_ds = MILDataset(val_df, data_dir, label_col=task)
        # Reproducibility of DataLoader
        g = torch.Generator()
        g.manual_seed(hp["seed"])
        train_loader = DataLoader(
            train_ds,
            batch_size=hp["batch_size"],
            sampler=WeightedRandomSampler(
                weights=get_sampling_weights(train_df, task),
                num_samples=len(train_ds),
                replacement=True,
            ),
            num_workers=0,
            worker_init_fn=seed_worker,
            pin_memory=True,
            collate_fn=collate_fn,
            generator=g,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        logger.info(f"=> Training on {len(train_loader)} patients")
        logger.info(f"=> Validating on {len(val_loader)} patients")

        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        if hp["scheduler"] == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["max_epochs"])
        elif hp["scheduler"] is None:
            scheduler = None
        else:
            raise NotImplementedError(f"Specified scheduler {hp['scheduler']} has not been implemented!")

        trainer = MILTrainer(
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=nn.CrossEntropyLoss(),
            train_loader=train_loader,
            val_loader=val_loader,
            writer=writer,
            early_stopping_patience=hp["earlystop_patience"],
            grad_accum_steps=1,
            output_dir=Path(output_dir),
            logger=logger,
        )

        best_model, best_metrics, best_val_df = trainer.train(hp["max_epochs"], gpu_logger)
        logger.info("Finished training.")
        trainer.save_results(
            output_dir,
            best_model,
            best_metrics,
            best_val_df,
            {
                "exp_idx": exp_idx,
                "task": task,
                **best_metrics,
                **hp,
                "um_size": um_size,
                "px_size": px_size,
            },
        )
        gpu_logger.stop_and_log_total()
        del model
        torch.cuda.empty_cache()
    finally:
        gpu_logger.stop_and_log_interrupted()


def run_experiment_wrapper(args_tuple):
    return run_experiment(*args_tuple)


if __name__ == "__main__":
    #### USER INPUTS - PLEASE CONFIGURE BEFORE RUNNING ####
    data_dir = ""  # <--- Path to pre-extracted features, organized as {data_dir}/{slide_id}.npz
    output_dir = ""  # <--- Output directory for trained models and results
    num_parallel = 3  # <--- Number of parallel processes, runs 3 trainings at once until all experiments are finished

    #### Download from Hugging Face Repo: https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/abmil_simulation_hyperparams ####
    split_dir = "./abmil_simulation_hyperparams/fixed_splits_n=300"
    simulation_settings = pd.read_csv("./abmil_simulation_hyperparams/fixed_simulation_hps_n=300.csv")
    ##############################################################################
    um_size = "224um"
    px_size = "224px"
    task = "MSI"

    exps = []
    for idx, hp in simulation_settings.iterrows():
        train_df: pd.DataFrame = pd.read_csv(f"{split_dir}/train_{idx}.csv")
        val_df: pd.DataFrame = pd.read_csv(f"{split_dir}/val_{idx}.csv")

        # Ignore nan entries for the specific task
        train_df = train_df[~train_df[task].isna()].reset_index(drop=True)
        val_df = val_df[~val_df[task].isna()]
        # Only consider single slide per patient
        val_df = val_df[val_df["primary_patient_slide"]].reset_index(drop=True)
        exp_data_dir = f"{data_dir}/{hp['foundation_model']}_features_{um_size}_{px_size}_fcnn"
        exps.append(
            (
                idx,
                task,
                um_size,
                px_size,
                hp,
                train_df,
                val_df,
                exp_data_dir,
                output_dir,
            )
        )

    with mp.Pool(processes=num_parallel) as pool:
        pool.map(run_experiment_wrapper, exps)
