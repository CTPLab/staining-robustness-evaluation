"""
    GPUStatsLogger: A utility to estimate GPU energy usage and timing by periodically sampling power.draw via nvidia-smi.

    __author__ = "Lydia Schönpflug, lydia.schoenpflug[at]unibas[dot]ch"
    __creation__ = "2025"
"""

import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from filelock import FileLock


class GPUStatsLogger:
    """Estimates GPU energy usage and timing using periodic power.draw sampling."""

    def __init__(
        self,
        csv_path: str = "run_summary.csv",
        poll_interval: float = 0.5,
        gpu_ids: Optional[List[int]] = None,
    ) -> None:
        self.csv_path: Path = Path(csv_path)
        self.poll_interval: float = poll_interval
        self.total_start_time: Optional[float] = None
        self.sections: Dict[str, float] = {}
        self.records: List[Dict[str, float | str | int]] = []
        self._polling: bool = False
        self._poll_thread: Optional[threading.Thread] = None
        self._power_readings: defaultdict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._lock: threading.Lock = threading.Lock()
        self.gpu_ids: Optional[List[int]] = gpu_ids
        self._stopped: bool = False

    def _poll_power(self) -> None:
        while self._polling:
            try:
                output = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits",
                        ]
                    )
                    .decode("utf-8")
                    .strip()
                    .splitlines()
                )

                timestamp = time.time()
                powers = [float(x.strip()) for x in output]

                with self._lock:
                    for idx, power in enumerate(powers):
                        if self.gpu_ids is None or idx in self.gpu_ids:
                            self._power_readings[idx].append((timestamp, power))

            except Exception as e:
                print(f"Power polling failed: {e}")
            time.sleep(self.poll_interval)

    def _start_polling(self) -> None:
        self._polling = True
        self._power_readings = defaultdict(list)
        self._poll_thread = threading.Thread(target=self._poll_power)
        self._poll_thread.start()

    def _stop_polling(self) -> None:
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join()

    def _compute_energy(self, t_start: float, t_end: float) -> Dict[int, float]:
        """Compute energy per GPU (in J) using trapezoidal integration."""
        energy_per_gpu: Dict[int, float] = {}

        with self._lock:
            for gpu_id, readings in self._power_readings.items():
                relevant: List[Tuple[float, float]] = [(t, p) for t, p in readings if t_start <= t <= t_end]
                if len(relevant) < 2:
                    energy_per_gpu[gpu_id] = 0.0
                    continue
                relevant.sort()
                energy: float = sum(
                    ((relevant[i][1] + relevant[i - 1][1]) / 2) * (relevant[i][0] - relevant[i - 1][0])
                    for i in range(1, len(relevant))
                )
                energy_per_gpu[gpu_id] = energy

        return energy_per_gpu

    def start(self) -> None:
        self.total_start_time = time.time()
        self._start_polling()

    def start_section(self, label: str) -> None:
        self.sections[label] = time.time()

    def stop_section(self, label: str) -> None:
        if label not in self.sections:
            raise ValueError(f"Section '{label}' was not started.")
        end_time = time.time()
        start_time = self.sections.pop(label)
        energy_used = self._compute_energy(start_time, end_time)
        duration = round(end_time - start_time, 2)

        row = {
            "timestamp": datetime.now().isoformat(),
            "section": label,
            "duration_sec": duration,
        }
        for gpu_id, energy in energy_used.items():
            row[f"energy_gpu_{gpu_id}_J"] = round(energy, 2)

        self.records.append(row)
        self._save_to_csv()

    def stop_and_log_total(self) -> None:
        if self.total_start_time is None:
            raise RuntimeError("Logger was not started.")
        end_time = time.time()
        self._stop_polling()
        energy_used = self._compute_energy(self.total_start_time, end_time)
        duration = round(end_time - self.total_start_time, 2)

        row = {
            "timestamp": datetime.now().isoformat(),
            "section": "TOTAL",
            "duration_sec": duration,
        }
        for gpu_id, energy in energy_used.items():
            row[f"energy_gpu_{gpu_id}_J"] = round(energy, 2)

        self.records.append(row)
        self._save_to_csv()
        self._stopped = True

    def stop_and_log_interrupted(self) -> None:
        if self.total_start_time is None or self._stopped:
            return
        end_time = time.time()
        self._stop_polling()
        energy_used = self._compute_energy(self.total_start_time, end_time)
        duration = round(end_time - self.total_start_time, 2)

        row = {
            "timestamp": datetime.now().isoformat(),
            "section": "TOTAL_INTERRUPTED",
            "duration_sec": duration,
        }
        for gpu_id, energy in energy_used.items():
            row[f"energy_gpu_{gpu_id}_J"] = round(energy, 2)

        self.records.append(row)
        self._save_to_csv()
        self._stopped = True

    def _save_to_csv(self) -> None:
        lock_path = self.csv_path.with_suffix(self.csv_path.suffix + ".lock")
        with FileLock(lock_path):
            df: pd.DataFrame = pd.DataFrame(self.records)
            df.to_csv(self.csv_path, index=False)


if __name__ == "__main__":

    def gpu_matrix_multiply_test(device: int = 0, iterations: int = 100, size: int = 4096) -> None:
        import torch

        torch.cuda.set_device(device)
        a = torch.randn(size, size, device=f"cuda:{device}")
        b = torch.randn(size, size, device=f"cuda:{device}")

        for _ in range(iterations):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()

    base_dir = "/home/user/outputs/gpu_energy_logs"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{base_dir}/gpu_log_{timestamp_str}.csv"
    tracked_gpus = [0, 4]
    interval = 0.5

    logger = GPUStatsLogger(csv_path=log_file, poll_interval=0.2, gpu_ids=tracked_gpus)
    logger.start()
    print("Logger started.")

    logger.start_section("matrix_multiplication")
    gpu_matrix_multiply_test(device=0, iterations=100, size=4096)
    logger.stop_section("matrix_multiplication")
    print("Section logged.")

    logger.stop_and_log_total()
    print("Logging complete.")

    print("Log written to:", logger.csv_path.resolve())
    print(pd.read_csv(logger.csv_path))
