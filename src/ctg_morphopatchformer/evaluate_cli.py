"""Score caller-supplied arrays; no data download or training."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from .evaluation import evaluate_record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score y_true/y_pred arrays from your own NPZ file.")
    parser.add_argument("input", help="NPZ with numeric one-dimensional y_true and y_pred arrays")
    parser.add_argument("--fs-hz", type=float, required=True, help="Actual sampling rate of both arrays")
    args = parser.parse_args(argv)
    with np.load(args.input, allow_pickle=False) as payload:
        metrics = evaluate_record(payload["y_true"], payload["y_pred"], fs_hz=args.fs_hz)
    print(json.dumps({k: v if math.isfinite(v) else None for k, v in metrics.items()},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
