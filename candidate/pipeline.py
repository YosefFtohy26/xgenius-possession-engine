"""Generate and validate a full-match possession submission."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from candidate.data import ROOT, load_tracking
from games import GAMES

EVALUATION = ROOT / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))
def write_submission(labels, path):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False, lineterminator="\r\n")
    return path


def validate_file(path: Path):
    import possession_metrics as pm
    table = pm.read_label_csv(path)
    return pm.validate_submission(table, GAMES["BAR-ATH"]["n_video_frames"])


def main(argv=None):
    # Keep the established entry point while sharing the validated CSV writer.
    from candidate.boosted_pipeline import main as boosted_main
    return boosted_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
