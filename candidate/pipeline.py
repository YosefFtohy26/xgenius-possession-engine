"""Generate and validate a full-match possession submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from candidate.data import ROOT, load_tracking
from candidate.external_model import load_model, predict
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "submission.csv"))
    parser.add_argument("--metadata", default=str(ROOT / "submission_metadata.json"))
    parser.add_argument("--model", default=str(ROOT / "candidate" / "external_config.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        tracking = load_tracking(verbose=not args.quiet)
    except FileNotFoundError as error:
        parser.error(str(error))
    model = load_model(args.model)
    labels = predict(tracking, model, fps=GAMES["BAR-ATH"]["fps"],
                     pitch=(105.0, 68.0))
    output = write_submission(labels, args.output)
    reason = validate_file(output)

    result = {
        "submission": str(output),
        "feature_names": model["feature_names"],
        "n_frames": int(len(labels)),
        "model": str(args.model),
        "external_source": model["source"],
        "external_match": model["source_match"],
        "external_tracking_sha256": model["tracking_sha256"],
        "counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "validation": "OK" if reason is None else reason,
    }

    meta_path = Path(args.metadata)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.quiet:
        print("wrote", output)
        print("validation:", result["validation"])
        print("counts:", result["counts"])
    return 0 if reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
