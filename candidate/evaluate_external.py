"""Evaluate a frozen external model on a different SkillCorner match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from candidate.external_model import decode, load_model, postprocess, probabilities
from candidate.train_external import _score, load_match


def evaluate(match_dir: Path, model_path: Path) -> dict:
    model = load_model(model_path)
    if match_dir.name == model["source_match"]:
        raise ValueError("cross-match validation requires a different match")
    features, labels, period, tracking_hash = load_match(match_dir)
    probs = probabilities(features, model)
    results = {}
    for half in (1, 2):
        indices = np.flatnonzero(period == half)
        raw = decode(probs[indices], model["team_penalty"], model["dead_penalty"])
        prediction = postprocess(raw, round(model["team_bridge_s"] * 10),
                                 round(model["dead_bridge_s"] * 10))
        results[f"half_{half}"] = _score(labels[indices], prediction)
    return {
        "source_match": model["source_match"],
        "validation_match": match_dir.name,
        "validation_tracking_sha256": tracking_hash,
        "metrics": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).with_name("external_config.json"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("external_validation.json"))
    args = parser.parse_args()
    result = evaluate(args.match_dir, args.model)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
