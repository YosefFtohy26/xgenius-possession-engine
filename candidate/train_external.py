"""Fit a possession model on SkillCorner Open Data, never on the challenge game.

Usage: python -m candidate.train_external --match-dir data/external/skillcorner-opendata/data/matches/2017461
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from candidate.external_model import (FEATURE_NAMES, LABELS, decode, design_matrix,
                                      frame_features, postprocess, probabilities)


def load_match(match_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, str]:
    match_id = match_dir.name
    meta = json.loads((match_dir / f"{match_id}_match.json").read_text(encoding="utf-8"))
    home_id = meta["home_team"]["id"]
    roster = [p for p in meta["players"] if p["team_id"] in
              (home_id, meta["away_team"]["id"])]
    slot = {player["id"]: index for index, player in enumerate(roster)}
    team = np.array([0 if player["team_id"] == home_id else 1 for player in roster], dtype=np.int8)
    tracking_path = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
    expected = int(meta["match_periods"][-1]["end_frame"])
    rows = []
    digest = hashlib.sha256()
    with tracking_path.open(encoding="utf-8") as handle:
        for line in handle:
            digest.update(line.encode("utf-8"))
            rows.append(json.loads(line))
    if expected is not None and len(rows) < expected:
        raise ValueError("incomplete external tracking file")

    n = len(rows)
    px = np.full((n, len(roster)), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    period = np.zeros(n, dtype=np.int8)
    previous_players: dict[int, tuple[float, float]] = {}
    fps = 10.0
    pitch_l = float(meta["pitch_length"])
    pitch_w = float(meta["pitch_width"])

    for i, row in enumerate(rows):
        if row["frame"] != i:
            raise ValueError(f"external frame index discontinuity at {i}")
        period[i] = int(row["period"] or 0)
        ball = row["ball_data"]
        ball_x, ball_y = ball.get("x"), ball.get("y")
        if ball_x is not None and ball_y is not None:
            bx[i] = (float(ball_x) + pitch_l / 2) / pitch_l
            by[i] = (float(ball_y) + pitch_w / 2) / pitch_w
        current_players = {}
        for player in row["player_data"]:
            x, y = player.get("x"), player.get("y")
            if x is None or y is None:
                continue
            pid = player["player_id"]
            if pid not in slot:
                continue
            column = slot[pid]
            current_players[pid] = (float(x), float(y))
            px[i, column] = (float(x) + pitch_l / 2) / pitch_l
            py[i, column] = (float(y) + pitch_w / 2) / pitch_w
            if period[i] and pid in previous_players:
                old_x, old_y = previous_players[pid]
                ps[i, column] = np.hypot(x - old_x, y - old_y) * fps
        previous_players = current_players if period[i] else {}
    features = frame_features({"px": px, "py": py, "ps": ps, "bx": bx, "by": by,
                               "team": team}, fps, (pitch_l, pitch_w))

    labels = np.full(n, "X", dtype="<U1")
    labels[period > 0] = "D"
    phases_path = match_dir / f"{match_id}_phases_of_play.csv"
    with phases_path.open(newline="", encoding="utf-8") as handle:
        for phase in csv.DictReader(handle):
            start, stop = int(phase["frame_start"]), int(phase["frame_end"])
            if start < 0 or stop > n or start >= stop:
                raise ValueError("invalid external phase frame range")
            label = "H" if int(phase["team_in_possession_id"]) == home_id else "A"
            labels[start:stop] = label
    return features, labels, period, digest.hexdigest()


def _spells(labels: np.ndarray) -> list[tuple[int, int, str]]:
    out = []
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and labels[stop] == labels[start]:
            stop += 1
        if labels[start] in ("H", "A"):
            out.append((start, stop, str(labels[start])))
        start = stop
    return out


def spell_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    from evaluation import possession_metrics as pm
    actual = pm.label_codes(truth)
    guesses = pm.label_codes(predicted)
    return pm.spell_scores(actual, guesses, [(0, len(truth))])["mean_f1"]


def _score(truth: np.ndarray, predicted: np.ndarray) -> dict:
    return {"spell_f1": spell_f1(truth, predicted),
            "macro_f1": float(f1_score(truth, predicted, labels=list(LABELS), average="macro")),
            "frames": int(len(truth)), "true_spells": len(_spells(truth)),
            "predicted_spells": len(_spells(predicted))}


def train(match_dir: Path, output: Path) -> dict:
    features, labels, period, tracking_hash = load_match(match_dir)
    matrix = design_matrix(features)
    first = np.flatnonzero(period == 1)
    second = np.flatnonzero(period == 2)
    if len(first) < 1000 or len(second) < 1000:
        raise ValueError("both external halves are required")
    split = first[int(len(first) * 0.8)]
    train_indices = first[first < split]
    tune_indices = first[first >= split]
    mean = matrix[train_indices].mean(axis=0)
    scale = matrix[train_indices].std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    classifier = LogisticRegression(max_iter=500, class_weight="balanced")
    classifier.fit(normalized[train_indices], labels[train_indices])
    if list(classifier.classes_) != list(LABELS):
        order = [list(classifier.classes_).index(label) for label in LABELS]
    else:
        order = list(range(3))
    model = {
        "source": "https://github.com/SkillCorner/opendata",
        "license": "MIT",
        "source_match": match_dir.name,
        "tracking_sha256": tracking_hash,
        "feature_names": list(FEATURE_NAMES), "labels": list(LABELS),
        "training_fps": 10,
        "mean": mean.tolist(), "scale": scale.tolist(),
        "coef": classifier.coef_[order].tolist(),
        "intercept": classifier.intercept_[order].tolist(),
        "split": "period 1 first 80% fit, period 1 last 20% tune, period 2 held out",
    }
    probs = probabilities(features, model)
    options = (0.5, 1.0, 2.0, 4.0, 8.0)
    team_bridges = (0.0, 0.5, 1.0, 2.0)
    dead_bridges = (0.0, 0.5, 1.0)
    candidates = []
    for team_penalty in options:
        for dead_penalty in options:
            raw = decode(probs[tune_indices], team_penalty, dead_penalty)
            for team_bridge in team_bridges:
                for dead_bridge in dead_bridges:
                    predicted = postprocess(raw, round(team_bridge * 10), round(dead_bridge * 10))
                    scores = _score(labels[tune_indices], predicted)
                    candidates.append((scores["spell_f1"], scores["macro_f1"],
                                       -team_penalty, -dead_penalty, -team_bridge, -dead_bridge,
                                       team_penalty, dead_penalty, team_bridge, dead_bridge))
    best = max(candidates)
    (model["team_penalty"], model["dead_penalty"], model["team_bridge_s"],
     model["dead_bridge_s"]) = best[-4:]
    held_raw = decode(probs[second], model["team_penalty"], model["dead_penalty"])
    held_pred = postprocess(held_raw, round(model["team_bridge_s"] * 10),
                            round(model["dead_bridge_s"] * 10))
    model["external_holdout"] = _score(labels[second], held_pred)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("external_config.json"))
    args = parser.parse_args()
    model = train(args.match_dir, args.output)
    print("wrote", args.output)
    print("external held-out second half:", model["external_holdout"])


if __name__ == "__main__":
    main()
