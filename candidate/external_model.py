"""Possession model fitted exclusively on licensed external matches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate.data import half_runs


FEATURE_NAMES = (
    "ball_obs", "d_home", "d_away", "control_net", "ball_out",
    "mean_speed", "ball_speed",
)
LABELS = np.array(["H", "A", "D"])


def frame_features(tracking: dict, fps: float, pitch: tuple[float, float],
                   control_radius: float = 3.0) -> dict[str, np.ndarray]:
    """Extract the same seven physical signals from either provider's tracking.

    Coordinates are normalised to [0, 1] pitch space; speeds are metres/second.
    No tagging, video, possession labels, or game-specific frame constants enter here.
    """
    px, py, ps = tracking["px"], tracking["py"], tracking["ps"]
    bx, by = tracking["bx"], tracking["by"]
    team = tracking["team"]
    pitch_l, pitch_w = pitch
    n = len(bx)
    observed = np.isfinite(bx) & np.isfinite(by)
    valid_players = np.isfinite(px) & np.isfinite(py)
    dx = (px - np.where(observed, bx, 0.5)[:, None]) * pitch_l
    dy = (py - np.where(observed, by, 0.5)[:, None]) * pitch_w
    distance = np.hypot(dx, dy)
    distance[~valid_players | ~observed[:, None]] = np.inf
    home = team == 0
    away = team == 1
    home_distance = distance[:, home]
    away_distance = distance[:, away]
    if home_distance.shape[1] == 0 or away_distance.shape[1] == 0:
        raise ValueError("both teams need player tracking slots")
    d_home = home_distance.min(axis=1)
    d_away = away_distance.min(axis=1)
    home_control = np.exp(-0.5 * (home_distance / control_radius) ** 2).sum(axis=1)
    away_control = np.exp(-0.5 * (away_distance / control_radius) ** 2).sum(axis=1)
    control_net = (home_control - away_control) / (home_control + away_control + 1e-6)
    speed_valid = valid_players & np.isfinite(ps)
    mean_speed = np.where(speed_valid, ps, 0).sum(axis=1) / np.maximum(1, speed_valid.sum(axis=1))
    ball_speed = np.zeros(n)
    adjacent = observed[1:] & observed[:-1]
    ball_speed[1:] = np.where(
        adjacent, np.hypot(np.diff(bx) * pitch_l, np.diff(by) * pitch_w) * fps, 0)
    ball_out = observed & ((bx < 0) | (bx > 1) | (by < 0) | (by > 1))
    return {
        "ball_obs": observed.astype(float),
        "d_home": np.nan_to_num(d_home, posinf=999),
        "d_away": np.nan_to_num(d_away, posinf=999),
        "control_net": control_net,
        "ball_out": ball_out.astype(float),
        "mean_speed": mean_speed,
        "ball_speed": ball_speed,
    }


def design_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Use physical, frame-level features shared by both tracking formats."""
    columns = [np.asarray(features[name], dtype=np.float64) for name in FEATURE_NAMES]
    columns[1] = np.clip(columns[1], 0, 20)
    columns[2] = np.clip(columns[2], 0, 20)
    columns[5] = np.clip(columns[5], 0, 12)
    columns[6] = np.clip(columns[6], 0, 40)
    return np.nan_to_num(np.column_stack(columns), nan=0, posinf=0, neginf=0)


def probabilities(features: dict[str, np.ndarray], model: dict) -> np.ndarray:
    matrix = design_matrix(features)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    logits = ((matrix - mean) / scale) @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def decode(probs: np.ndarray, team_penalty: float, dead_penalty: float) -> np.ndarray:
    """Three-state Viterbi; penalties are in negative-log-probability units."""
    n = len(probs)
    if n == 0:
        return np.empty(0, dtype="<U1")
    cost = -np.log(np.clip(probs, 1e-12, 1.0))
    transitions = np.array([
        [0, team_penalty, dead_penalty],
        [team_penalty, 0, dead_penalty],
        [dead_penalty, dead_penalty, 0],
    ])
    totals = cost[0].copy()
    back = np.empty((n, 3), dtype=np.int8)
    for frame in range(1, n):
        options = totals[:, None] + transitions
        back[frame] = np.argmin(options, axis=0)
        totals = cost[frame] + options[back[frame], np.arange(3)]
    states = np.empty(n, dtype=np.int8)
    states[-1] = int(np.argmin(totals))
    for frame in range(n - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    return LABELS[states]


def _bridge(labels: np.ndarray, target: str, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return labels
    output = labels.copy()
    while True:
        changed = False
        start = 0
        while start < len(output):
            stop = start + 1
            while stop < len(output) and output[stop] == output[start]:
                stop += 1
            if (output[start] != target and start > 0 and stop < len(output)
                    and output[start - 1] == target and output[stop] == target
                    and stop - start <= max_gap):
                output[start:stop] = target
                changed = True
                break
            start = stop
        if not changed:
            return output


def postprocess(labels: np.ndarray, team_gap: int, dead_gap: int) -> np.ndarray:
    labels = _bridge(labels, "D", dead_gap)
    labels = _bridge(labels, "H", team_gap)
    return _bridge(labels, "A", team_gap)


def load_model(path: str | Path) -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("feature_names") != list(FEATURE_NAMES) or model.get("labels") != list(LABELS):
        raise ValueError("external model feature/label contract mismatch")
    if model.get("source_match") == "BAR-ATH":
        raise ValueError("the provided match cannot be a training source")
    return model


def predict(tracking: dict, model: dict, fps: int = 25,
            pitch: tuple[float, float] = (105.0, 68.0)) -> np.ndarray:
    features = frame_features(tracking, fps, pitch)
    probs = probabilities(features, model)
    result = np.full(len(tracking["bx"]), "X", dtype="<U1")
    factor = fps / float(model["training_fps"])
    tp = float(model["team_penalty"]) * factor
    dp = float(model["dead_penalty"]) * factor
    team_gap = round(float(model["team_bridge_s"]) * fps)
    dead_gap = round(float(model["dead_bridge_s"]) * fps)
    for _, start, stop in half_runs():
        piece = decode(probs[start:stop], tp, dp)
        result[start:stop] = postprocess(piece, team_gap, dead_gap)
    return result
