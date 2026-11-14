import numpy as np
import pytest

from candidate.external_model import (FEATURE_NAMES, decode, design_matrix,
                                      frame_features, load_model, postprocess, predict)


def test_external_model_contract_and_frame_spans(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    model = {
        "source": "external", "source_match": "other-game", "training_fps": 10,
        "feature_names": list(FEATURE_NAMES), "labels": ["H", "A", "D"],
        "mean": [0] * 7, "scale": [1] * 7,
        "coef": [[0] * 7, [0] * 7, [0] * 7],
        "intercept": [2, 0, -2], "team_penalty": 1, "dead_penalty": 1,
        "team_bridge_s": 0, "dead_bridge_s": 0,
    }
    import json
    path.write_text(json.dumps(model), encoding="utf-8")
    loaded = load_model(path)
    monkeypatch.setattr("candidate.external_model.half_runs", lambda: [(1, 2, 5), (2, 7, 10)])
    tracking = {
        "px": np.full((12, 2), np.nan), "py": np.full((12, 2), np.nan),
        "ps": np.full((12, 2), np.nan), "bx": np.full(12, np.nan),
        "by": np.full(12, np.nan), "team": np.array([0, 1]),
    }
    output = predict(tracking, loaded)
    assert output.tolist() == ["X", "X", "H", "H", "H", "X", "X", "H", "H", "H", "X", "X"]


def test_external_model_rejects_challenge_source(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"feature_names": ["ball_obs", "d_home", "d_away", "control_net", "ball_out", "mean_speed", "ball_speed"], "labels": ["H", "A", "D"], "source_match": "BAR-ATH"}', encoding="utf-8")
    with pytest.raises(ValueError, match="provided match"):
        load_model(path)


def test_design_matrix_clips_missing_geometry():
    features = {name: np.array([999.0]) for name in FEATURE_NAMES}
    features["ball_obs"] = np.array([0.0])
    features["control_net"] = np.array([np.nan])
    features["ball_out"] = np.array([0.0])
    assert design_matrix(features).tolist() == [[0.0, 20.0, 20.0, 0.0, 0.0, 12.0, 40.0]]
    assert decode(np.array([[0.9, 0.05, 0.05]]), 1, 1).tolist() == ["H"]


def test_shared_frame_features_team_geometry():
    tracking = {
        "px": np.array([[0.5, 0.9], [0.5, 0.9]]),
        "py": np.array([[0.5, 0.5], [0.5, 0.5]]),
        "ps": np.array([[1.0, 3.0], [2.0, 4.0]]),
        "bx": np.array([0.5, 0.6]), "by": np.array([0.5, 0.5]),
        "team": np.array([0, 1]),
    }
    features = frame_features(tracking, 10, (100, 60))
    assert features["d_home"][0] == 0
    assert features["d_away"][0] == 40
    assert features["control_net"][0] > 0.99
    assert features["mean_speed"].tolist() == [2.0, 3.0]
    assert features["ball_speed"][1] == pytest.approx(100)


def test_external_postprocess_bridges_only_short_gaps():
    labels = np.array(list("HHAAHHDDDHHHH"))
    assert "".join(postprocess(labels, team_gap=2, dead_gap=0)) == "HHHHHHDDDHHHH"
    assert "".join(postprocess(labels, team_gap=1, dead_gap=0)) == "HHAAHHDDDHHHH"
