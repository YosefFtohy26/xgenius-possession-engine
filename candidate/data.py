"""Tactical tracking loading and caching for the possession candidate.

The ATD raw text is parsed once into arrays and cached under ``data/cache`` (which is already
gitignored because it lives inside ``data/``). Nothing here changes the official starter loaders;
it is a faster, dependency-free reader for the 199k x 46 table used by the candidate.
"""

from __future__ import annotations

import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import games  # noqa: E402

CACHE_VERSION = 1
SLOTS = 46


def _paths(source: str):
    paths = games.asset_paths(source=source)
    return Path(paths["atd_meta"]), Path(paths["atd_raw"])


def _team_layout(meta_path: Path):
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]
    members = [p.get("teamId") for p in root.iter("Player")]
    mask = np.array([0 if m == home_id else 1 for m in members], dtype=np.int8)
    return teams, mask


def _cache_path(raw_path: Path, cache_dir: Path, source: str):
    stat = raw_path.stat()
    key = "{}:{}:{}:{}".format(raw_path.resolve(), stat.st_size, stat.st_mtime_ns, CACHE_VERSION)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return cache_dir / "{}_{}.npz".format(source, digest)


def _parse_raw(raw_path: Path, n_frames: int, n_slots: int):
    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    ps = np.full((n_frames, n_slots), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            index = int(parts[0]) - 1
            if index < 0 or index >= n_frames:
                raise ValueError("frame id {} outside 1..{}".format(index + 1, n_frames))
            seen[index] = True

            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or group[0] == "N":
                    continue
                x, y, speed = group.split(",")
                px[index, slot] = float(x)
                py[index, slot] = float(y)
                ps[index, slot] = float(speed)

            ball = parts[2]
            if ball[0] != "N":
                x, y, _ = ball.strip().split(",")
                bx[index] = float(x)
                by[index] = float(y)

    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError("{} missing frame ids in {}".format(missing, raw_path))
    return px, py, ps, bx, by


def load_tracking(source: str = "tactical", cache_dir: str | Path | None = None,
                  verbose: bool = False):
    """Return a dict of frame-aligned tactical tracking arrays.

    Keys: ``px, py, ps`` (N x 46), ``bx, by`` (N), ``ball_obs`` (N bool), ``team`` (46 int8,
    0=home/Barcelona, 1=away/Athletic), ``home_name, away_name`` and path metadata.
    """
    meta_path, raw_path = _paths(source)
    missing = [str(path) for path in (meta_path, raw_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing tracking assets: {}. Unpack the organizers' data bundle "
            "under data/ or set MTAH_DATA_DIR.".format(", ".join(missing)))
    n_frames = games.source_frames(source)
    cache_root = Path(cache_dir) if cache_dir else (ROOT / "data" / "cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(raw_path, cache_root, source)

    if cache_path.is_file():
        if verbose:
            print("tracking cache:", cache_path)
        with np.load(cache_path, allow_pickle=False) as cached:
            return {key: cached[key] for key in cached.files}

    if verbose:
        print("parsing", raw_path)
    teams, team = _team_layout(meta_path)
    px, py, ps, bx, by = _parse_raw(raw_path, n_frames, len(team))
    data = {
        "px": px, "py": py, "ps": ps, "bx": bx, "by": by,
        "ball_obs": np.isfinite(bx) & np.isfinite(by),
        "team": team,
        "home_name": np.array(teams[0][1]),
        "away_name": np.array(teams[1][1]),
    }
    np.savez(cache_path, **data)
    if verbose:
        print("wrote", cache_path)
    return data


def half_runs():
    """Published graded blocks as half-open ``(half, start, stop)`` spans."""
    return [(half, start, stop + 1) for half, start, stop in games.half_video_blocks(games.GAME)]


def graded_mask(n_frames: int | None = None):
    n = n_frames if n_frames is not None else games.source_frames("tactical")
    mask = np.zeros(n, dtype=bool)
    for _, start, stop in half_runs():
        mask[start:stop] = True
    return mask
