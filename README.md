# xGenius Possession Engine

Frame-level H/A/D/X possession inference for More Than a Hack 2027 using a frozen,
externally trained two-stage boosted-tree model and temporal tracking features.

## Reproduce the submission (Python 3.12)

Place the licensed organizer files `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt` in `data/tactical/`, or set `MTAH_DATA_DIR` to the data folder
containing `tactical/`. No video or public possession labels are needed for inference.

```bash
python -m pip install -r requirements-candidate.txt
python -m candidate.boosted_pipeline --output submission.csv --metadata submission_metadata.json
python evaluation/validate_submission.py submission.csv
```

Expected output: 199,319 rows, frames 0 through 199318. SHA-256:
`c7b164111135b5db0f853cea70915675fa7ad12fb926d4a41521a81eb985a23c`

The frozen `candidate/boosted_model.joblib` is included. Exact CSV reproduction uses that
artifact and the pinned environment; retraining is optional. Prediction uses one CPU thread.
The pipeline applies one deterministic structural rule: the organizer-published
second-half pre-kickoff block (frames 110,808–113,741) is set to D before writing the CSV.
Match assets, CSV predictions, caches, and videos are excluded from this public repository.

## Optional external training

SkillCorner's MIT-licensed data is pinned to revision
`4340d274572876239c154c90bc507a9b3250a656`. The downloader needs curl and verifies
every external tracking file against its pinned SHA-256.

```bash
python -m candidate.download_external
python -m candidate.train_boosted
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Training fixes the 14/3/3 match split in source and uses six CPU threads. Selection uses
external matches only. The linear model modules and frozen JSON support the external
baseline comparison in training. They are not the default submission method.

## Documentation and rights

- [Technical document](TECHNICAL_DOCUMENT.md)
- [Training provenance and external results](MODEL_PROVENANCE.md)
- [Official submission requirements](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)
- [Official rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)

The supplied match data is licensed for the competition and must be obtained separately.
The validator, metric library, and game metadata retain Metrica Sports' MIT notice in
LICENSE. SkillCorner attribution and its MIT notice are in THIRD_PARTY_NOTICES.md.
The boosted model, temporal feature extractor, and duration decoder are original implementations.

## Optional prediction-only chart

```bash
python -m pip install -r requirements-visualization.txt
python tools/plot_submission.py --csv submission.csv --output outputs/submission_overview.png
```

This chart reads predictions alone. Do not compare predictions against the supplied video.
