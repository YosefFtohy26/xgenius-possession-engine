# xGenius: tracking-based possession inference

## Task and output

The system generates one `H`, `A`, `D`, or `X` label for each of the 199,319 tactical
video frames. H and A denote the home and away teams in live possession; D denotes
dead ball. Published graded-half boundaries determine where inference runs; all other
frames receive X. There are no match-specific possession or kickoff overrides.

## Inputs and method

Inference uses only the provided automatic tactical tracking and team metadata.
Video, broadcast tracking, homographies, and Smart Tagging are not needed by this method.
Tracking frame IDs are converted from one-based to zero-based video indices.

Two histogram gradient boosting classifiers estimate dead-ball probability and team
possession conditional on live play. Each has 89 inputs describing ball availability,
motion, distance to pitch boundaries, nearby-player control, team movement and shape.
Temporal means over 0.5, 2, and 5 seconds and time before/after control and observations
provide context. Features are computed independently within each half. Future context
makes this an offline method. Geometry uses a 105 by 68 m pitch for the provided match.

Ball gaps of at most 0.4 seconds are interpolated between plausible endpoints. Longer
gaps remain missing and availability is represented explicitly. Both team perspectives
are used during training and averaged at inference. The dead-ball inputs are invariant
under swapping team names. Boosted trees model interactions among these signals, while
separate classifiers distinguish stoppage detection from team control.

## Training data and selection

The external source is [SkillCorner Open Data](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656),
credited to SkillCorner and its initiative with PySport, under its
[MIT license](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE).
SkillCorner supplies tracking and Game Intelligence phase-of-play intervals. Our adapter
assigns the phase's possessing team and assumes uncovered frames within each half are D.
These are provider-derived weak labels; their individual annotation/review process is
not specified in our source documentation. We create no manual labels for the provided game.

Twenty external Australian A-League matches are split by match into 14 fitting matches,
three for decoder selection, and three for final testing. Fitting covers 822,845 labelled
frames at 10 Hz (22.9 hours) before subsampling to 2 Hz. Both team perspectives are
included, and team-classifier sample weights reduce domination by long possessions.
Each classifier uses 400 boosting iterations, learning rate 0.05, up to 31 leaves,
minimum leaf size 200, L2 regularization 1, and seed 2026. Early stopping is disabled.

Decoder settings are selected by mean official SpellF1 on the three external selection
matches, with MacroF1 breaking ties. The candidate must improve on the frozen linear
baseline there before final test labels are opened. No provided-game labels or video
are used for fitting or parameter selection. The final external six-half mean SpellF1
is 0.7217; baseline 0.4730 on those same halves.
These are external diagnostics, not competition scores. Match lists, hashes, settings,
and per-half aggregate results are in [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

## Possession rules and limitations

Control evidence combines proximity, opponent distance, and ball speed. A nearby touch
alone does not force a turnover. A duration-constrained Viterbi decoder combines frame
probabilities across each half and charges for team changes and live/dead transitions.
Passes and loose balls can retain team possession through temporal context; no separate
contested state is emitted. Missing ball data alone does not force D or X. Stoppages
within a half must last at least 1 second; half-edge intervals are exempt.
After D, renewed H/A control starts the next possession. Dead-ball ownership is not emitted.

Tracking errors, missing balls, and differences between broadcast-derived external labels
and the challenge's possession definition remain limitations. There is no referee-whistle
detector, and deflections can resemble control. More diverse external data and better
automatic control evidence would be the next improvements. The private full-match score
is unknown; format checks do not establish football accuracy.

## Reproduction

Use Python 3.12 and the pinned dependencies. Place `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt` in `data/tactical/`, or set `MTAH_DATA_DIR` to the folder
containing `tactical/`. The frozen model is included; training is not required.

```bash
python -m pip install -r requirements-candidate.txt
python -m candidate.boosted_pipeline --output submission.csv --metadata submission_metadata.json
python evaluation/validate_submission.py submission.csv
```

Expected CSV SHA-256: `ce872a26f65b5d7ce9d3659c0cd185b78cd5386177a01808cef0311c655e16af`.
Install `requirements-dev.txt` and run `python -m pytest -q` for software checks.
