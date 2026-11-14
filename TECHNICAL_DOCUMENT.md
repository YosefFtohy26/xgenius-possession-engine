# xGenius: external-data-trained possession submission

## Task and output

The program labels every tactical-video frame of Barcelona vs Athletic Club as `H`
(Barcelona in controlled possession while play is live), `A` (Athletic in controlled
possession), `D` (ball out of play), or `X` (ungraded time). The output is a two-column
`frame,label` CSV covering frames 0–199318. The two published graded halves receive
H/A/D predictions; the pre-match, halftime, and post-match intervals receive X.

An earlier prototype used the public window for tuning and was retired. This repository
contains the replacement only. See COMPLIANCE_REVIEW.md for the disclosure.

## Inputs and method

Inference uses the organizers' **automatic tactical tracking**: player and ball positions,
team membership, and player speeds. It does not use the provided video, broadcast tracking,
Smart Tagging, manually created game data, or the public possession ground truth as input.
The tactical ATD frame ID is converted from one-based to the zero-based tactical video frame.

Seven physical features are computed for each frame: ball visibility; distance from the
ball to the nearest home and away player; relative soft control within 3 m of the ball;
whether the observed ball is outside the pitch; mean player speed; and ball speed.
Missing ball observations are represented explicitly. A balanced multinomial logistic
regression predicts probabilities for H, A, and D. A three-state Viterbi decoder discourages
rapid switches. A short-gap postprocess joins isolated labels. This preserves a team through
brief uncertainty instead of treating every close touch as a turnover. A restart begins a new
possession after D; dead-ball ownership is not predicted.

The physical feature extractor is shared by external training and this submission. It uses
the standard 105 × 68 m pitch and does not hardcode labels for any particular interval of
the challenge game.

## Training, tuning, and rights

All fitted coefficients and transition/gap parameters come from
[SkillCorner Open Data](https://github.com/SkillCorner/opendata), a different Australian
A-League match (`2017461`). SkillCorner's repository carries an
[MIT license](https://github.com/SkillCorner/opendata/blob/master/LICENSE); credit to
SkillCorner. Its 10 fps tracking supplies positions, and its phase-of-play intervals supply
external team-possession labels. Frames within an external match period outside those phases
are treated as D. No label is made by hand on the provided Barcelona game.

The first 80% of external period 1 fits the logistic regression. The last 20% selects
two transition penalties and two short-gap durations by possession-spell F1. External
period 2 is held out. A second match (`2006229`) is evaluated with the model frozen and
never used for selection. Its two halves scored spell F1 0.4523 and 0.4253 against
SkillCorner's labels. These are external diagnostics, **not** competition scores.
Full provenance, split, source hashes, and limitations are in [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

## Reproduction and checks

Use Python 3.12 and install `requirements-candidate.txt`. Place the organizers' tactical
ATD XML and TXT under `data/tactical/`, or configure `MTAH_DATA_DIR`. Then run:

```bash
python -m candidate.pipeline --output submission.csv --metadata submission_metadata.json
python evaluation/validate_submission.py submission.csv
python -m pytest -q
```

The current CSV has 199,319 rows. Two independent generations were byte-identical, with
SHA-256 `ba7f7942f705a6704d567b7f08cab1efaca35e128608d86c6a8a736e799c158d`.
The official format validator passes. The six focused public tests pass; the development checkout also passed 48 tests. The local automatic quality gate
has one nonfatal warning: 5.03 predicted possession spells per minute, slightly above its
5.00 heuristic limit. The current CSV was not checked against the provided video or public labels. Earlier prototype work is disclosed in COMPLIANCE_REVIEW.md.

## Limitations

The ball is missing in much of the automatic tracking, and nearby-player geometry does not
always distinguish a controlled reception from a deflection. SkillCorner's broadcast
tracking and phase labels differ from the challenge's tactical data and possession
definition. External validation still shows excess possession fragmentation. The private
full-match score is unknown; format and plausibility checks cannot establish accuracy.
