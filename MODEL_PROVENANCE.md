# External possession model provenance

## Data and rights

The submitted model uses [SkillCorner Open Data](https://github.com/SkillCorner/opendata),
specifically Australian A-League match `2017461`. Its tracking file and phase-of-play CSV
are from a **different match** than the challenge game. Match `2006229` is a separate,
frozen validation match. SkillCorner describes this dataset
as open sourced, and the repository carries an [MIT license](https://github.com/SkillCorner/opendata/blob/master/LICENSE).
Credit: SkillCorner Open Data, copyright SkillCorner. The local copy is kept under ignored
`data/external/skillcorner-opendata/`; no match data is committed with the model.

The external tracking runs at 10 fps and has player and ball positions in pitch metres.
The phase file names the team in possession during live play. Frames within each period
outside a phase are treated as dead ball. These are provider-generated labels. They may
not exactly match the challenge's definition of controlled possession, and broadcast
tracking has different missingness from the tactical input.

## Method

`candidate/train_external.py` converts the external coordinates to normalised pitch space.
The shared `candidate/external_model.py` extractor computes seven features from tracking:
ball visibility, each team's nearest-player distance, a soft 3 m control difference,
ball outside the pitch, mean player speed, and ball speed. No provided-game possession
label, video review, or Smart Tagging enters this model.
Inference uses a standard 105 × 68 m pitch and decodes both graded halves directly;
it does not force game-specific labels at the second-half kickoff.

A balanced multinomial logistic regression learns H/A/D emission probabilities. A
three-state Viterbi decoder adds team-switch and live/dead transition costs. A short-gap
postprocess bridges isolated labels. The penalties and gap durations are tuned on the
external match only; penalties and durations are scaled by frame rate at 25 fps.

Fit split: first 80% of period 1. Penalty and gap tuning: last 20% of period 1. Held-out check:
period 2. The source tracking SHA-256 and fitted coefficients are in
`candidate/external_config.json`. A second external match is used as a frozen cross-match
check and is never used to select the model.

## Reproduce

```bash
python -m candidate.train_external --match-dir data/external/skillcorner-opendata/data/matches/2017461
python -m candidate.evaluate_external --match-dir data/external/skillcorner-opendata/data/matches/2006229
python -m candidate.pipeline --output submission.csv --metadata submission_metadata.json
python evaluation/validate_submission.py submission.csv
```

The final two commands need the organizers' gated BAR-ATH tactical tracking bundle under
`data/tactical/` or configured through `MTAH_DATA_DIR`. The broadcast video is not used.

## Current evidence and limits

The external source match's held-out second half scored spell F1 **0.3932** and macro F1
**0.8066** against SkillCorner's phase-derived labels. This is an external diagnostic,
**not** an estimate of the challenge score.

The frozen model scored spell F1 **0.4523** and **0.4253** on the first and second halves
of external match `2006229`. It predicted 184 and 193 spells where the external labels
contain 122 and 123. Fragmentation remains a limitation; these results were
recorded without refitting or selecting parameters on the validation match. Exact
aggregates and the second tracking hash are in `candidate/external_validation.json`.

The regenerated challenge `submission.csv` has 199,319 rows and SHA-256
`ba7f7942f705a6704d567b7f08cab1efaca35e128608d86c6a8a736e799c158d`.
A second run was byte-identical. The official format validator and the six focused
public software tests pass. These checks do not measure challenge accuracy or
substitute for the organizers' private evaluation.
