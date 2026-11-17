# External model provenance

The submitted artifact is `candidate/boosted_model.joblib`, SHA-256
`92d70f05567e12da49228284223fa60b456ed175c862a47eceee0ea498e92bbe`. All fitted weights and decoder selection use licensed
external SkillCorner data. The full aggregate audit is [candidate/boosted_validation.json](candidate/boosted_validation.json).

## Source and supervision

- Source: https://github.com/SkillCorner/opendata
- Pinned revision: `4340d274572876239c154c90bc507a9b3250a656`
- License: MIT; copyright SkillCorner, credited with its PySport initiative.
- Inputs: external match metadata, 10 Hz tracking and provider phase-of-play intervals.
- Labels: H/A inside each provider phase; D in the phase complement inside a half.
  Phase ends are exclusive and clipped to their period. This complement assumption and
  provider label quality can introduce supervision error. No labels are made manually
  on the provided game. Individual provider annotators/review procedures are unspecified.

## Fixed match split

- Fit (14): 1959846, 1986691, 1996435, 1996436, 2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461.
- Decoder selection (3): 1925299, 1927964, 1953632.
- Final test (3): 1874553, 1886347, 1899585.

All split groups are disjoint. Fitting uses 822,845 labelled half frames at 10 Hz:
H 222,562, A 207,769, D 392,514. Samples are taken every fifth
external frame and both team perspectives are used. Only the team classifier uses
clipped inverse-possession-duration weights. Training source file hashes, feature names,
classifier configuration and the selected decoder are recorded in the JSON audit.

## Selection and results

The 12 predeclared decoder candidates are ranked by equal-half mean official SpellF1
on the external selection group, with MacroF1 breaking ties. The final model is saved
before opening the final test labels. No change is selected using provided-game labels,
video, or the reported external test results.

| External group | Submitted model SpellF1 | Linear baseline SpellF1 |
| --- | ---: | ---: |
| Selection, six halves | 0.7344 | 0.4729 |
| Final test, six halves | 0.7217 | 0.4730 |

These are diagnostics against external weak labels, not estimates of the competition score.

Selected decoder (seconds-scaled transition costs):

```json
{
  "dead_weight": 1.0,
  "team_weight": 2.0,
  "switch_seconds": 2.0,
  "stop_seconds": 2.0,
  "minimum_dead_seconds": 1.0
}
```

## Reproduction

Follow README.md for inference from the included artifact. External data is not needed
to regenerate the submission. Optional training downloads only the external source;
organizer data is processed locally by the inference command. No video or match-specific
label overrides are used. Expected submission SHA-256: `ce872a26f65b5d7ce9d3659c0cd185b78cd5386177a01808cef0311c655e16af`.
