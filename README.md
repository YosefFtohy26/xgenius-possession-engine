# xGenius Possession Engine — More Than a Hack 2027

Reproduce a frame-level H/A/D/X possession submission from the organizers' automatic
tactical tracking using a frozen model fitted on separate SkillCorner matches.

## Reproduce (Python 3.12)

```bash
python -m pip install -r requirements-candidate.txt
python -m candidate.pipeline --output submission.csv --metadata submission_metadata.json
python evaluation/validate_submission.py submission.csv
```

Place the licensed organizer files `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt` in `data/tactical/` first, or set `MTAH_DATA_DIR`
to the data bundle folder (containing `tactical/`). No video is needed.
The model artifact is included; training is not required to reproduce the CSV.

Expected output: 199,319 rows; SHA-256
`ba7f7942f705a6704d567b7f08cab1efaca35e128608d86c6a8a736e799c158d`.
This hash was verified from this source-only folder using a fresh tracking parse.

## Training and external validation

See [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) for the source matches and labels.
Download the three files for each external match (match JSON, tracking JSONL, phases CSV)
from [SkillCorner Open Data](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656/data/matches).
The Git LFS tracking file can be downloaded through its media.githubusercontent.com URL.
Keep match folders named `2017461` and `2006229`.

```bash
python -m candidate.train_external --match-dir data/external/skillcorner-opendata/data/matches/2017461
python -m candidate.evaluate_external --match-dir data/external/skillcorner-opendata/data/matches/2006229
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Documentation and rules

- [Technical document](TECHNICAL_DOCUMENT.md)
- [Compliance review and historical disclosure](COMPLIANCE_REVIEW.md)
- [Official submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)
- [Official rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)

Only source, frozen coefficients, aggregate external metrics, and documentation are
published here. Match assets, ground-truth labels, generated frame-level output, and
video are distributed separately under the organizers' terms. Upload your generated
CSV and technical document through the official form, using your registered team name.

## Attribution

Optional local prediction-only visual:

```bash
python -m pip install -r requirements-visualization.txt
python tools/plot_submission.py --csv submission.csv --output outputs/submission_overview.png
```

This plot uses the generated CSV only. Do not compare predictions against the supplied video.

The official validator, metric library, and game constants are adapted/copied from Metrica
Sports' MIT-licensed challenge repository; its notice is retained in LICENSE.
External training uses SkillCorner Open Data under MIT; see THIRD_PARTY_NOTICES.md.
