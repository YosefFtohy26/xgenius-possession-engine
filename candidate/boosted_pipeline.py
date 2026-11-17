"""Reproduce the submission using the frozen externally trained boosted model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from candidate.data import ROOT, load_tracking, half_runs
from candidate.boosted_model import load_bundle, predict_match
from candidate.pipeline import write_submission, validate_file


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=Path(__file__).with_name('boosted_model.joblib'))
    parser.add_argument('--output',type=Path,default=ROOT/'submission.csv')
    parser.add_argument('--metadata',type=Path,default=ROOT/'submission_metadata.json')
    parser.add_argument('--quiet',action='store_true')
    a=parser.parse_args(argv)
    model=load_bundle(a.model)
    tracking=load_tracking(verbose=not a.quiet)
    prediction=predict_match(tracking,model,25,[(start,stop) for _,start,stop in half_runs()])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.csv.partial')
    write_submission(prediction,temporary)
    reason=validate_file(temporary)
    if reason:
        temporary.unlink(missing_ok=True)
        raise ValueError(reason)
    temporary.replace(a.output)
    metadata={'model_type':'two_stage_external_boosted_trees',
              'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
              'submission_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
              'rows':len(prediction),'fit_matches':model['fit_matches'],'decoder':model['decoder'],
              'counts':dict(zip(*[x.tolist() for x in np.unique(prediction,return_counts=True)])),
              'validation':'OK'}
    a.metadata.parent.mkdir(parents=True,exist_ok=True)
    a.metadata.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if not a.quiet:
        print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
