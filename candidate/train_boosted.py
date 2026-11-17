"""Fit, tune, and test using disjoint licensed external matches only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from candidate.boosted_model import FEATURE_VERSION, feature_matrices, predict_probabilities, decode_intervals
from candidate.train_external import _score

TEST = ('1874553','1886347','1899585')
TUNE = ('1925299','1927964','1953632')


def external_halves(folder, cache_root):
    match_id = folder.name
    if not match_id.isdigit():
        raise ValueError('External numeric match ID required')
    paths = [folder/f'{match_id}_{suffix}' for suffix in ('match.json','phases_of_play.csv','tracking_extrapolated.jsonl')]
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    expected=json.loads(Path(__file__).with_name('external_sources.json').read_text())['tracking'][match_id]
    if hashes[paths[2].name]!=expected['sha256']:
        raise ValueError(f'External tracking is incomplete or does not match pinned source: {match_id}')
    digest = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    cache = cache_root/f'{match_id}_v{FEATURE_VERSION}_{digest}.joblib'
    if cache.exists():
        return joblib.load(cache),hashes
    meta = json.loads(paths[0].read_text(encoding='utf-8'))
    home = meta['home_team']['id']; away = meta['away_team']['id']
    roster = [p for p in meta['players'] if p['team_id'] in (home,away)]
    slots = {p['id']:i for i,p in enumerate(roster)}
    with paths[2].open(encoding='utf-8') as handle:
        n = sum(1 for _ in handle)
    px = np.full((n,len(roster)),np.nan,dtype=np.float32); py = px.copy()
    bx = np.full(n,np.nan,dtype=np.float32); by = bx.copy()
    period = np.zeros(n,dtype=np.int8)
    length,width = float(meta['pitch_length']),float(meta['pitch_width'])
    with paths[2].open(encoding='utf-8') as handle:
        for i,line in enumerate(handle):
            row=json.loads(line)
            if row['frame']!=i:
                raise ValueError('Discontinuous external tracking')
            period[i]=int(row['period'] or 0)
            ball=row['ball_data']
            if ball.get('x') is not None and ball.get('y') is not None:
                bx[i]=(ball['x']+length/2)/length; by[i]=(ball['y']+width/2)/width
            for player in row['player_data']:
                slot=slots.get(player['player_id'])
                if slot is not None and player.get('x') is not None and player.get('y') is not None:
                    px[i,slot]=(player['x']+length/2)/length; py[i,slot]=(player['y']+width/2)/width
    labels=np.where(period>0,'D','X')
    with paths[1].open(newline='',encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            start,stop=int(row['frame_start']),int(row['frame_end'])
            if not 0<=start<stop<=n:
                raise ValueError('Invalid external phase')
            # Phase supervision is clipped to the declared period.
            half=int(row['period']) if row.get('period') else int(period[start])
            indices=np.arange(start,stop)
            indices=indices[period[indices]==half]
            labels[indices]='H' if int(row['team_in_possession_id'])==home else 'A'
    halves=[]
    team=np.array([0 if p['team_id']==home else 1 for p in roster],dtype=np.int8)
    for half in (1,2):
        indices=np.flatnonzero(period==half)
        if not len(indices) or not np.all(np.diff(indices)==1):
            raise ValueError('External half must be contiguous')
        tracking={'px':px[indices],'py':py[indices],'bx':bx[indices],'by':by[indices],'team':team}
        dead,h,a,names=feature_matrices(tracking,10,(length,width))
        halves.append({'half':half,'dead':dead,'home':h,'away':a,'truth':labels[indices],'names':names})
    cache_root.mkdir(parents=True,exist_ok=True)
    joblib.dump(halves,cache,compress=1)
    return halves,hashes


def possession_weights(labels):
    starts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    weights=np.ones(len(labels))
    for start,stop in zip(starts[:-1],starts[1:]):
        if labels[start] in ('H','A'):
            weights[start:stop]=np.clip(80/(stop-start),.25,4)
    return weights


def summarize(items, bundle, settings):
    scores=[]
    for item in items:
        pred=decode_intervals(item['pd'],item['ph'],10,settings)
        scores.append({'match':item['match'],'half':item['half'],**_score(item['truth'],pred)})
    return {'mean_spell_f1':float(np.mean([s['spell_f1'] for s in scores])),
            'mean_macro_f1':float(np.mean([s['macro_f1'] for s in scores])), 'halves':scores}


def baseline_scores(root, matches, cache_root):
    from candidate.train_external import load_match
    from candidate.external_model import load_model, probabilities, decode, postprocess
    model=load_model(Path(__file__).with_name('external_config.json'))
    scores=[]
    for match in matches:
        features,_,period,_=load_match(root/match)
        references,_=external_halves(root/match,cache_root)
        probs=probabilities(features,model)
        for half in (1,2):
            idx=np.flatnonzero(period==half)
            raw=decode(probs[idx],model['team_penalty'],model['dead_penalty'])
            pred=postprocess(raw,round(model['team_bridge_s']*10),round(model['dead_bridge_s']*10))
            truth=references[half-1]['truth']
            if len(truth)!=len(pred):
                raise ValueError('Baseline and candidate external timelines differ')
            scores.append({'match':match,'half':half,**_score(truth,pred)})
    return {'mean_spell_f1':float(np.mean([s['spell_f1'] for s in scores])),
            'mean_macro_f1':float(np.mean([s['macro_f1'] for s in scores])),'halves':scores}


def train(root,output,report_path,cache_root):
    started=time.time()
    ids=sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    fit=[x for x in ids if x not in TEST+TUNE]
    if len(fit)!=14 or set(fit)&set(TUNE+TEST):
        raise ValueError('Expected fixed disjoint 14/3/3 match split')
    manifest={}; dx=[]; dy=[]; tx=[]; ty=[]; tw=[]; names=None
    fit_counts={'H':0,'A':0,'D':0}
    for match in fit:
        halves,manifest[match]=external_halves(root/match,cache_root)
        for item in halves:
            names=item['names']
            for label in fit_counts:
                fit_counts[label]+=int(np.sum(item['truth']==label))
            select=np.arange(0,len(item['truth']),5)  # 2 Hz, external tracking is 10 Hz.
            lab=item['truth'][select]
            dx.append(item['dead'][select]); dy.append(lab=='D')
            alive=lab!='D'; home=item['home'][select][alive]; away=item['away'][select][alive]
            tx.extend((home,away)); ty.extend((lab[alive]=='H',lab[alive]=='A'))
            weight=possession_weights(item['truth'])[select][alive]; tw.extend((weight,weight))
        print('Prepared fit match',match,flush=True)
    params=dict(max_iter=400,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=200,
                l2_regularization=1.,early_stopping=False,random_state=2026)
    print('Fitting two boosted classifiers',flush=True)
    with threadpool_limits(limits=6):
        dead=HistGradientBoostingClassifier(**params).fit(np.concatenate(dx),np.concatenate(dy))
        del dx,dy
        team=HistGradientBoostingClassifier(**params).fit(np.concatenate(tx),np.concatenate(ty),sample_weight=np.concatenate(tw))
        del tx,ty,tw
    bundle={'feature_version':FEATURE_VERSION,'feature_names':names,'dead':dead,'team':team,'fit_matches':fit}
    tune=[]
    for match in TUNE:
        halves,manifest[match]=external_halves(root/match,cache_root)
        for item in halves:
            pd,ph=predict_probabilities(bundle,item['dead'],item['home'],item['away'])
            tune.append({'match':match,'half':item['half'],'truth':item['truth'],'pd':pd,'ph':ph})
    options=[]
    for team_weight,switch,stop in itertools.product((1.,2.),(.5,1.,2.,4.),(.5,1.,2.)):
        # Prefer a direct turnover to a fake stop/restart route.
        if switch>1.5*stop:
            continue
        settings=dict(dead_weight=1.,team_weight=team_weight,switch_seconds=switch,
                      stop_seconds=stop,minimum_dead_seconds=1.)
        result=summarize(tune,bundle,settings)
        options.append((result['mean_spell_f1'],result['mean_macro_f1'],settings,result))
        print('External decoder candidate',len(options),round(result['mean_spell_f1'],4),flush=True)
    best=max(options,key=lambda x:(x[0],x[1]))
    baseline_tune=baseline_scores(root,TUNE,cache_root)
    print('External selection baseline/new',baseline_tune['mean_spell_f1'],best[0],flush=True)
    if best[0]<=baseline_tune['mean_spell_f1']:
        raise RuntimeError('New candidate did not improve external selection score; test set remains unopened')
    bundle['decoder']=best[2]
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output,compress=3)
    # The selected model and decoder are frozen BEFORE the test set is opened.
    tests=[]
    for match in TEST:
        halves,manifest[match]=external_halves(root/match,cache_root)
        for item in halves:
            pd,ph=predict_probabilities(bundle,item['dead'],item['home'],item['away'])
            tests.append({'match':match,'half':item['half'],'truth':item['truth'],'pd':pd,'ph':ph})
    result={'source':'https://github.com/SkillCorner/opendata','revision':'4340d274572876239c154c90bc507a9b3250a656',
            'license':'MIT','fit_matches':fit,'tune_matches':list(TUNE),'test_matches':list(TEST),
            'classifier_parameters':params,'decoder':best[2], 'feature_names':names,
            'feature_version':FEATURE_VERSION,'fit_label_counts_10hz':fit_counts,
            'input_counts':{'dead':dead.n_features_in_,'team':team.n_features_in_},
            'tuning':best[3],'baseline_tuning':baseline_tune,'test':summarize(tests,bundle,best[2]),
            'baseline_test':baseline_scores(root,TEST,cache_root),
            'decoder_candidates':len(options),'files_sha256':manifest,
            'reference_policy':'Both models scored against the same period-clipped provider phases',
            'model_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Frozen external test',result['test']['mean_spell_f1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('data/external/skillcorner-opendata/data/matches'))
    parser.add_argument('--output',type=Path,default=Path('candidate/boosted_model.joblib'))
    parser.add_argument('--report',type=Path,default=Path('candidate/boosted_validation.json'))
    parser.add_argument('--cache',type=Path,default=Path('data/cache/boosted_external'))
    a=parser.parse_args(); train(a.root,a.output,a.report,a.cache)
