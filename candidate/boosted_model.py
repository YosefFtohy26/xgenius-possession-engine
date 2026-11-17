"""External-data two-stage possession model with half-local temporal features.

Original implementation; no match-specific observations or labels enter features.
"""
from __future__ import annotations

import warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits

FEATURE_VERSION = 3


def _mean(a, seconds, fps):
    size = max(1, round(seconds * fps))
    return uniform_filter1d(np.asarray(a, dtype=np.float64), size=size, axis=0, mode='nearest')


def _event_distance(event, fps):
    n = len(event)
    index = np.arange(n)
    before = np.maximum.accumulate(np.where(event, index, -100000))
    after = np.minimum.accumulate(np.where(event, index, n + 100000)[::-1])[::-1]
    return np.minimum((index-before)/fps, 30), np.minimum((after-index)/fps, 30)


def _fill_ball(x, y, fps):
    x, y = x.copy(), y.copy()
    seen = np.isfinite(x) & np.isfinite(y)
    anchors = np.flatnonzero(seen)
    for left, right in zip(anchors[:-1], anchors[1:]):
        if 1 < right-left <= round(0.4*fps)+1:
            # Bound both displacement and gap duration to avoid bridging a bad observation.
            if np.hypot(x[right]-x[left], y[right]-y[left]) * fps/(right-left) <= 60:
                x[left:right+1] = np.linspace(x[left], x[right], right-left+1)
                y[left:right+1] = np.linspace(y[left], y[right], right-left+1)
    return x, y, seen


def feature_matrices(tracking, fps, pitch=(105., 68.)):
    """Return symmetric dead/live designs and home/away team designs for ONE half."""
    length, width = pitch
    px = np.asarray(tracking['px']) * length
    py = np.asarray(tracking['py']) * width
    bx, by, original_seen = _fill_ball(np.asarray(tracking['bx'])*length,
                                      np.asarray(tracking['by'])*width, fps)
    n = len(bx)
    seen = np.isfinite(bx) & np.isfinite(by)
    ball_speed = np.zeros(n)
    if n > 1:
        adjacent = seen[1:] & seen[:-1]
        ball_speed[1:] = np.where(adjacent, np.hypot(np.diff(bx), np.diff(by))*fps, 0)
    ball_speed = np.minimum(ball_speed, 60)
    player_speed = np.zeros_like(px)
    if n > 1:
        player_speed[1:] = np.hypot(np.diff(px,axis=0), np.diff(py,axis=0))*fps
    player_speed = np.clip(player_speed, 0, 12)
    valid = np.isfinite(px) & np.isfinite(py)
    distances = np.hypot(px-bx[:,None], py-by[:,None])
    distances[~valid | ~seen[:,None]] = np.inf
    teams = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for team in (0, 1):
            select = np.asarray(tracking['team']) == team
            d = distances[:,select]
            tx, ty, speed = px[:,select], py[:,select], player_speed[:,select]
            near = np.minimum(d.min(axis=1), 25)
            speed_valid = np.isfinite(speed) & np.isfinite(tx) & np.isfinite(ty)
            speed_count = np.maximum(1,speed_valid.sum(axis=1))
            avg_speed = np.where(speed_valid,speed,0).sum(axis=1)/speed_count
            fast_fraction = ((speed>3)&speed_valid).sum(axis=1)/speed_count
            cx, cy = np.nanmean(tx,axis=1), np.nanmean(ty,axis=1)
            teams.append(dict(distance=near, near2=(d<2).sum(axis=1), near5=(d<5).sum(axis=1),
                              proximity=np.exp(-0.5*(d/3)**2).sum(axis=1),
                              speed=avg_speed, fast=fast_fraction,
                              spread_x=np.nanstd(tx,axis=1), spread_y=np.nanstd(ty,axis=1),
                              ball_cx=np.abs(bx-cx), ball_cy=np.abs(by-cy)))
    for own, opp in ((teams[0],teams[1]),(teams[1],teams[0])):
        for name in own:
            own[name] = np.nan_to_num(own[name],nan=0,posinf=25,neginf=-25)
        own['advantage'] = opp['distance']-own['distance']
        own['control'] = ((own['distance']<1.5)&(own['advantage']>.5)&(ball_speed<12)).astype(float)
        since, until = _event_distance(own['control']>0, fps)
        own['since_control'], own['until_control'] = since, until
        for name in ('distance','advantage','control','speed','proximity'):
            for window in (.5, 2., 5.):
                own[f'{name}_mean_{window}'] = _mean(own[name], window, fps)
    outside = seen & ((bx<0)|(bx>length)|(by<0)|(by>width))
    shared = dict(ball_seen=original_seen.astype(float), ball_available=seen.astype(float),
                  ball_speed=ball_speed, ball_out=outside.astype(float),
                  ball_still=(seen&(ball_speed<1)).astype(float),
                  touchline=np.minimum(np.abs(by),np.abs(by-width)),
                  goalline=np.minimum(np.abs(bx),np.abs(bx-length)),
                  centre=np.hypot(bx-length/2,by-width/2),
                  min_distance=np.minimum(teams[0]['distance'],teams[1]['distance']),
                  any_control=np.maximum(teams[0]['control'],teams[1]['control']))
    for name in ('ball_seen','ball_speed','ball_out','ball_still','any_control'):
        for window in (.5,2.,5.):
            shared[f'{name}_mean_{window}'] = _mean(shared[name],window,fps)
    for name, event in [('seen',original_seen),('out',outside),('control',shared['any_control']>0)]:
        shared[f'since_{name}'], shared[f'until_{name}'] = _event_distance(event,fps)
    team_names, shared_names = sorted(teams[0]), sorted(shared)
    def stack(values,names):
        return np.nan_to_num(np.column_stack([values[k] for k in names]),nan=0,posinf=30,neginf=-30).astype(np.float32)
    h, a, s = stack(teams[0],team_names), stack(teams[1],team_names), stack(shared,shared_names)
    # Invariant dead-ball features; exactly swapped team features for possession.
    dead = np.column_stack([s, np.minimum(h,a), np.maximum(h,a)]).astype(np.float32)
    home = np.column_stack([s,h,a]).astype(np.float32)
    away = np.column_stack([s,a,h]).astype(np.float32)
    return dead, home, away, {'shared':shared_names,'team':team_names}


def predict_probabilities(bundle, dead, home, away):
    with threadpool_limits(limits=1):
        p_dead = bundle['dead'].predict_proba(dead)[:,1]
        p_home = (bundle['team'].predict_proba(home)[:,1] + 1-bundle['team'].predict_proba(away)[:,1])/2
    return p_dead, p_home


def decode_intervals(p_dead, p_home, fps, settings):
    """Exact duration-constrained 3-state Viterbi in O(T), with half-edge exemptions."""
    pd = np.clip(p_dead,1e-5,1-1e-5)
    ph = np.clip(p_home,1e-5,1-1e-5)
    alive = settings['dead_weight']*np.log1p(-pd)
    scores = np.column_stack([alive+settings['team_weight']*np.log(ph),
                              alive+settings['team_weight']*np.log1p(-ph),
                              settings['dead_weight']*np.log(pd)])
    n = len(scores)
    if not n:
        return np.empty(0,dtype='<U1')
    switch = settings['switch_seconds']*fps
    stop = settings['stop_seconds']*fps
    k = max(1,round(settings['minimum_dead_seconds']*fps))
    summed = np.concatenate([[0.],np.cumsum(scores[:,2])])
    value = np.empty((n,3)); previous = np.zeros((n,2),dtype=np.int8)
    starts = np.full(n,-1,dtype=np.int32)
    entry = np.full(n,-np.inf); entry_state = np.zeros(n,dtype=np.int8)
    value[0] = scores[0]
    for t in range(1,n):
        h,a,d = value[t-1]
        options_h = (h,a-switch,d-stop)
        options_a = (h-switch,a,d-stop)
        ih = 0 if options_h[0]>=max(options_h[1:]) else (1 if options_h[1]>=options_h[2] else 2)
        ia = 1 if options_a[1]>=max(options_a[0],options_a[2]) else (0 if options_a[0]>=options_a[2] else 2)
        previous[t] = ih,ia
        value[t,0],value[t,1] = options_h[ih]+scores[t,0],options_a[ia]+scores[t,1]
        entry_state[t] = 0 if h>=a else 1
        entry[t] = max(h,a)-stop
        value[t,2] = d+scores[t,2]
        start = t-k+1
        if start>=1:
            candidate = entry[start]+summed[t+1]-summed[start]
            if candidate>value[t,2]:
                value[t,2],starts[t] = candidate,start
    state = int(value[-1].argmax()); best = value[-1,state]
    path = np.empty(n,dtype=np.int8)
    t = n-1
    for start in range(max(1,n-k+1),n):
        candidate = entry[start]+summed[n]-summed[start]
        if candidate>best:
            best,state,t = candidate,int(entry_state[start]),start-1
            path[start:] = 2
    while t>=0:
        if state==2 and starts[t]>=1:
            start = int(starts[t]); path[start:t+1] = 2
            state,t = int(entry_state[start]),start-1
        else:
            path[t] = state
            if state<2:
                state = int(previous[t,state])
            t-=1
    return np.array(['H','A','D'])[path]


def load_bundle(path):
    model = joblib.load(Path(path))
    if model['feature_version']!=FEATURE_VERSION or any(not str(x).isdigit() for x in model['fit_matches']):
        raise ValueError('Invalid external model provenance or feature version')
    return model


def predict_match(tracking, model, fps, spans, pitch=(105.,68.)):
    labels = np.full(len(tracking['bx']),'X',dtype='<U1')
    for start,stop in spans:
        piece = {key:(value if key=='team' else value[start:stop]) for key,value in tracking.items()
                 if key in ('px','py','ps','bx','by','team')}
        dead,home,away,names = feature_matrices(piece,fps,pitch)
        if names != model['feature_names']:
            raise ValueError('Model feature names differ from inference features')
        pd,ph = predict_probabilities(model,dead,home,away)
        labels[start:stop] = decode_intervals(pd,ph,fps,model['decoder'])
    return labels
