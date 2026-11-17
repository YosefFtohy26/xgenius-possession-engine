import itertools
import numpy as np
import pytest
from candidate.boosted_model import feature_matrices, decode_intervals, _event_distance


def test_team_swap_exchanges_possession_design_and_preserves_dead_design():
    tracking={'px':np.tile([.2,.7],(30,1)), 'py':np.full((30,2),.5),
              'bx':np.linspace(.2,.4,30), 'by':np.full(30,.5), 'team':np.array([0,1])}
    dead,home,away,names=feature_matrices(tracking,10)
    swapped={**tracking,'team':1-tracking['team']}
    d2,h2,a2,n2=feature_matrices(swapped,10)
    np.testing.assert_array_equal(dead,d2)
    np.testing.assert_array_equal(home,a2)
    np.testing.assert_array_equal(away,h2)
    assert names==n2
    assert np.isfinite(dead).all() and np.isfinite(home).all()


def test_missing_ball_features_remain_finite():
    tracking={'px':np.full((20,2),np.nan),'py':np.full((20,2),np.nan),
              'bx':np.full(20,np.nan),'by':np.full(20,np.nan),'team':np.array([0,1])}
    for matrix in feature_matrices(tracking,10)[:3]:
        assert np.isfinite(matrix).all()
    before,after=_event_distance(np.zeros(20,dtype=bool),10)
    assert np.all(before==30) and np.all(after==30)


def test_unused_roster_slots_do_not_change_features():
    tracking={'px':np.column_stack([np.linspace(.2,.3,30),np.linspace(.7,.8,30)]),
              'py':np.full((30,2),.5),'bx':np.full(30,.25),'by':np.full(30,.5),
              'team':np.array([0,1])}
    enlarged={**tracking,'px':np.column_stack([tracking['px'],np.full((30,2),np.nan)]),
              'py':np.column_stack([tracking['py'],np.full((30,2),np.nan)]),
              'team':np.array([0,1,0,1])}
    first=feature_matrices(tracking,10);second=feature_matrices(enlarged,10)
    for a,b in zip(first[:3],second[:3]):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('seed',[1,9,42])
def test_duration_decoder_matches_exhaustive_search(seed):
    rng=np.random.default_rng(seed)
    pd=rng.uniform(.1,.9,7); ph=rng.uniform(.1,.9,7)
    prm={'dead_weight':1.,'team_weight':1.,'switch_seconds':.7,
         'stop_seconds':.6,'minimum_dead_seconds':2.}
    result=decode_intervals(pd,ph,1,prm)
    values=np.column_stack([np.log1p(-pd)+np.log(ph),np.log1p(-pd)+np.log1p(-ph),np.log(pd)])
    def value(path):
        total=values[np.arange(7),path].sum()
        for left,right in zip(path[:-1],path[1:]):
            if left!=right:
                total-=.6 if 2 in (left,right) else .7
        return total
    best=-np.inf
    for path in itertools.product(range(3),repeat=7):
        if any(path[t]==2 and path[t-1]!=2 and path[t+1]!=2 for t in range(1,6)):
            continue
        best=max(best,value(path))
    assert value(np.array([{'H':0,'A':1,'D':2}[x] for x in result]))==pytest.approx(best)
