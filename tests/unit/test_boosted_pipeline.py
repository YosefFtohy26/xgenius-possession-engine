import numpy as np
from candidate.boosted_pipeline import apply_pre_kickoff_dead


def test_second_half_pre_kickoff_is_dead():
    labels=np.full(199319,"H",dtype="<U1")
    result=apply_pre_kickoff_dead(labels)
    assert np.all(result[110808:113742]=="D")
    assert result[110807]=="H"
    assert result[113742]=="H"
    assert np.all(result[:110808]=="H")
    assert np.all(result[113742:]=="H")
