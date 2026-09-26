"""Fast checks for signed ranking, matching, endpoint bins, and disjoint selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_pca import auc, correlations, matched_auc, top_disjoint, tuning


def test_auc_signed_and_ties():
    scores = np.array([[-2., 1.], [-1., 1.], [-1., 0.], [3., 0.]])
    right = np.array([False, False, True, True])
    expected = [roc_auc_score(right, scores[:,i]) for i in range(2)]
    np.testing.assert_allclose(auc(scores,right), expected)
    np.testing.assert_allclose(auc(-scores,right),1-auc(scores,right))


def test_matching_weights_and_exclusion():
    # Two qualifying bins (weights10/12), one ineligible (9 in one direction).
    directions=np.array([-1]*10+[1]*15+[-1]*12+[1]*12+[-1]*9+[1]*20)
    pos=np.array([.21]*25+[.31]*24+[.41]*29)
    values=np.concatenate([directions[:25],-directions[25:49],directions[49:]]).astype(float)[:,None]
    metadata=pd.DataFrame(dict(position=pos,direction=directions,ramp_clean=True))
    result, records=matched_auc(values,metadata,40,"ramp_clean")
    np.testing.assert_allclose(result,10/22)
    assert [r["weight"] for r in records]==[10,12]


def test_disjoint_signed_all_candidates():
    starts=np.arange(5)*50
    scores=np.array([-3.,-2.,-1.,-4.,-5.])
    chosen=top_disjoint(scores,starts,np.ones(5,dtype=bool),3)
    assert chosen.tolist()==[2,0,4]
    assert np.all(np.diff(np.sort(starts[chosen]))>=100)


def test_tuning_endpoint_and_signed_correlations():
    position=np.concatenate([(np.arange(40)+.5)/40,[1.]])
    values=np.arange(41,dtype=float)[:,None]
    result=tuning(values,position)
    assert result.shape==(1,40)
    assert result[0,-1]==39.5
    np.testing.assert_allclose(correlations(result,-result),[[-1.]])
