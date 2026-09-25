"""Enforce complete four-feature, equal-weight single-configuration selection."""
from experiments.churchland.sweeps.s10_endpoint_v1.langevinflow_feature_selection import (
    FEATURE_NAMES, select_configuration,
)


def test_no_three_feature_fallback():
    partial={name:dict(eligible=True,auroc=.9) for name in FEATURE_NAMES if name!='fast_target_specific'}
    assert select_configuration({'a':partial}) is None


def test_equal_weights_and_eligibility():
    a={name:dict(eligible=True,auroc=.8) for name in FEATURE_NAMES}
    b={name:dict(eligible=True,auroc=.9 if i<3 else .4) for i,name in enumerate(FEATURE_NAMES)}
    assert select_configuration({'a':a,'b':b})['config_id']=='a'
    a['recent_braking']['eligible']=False
    assert select_configuration({'a':a,'b':b})['config_id']=='b'
