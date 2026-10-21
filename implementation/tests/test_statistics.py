import math
import numpy as np
import pytest
from ctee.statistics import beta_regularized,f_survival,gamma_survival,friedman,wilcoxon_exact,compare_panels

def test_distribution_tails_match_closed_forms():
    for x in [.01,.2,.7,.99]:
        assert beta_regularized(1,1,x)==pytest.approx(x,abs=1e-12)
        assert beta_regularized(.5,.5,x)==pytest.approx(2*math.asin(math.sqrt(x))/math.pi,abs=1e-12)
    for x in [.1,1.,5.,20.]:
        assert f_survival(x,2,2)==pytest.approx(1/(1+x),abs=1e-12)
        assert gamma_survival(1.,x)==pytest.approx(math.exp(-x),abs=1e-12)
        assert gamma_survival(.5,x)==pytest.approx(math.erfc(math.sqrt(x)),abs=1e-12)

def test_paired_rank_tests_from_hand_examples():
    statistic,p=friedman(np.tile([1.,2.,3.],(4,1)))
    assert statistic==pytest.approx(8.) and p==pytest.approx(math.exp(-4))
    assert wilcoxon_exact([1.,2.,3.])==.25
    assert wilcoxon_exact([0.,0.,0.])==1.

def test_tied_models_are_not_significantly_different():
    result=compare_panels(np.ones((10,3)),['a','b','c'])
    assert result['friedman_p']==result['ANOVA_GG_p']==1.
    assert all(x['holm_adjusted_p']==1 for x in result['posthoc_paired_wilcoxon_holm'])
