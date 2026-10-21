import numpy as np
import pytest
from ctee.numerics import qr_factor,least_squares,nonnegative_fit,fft_radix2,regression_slope

def test_qr_and_least_squares_from_known_coefficients():
    a=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.]])
    q,r=qr_factor(a)
    np.testing.assert_allclose(q@r,a,atol=1e-13)
    np.testing.assert_allclose(q.T@q,np.eye(2),atol=1e-13)
    np.testing.assert_allclose(least_squares(a,a@np.array([2.,3.])),[2.,3.],atol=1e-12)

def test_rank_deficiency_is_explicit():
    with pytest.raises(ValueError):qr_factor(np.ones((4,2)))

def test_nonnegative_fit_checks_boundary_solution():
    x,error=nonnegative_fit(np.eye(3),np.array([2.,-3.,4.]))
    np.testing.assert_allclose(x,[2.,0.,4.]);assert error==3.

def test_fft_matches_independent_direct_dft():
    x=np.random.default_rng(3).normal(size=13);padded=np.r_[x,np.zeros(3)]
    direct=np.exp(-2j*np.pi*np.outer(np.arange(16),np.arange(16))/16)@padded
    np.testing.assert_allclose(fft_radix2(x),direct,atol=1e-13)

def test_fourier_dc_and_regression_slope():
    np.testing.assert_allclose(fft_radix2(np.ones(8)),[8,0,0,0,0,0,0,0],atol=1e-12)
    assert regression_slope([1.,2.,4.],[3.,5.,9.])==2.
