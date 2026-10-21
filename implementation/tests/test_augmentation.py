import numpy as np
import pytest
from ctee.augmentation import randomize,development_augmentation
from ctee.simulator import sample_circuit,episode


def test_sensor_configuration_is_shared_and_randomization_reproducible():
    circuit=sample_circuit('R',0,np.random.default_rng(22));waves,_=episode(circuit,'CZ')
    start=waves[0];a,sa,meta=randomize(waves,start,42);b,sb,other=randomize(waves,start,42)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(sa,sb)
    assert meta==other and 12<=meta['snr_db']<=35
    assert a.shape==waves.shape and np.isfinite(a).all()
    assert not np.array_equal(a,waves)


def test_augmentation_never_requests_test_or_calibration_circuits():
    circuit=sample_circuit('R',0,np.random.default_rng(9)).to_dict()
    manifest={'groups':[dict(circuit,split='train'),dict(circuit,split='test')]}
    data,ledger=development_augmentation(manifest,'train',replicas=1)
    assert set(data['group'])=={0} and len(ledger)==3
    assert set(data['split'])=={'train'}
    for split in ['test','calibration']:
        with pytest.raises(ValueError,match='development'):development_augmentation(manifest,split)
