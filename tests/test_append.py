import pytest
import numpy as np
from pytdigest import TDigest

from tdrust import Digest

@pytest.fixture(scope="session")
def array(pytestconfig) -> np.ndarray:
    size = pytestconfig.getoption("size")
    return np.random.normal(size=int(size))

def test_tdrust_append(benchmark, array):
    
    digest = Digest(delta=100.)

    benchmark(lambda digest, array: digest.append(array, weight=1.0), digest, array)

def test_pytdigest_append(benchmark, array):
    
    digest = TDigest()

    benchmark(lambda digest, array: digest.compute(array), digest, array)