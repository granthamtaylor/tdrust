import pytest
import numpy as np
from pytdigest import TDigest

from tdrust import Digest

@pytest.fixture(scope="session")
def array(pytestconfig) -> np.ndarray:
    size = pytestconfig.getoption("size")
    return np.random.normal(size=int(size))

def test_tdrust_multithreaded_cdf(benchmark, array):
    
    digest = Digest(delta=100.)

    digest.append(array, weight=1.0)

    benchmark(lambda digest, array: digest.cdfs(array, multithreaded=True), digest, array)

def test_tdrust_cdf(benchmark, array):
    
    digest = Digest(delta=100.)

    digest.append(array, weight=1.0)

    benchmark(lambda digest, array: digest.cdfs(array, multithreaded=False), digest, array)

def test_pytdigest_cdf(benchmark, array):
    
    digest = TDigest()

    digest.compute(array)

    benchmark(lambda digest, array: digest.cdf(array), digest, array)