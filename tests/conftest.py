def pytest_addoption(parser):
    parser.addoption("--size", action="store", default="array size")