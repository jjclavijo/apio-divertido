import pytest

def pytest_addoption(parser):
    parser.addoption("--stat_properties", action="store_true",
                     help="run the tests only in case of that command line (marked with marker @stat_properties)")

def pytest_runtest_setup(item):
    if 'stat_properties' in item.keywords and not item.config.getoption("--stat_properties"):
        pytest.skip("need --stat_properties option to run this test")
