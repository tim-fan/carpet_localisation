def pytest_addoption(parser):
    parser.addoption("--show_plot", action="store", default=False, type=bool)