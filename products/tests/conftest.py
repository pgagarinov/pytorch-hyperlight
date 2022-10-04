import pytest
from xdist import is_xdist_controller
from xdist.scheduler import LoadScopeScheduling

def pytest_configure(config):
    config.pluginmanager.register(XDistSerialPlugin())

class XDistSerialPlugin:
    def __init__(self):
        self._nodes = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection(self, session):
        if is_xdist_controller(session):
            self._nodes = {
                item.nodeid: item
                for item in session.perform_collect(None)
            }
            return True

    def pytest_xdist_make_scheduler(self, config, log):
        return SerialScheduling(config, log, nodes=self._nodes)


class SerialScheduling(LoadScopeScheduling):
    def __init__(self, config, log, *, nodes):
        super().__init__(config, log)
        self._nodes = nodes

    def _split_scope(self, nodeid):
        node = self._nodes[nodeid]
        if node.get_closest_marker("serial"):
            # put all `@pytest.mark.serial` tests in same scope, to
            # ensure they're all run in the same worker
            return "__serial__"

        # otherwise, each test is in its own scope
        return nodeid

