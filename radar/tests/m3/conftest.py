import pytest

from radar.m3_sources.common import build_matcher, load_m3_config
from radar.m3_sources.fetching import reset_shared_fetcher


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Every test gets its own cache dir and a clean environment."""
    monkeypatch.setenv("RADAR_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("RADAR_MODE", "GITHUB_TOKEN", "RADAR_SEC_USER_AGENT", "RADAR_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    reset_shared_fetcher()
    yield
    reset_shared_fetcher()


@pytest.fixture(scope="session")
def cfg():
    return load_m3_config()


@pytest.fixture(scope="session")
def matcher(cfg):
    return build_matcher(cfg)[0]
