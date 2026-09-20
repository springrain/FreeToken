"""EngineConfig.distributed_addr env override (multi-engine host sharing)."""


def test_distributed_addr_defaults_to_localhost_2333(monkeypatch):
    from freetoken.engine.config import EngineConfig

    monkeypatch.delenv("FREETOKEN_DIST_ADDR", raising=False)
    fake = object.__new__(EngineConfig)
    assert EngineConfig.distributed_addr.fget(fake) == "tcp://127.0.0.1:2333"


def test_distributed_addr_honors_freetoken_dist_addr(monkeypatch):
    from freetoken.engine.config import EngineConfig

    monkeypatch.setenv("FREETOKEN_DIST_ADDR", "tcp://127.0.0.1:29999")
    fake = object.__new__(EngineConfig)
    assert EngineConfig.distributed_addr.fget(fake) == "tcp://127.0.0.1:29999"
