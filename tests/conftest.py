import pytest


@pytest.fixture(autouse=True)
def _default_quant_backend():
    """Tests that install kernel requests must not leak them into the next test."""
    yield
    from freetoken.layers.quantization import QuantBackend, set_quant_backend

    set_quant_backend(QuantBackend())


@pytest.fixture(autouse=True)
def _restore_tp_info_after_test():
    """set_tp_info is process-global and one-shot: a test that changes the TP geometry
    must not poison later tests, while module/session fixtures that set it once must
    keep their state (a blind reset would break every test after the first in a module)."""
    from freetoken.distributed.info import reset_tp_info, set_tp_info, try_get_tp_info

    prev = try_get_tp_info()
    yield
    if try_get_tp_info() is not prev:
        reset_tp_info()
        if prev is not None:
            set_tp_info(prev.rank, prev.size)
