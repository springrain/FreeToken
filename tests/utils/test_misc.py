"""utils/misc shard-geometry helpers."""

import pytest


def test_div_even_divides_exactly():
    from freetoken.utils.misc import div_even

    assert div_even(64, 2) == 32
    assert div_even(64, 1) == 64


def test_div_even_raises_with_the_numbers_on_bad_shard():
    from freetoken.utils.misc import div_even

    with pytest.raises(ValueError, match="63 % 2"):
        div_even(63, 2)


def test_div_even_allows_kv_replication_and_checks_it():
    from freetoken.utils.misc import div_even

    assert div_even(2, 4, allow_replicate=True) == 1
    with pytest.raises(ValueError, match="replication"):
        div_even(2, 3, allow_replicate=True)
