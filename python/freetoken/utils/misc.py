from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers, requiring exact division (tensor parallelism shard
    geometry). If allow_replicate=True, allows b > a when b % a == 0, returning 1
    (KV heads replicate across TP ranks instead of sharding)."""
    if allow_replicate and b > a:
        if b % a != 0:
            raise ValueError(
                f"KV head replication requires tp size divisible by num KV heads, got tp={b}, kv_heads={a}"
            )
        return 1
    if a % b != 0:
        raise ValueError(f"tensor parallel shard requires exact division, got {a} % {b} != 0")
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


def mem_GB(size: int) -> str:
    """Format a byte count as a human-readable GiB string for logs."""
    return f"{size / (1024**3):.2f} GiB"


class Unset:
    pass


UNSET = Unset()
