"""JavaScript integer semantics, reproduced exactly.

`env.js` generates its maps from a hand-rolled 32-bit PRNG. To make the same seed
give the same map in Python, the arithmetic has to match JS bit for bit: signed
32-bit wraparound on `|0`, unsigned shifts on `>>>`, and `Math.imul`'s truncated
32-bit multiply. Python's unbounded ints do none of that on their own.
"""

MASK32 = 0xFFFFFFFF


def u32(x: int) -> int:
    """ToUint32."""
    return x & MASK32


def i32(x: int) -> int:
    """ToInt32."""
    x &= MASK32
    return x - 0x100000000 if x & 0x80000000 else x


def imul(a: int, b: int) -> int:
    """Math.imul — C-like 32-bit signed multiply."""
    return i32(u32(a) * u32(b))


class JsRng:
    """The `rng(seed)` closure from env.js, step for step.

        a = a + 0x6D2B79F5 | 0
        t = Math.imul(a ^ a >>> 15, 1 | a)
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
        return ((t ^ t >>> 14) >>> 0) / 4294967296

    Every intermediate is coerced the way the JS operator would coerce it. The one
    `+` that is *not* an int32 op sums two int32s, so it stays exact in a double
    and needs no special care beyond the ToInt32 that the following `^` applies.
    """

    __slots__ = ("a",)

    def __init__(self, seed: int) -> None:
        self.a = u32(seed)

    def __call__(self) -> float:
        a = i32(u32(self.a) + 0x6D2B79F5)
        self.a = a
        t = imul(i32(u32(a) ^ (u32(a) >> 15)), i32(1 | u32(a)))
        t = i32(u32(t + imul(i32(u32(t) ^ (u32(t) >> 7)), i32(61 | u32(t)))) ^ u32(t))
        return (u32(t) ^ (u32(t) >> 14)) / 4294967296.0


def floor_mul(r: float, n: int) -> int:
    """`Math.floor(r() * n)` — r is in [0,1), so a plain truncation is enough."""
    return int(r * n)
