"""Integer form of the classical alias construction; no floating arithmetic."""

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Callable, Iterable

MAX_ITEMS = 100_000
MAX_WEIGHT_BITS = 4096


class SamplingError(RuntimeError):
    """A bit source violated its contract or exhausted the rejection budget."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python integer >= {minimum}")
    return value


def _weights(values: Iterable[int]):
    # Bound iteration too: an infinite input iterator cannot fill memory forever.
    weights = []
    for value in values:
        if len(weights) >= MAX_ITEMS:
            raise ValueError(f"at most {MAX_ITEMS} weights are supported")
        value = _integer(value, "weight")
        if value.bit_length() > MAX_WEIGHT_BITS:
            raise ValueError(f"weights are limited to {MAX_WEIGHT_BITS} bits")
        weights.append(value)
    if not weights or not any(weights):
        raise ValueError("weights must be nonempty with a positive total")
    return tuple(weights)


def uniform_below(bound: int, getrandbits: Callable[[int], int], *, max_attempts=1024):
    """Uniform in range(bound), conditional on success under iid uniform bits.

    Rejection, never modulo, maps binary words to a non-power-of-two range.
    The finite budget raises rather than substituting a biased fallback value.
    The bit-source distribution cannot be validated by this function.
    """
    _integer(bound, "bound", 1)
    _integer(max_attempts, "max_attempts", 1)
    if not callable(getrandbits):
        raise ValueError("getrandbits must be callable")
    if bound == 1:
        return 0
    bits = (bound-1).bit_length()
    for _ in range(max_attempts):
        word = getrandbits(bits)
        if type(word) is not int or not 0 <= word < (1 << bits):
            raise SamplingError("getrandbits returned an invalid word")
        if word < bound:
            return word
    raise SamplingError("rejection budget exhausted; no sample returned")


@dataclass(frozen=True, slots=True)
class AliasTable:
    """Immutable, validated alias table with a common integer column height.

    Input weights are preserved. Their GCD is removed internally so scaling
    all weights by an integer does not waste additional random bits.
    Direct construction and JSON loading both verify the full output masses.
    """

    weights: tuple[int, ...]
    cutoffs: tuple[int, ...]
    aliases: tuple[int, ...]
    total: int = field(init=False)
    unit: int = field(init=False)

    def __post_init__(self):
        weights = _weights(self.weights)
        cutoffs, aliases = tuple(self.cutoffs), tuple(self.aliases)
        n = len(weights)
        if len(cutoffs) != n or len(aliases) != n:
            raise ValueError("table arrays must have the same length")
        unit = gcd(*weights)
        total = sum(weights)//unit
        counts = [0]*n
        for j, (cut, alias) in enumerate(zip(cutoffs, aliases)):
            _integer(cut, "cutoff")
            _integer(alias, "alias")
            if cut > total or alias >= n:
                raise ValueError("table cutoff or alias out of range")
            counts[j] += cut
            counts[alias] += total-cut
        if counts != [n*(w//unit) for w in weights]:
            raise ValueError("table does not reproduce the supplied weights")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "cutoffs", cutoffs)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "total", total)

    @property
    def cells(self):
        return len(self.weights)*self.total

    def cell_counts(self):
        """Count the implicit lattice exactly without enumerating its cells."""
        counts = [0]*len(self.weights)
        for j, (cut, alias) in enumerate(zip(self.cutoffs, self.aliases)):
            counts[j] += cut
            counts[alias] += self.total-cut
        return tuple(counts)

    def probabilities(self):
        return tuple(Fraction(count, self.cells) for count in self.cell_counts())

    def decode(self, ticket: int):
        """Map one uniform lattice ticket to its output category index."""
        _integer(ticket, "ticket")
        if ticket >= self.cells:
            raise ValueError("ticket outside table lattice")
        column, row = divmod(ticket, self.total)
        return column if row < self.cutoffs[column] else self.aliases[column]

    def draw(self, getrandbits: Callable[[int], int], *, max_attempts=1024):
        ticket = uniform_below(self.cells, getrandbits, max_attempts=max_attempts)
        return self.decode(ticket)


def build(weights: Iterable[int]):
    """Construct an alias table in O(n) integer operations and O(n) storage.

    This is an operation-count statement, not a constant-cost bigint model.
    The small/large worklists contain indices with mass below/above one column.
    """
    weights = _weights(weights)
    n = len(weights)
    unit = gcd(*weights)
    total = sum(weights)//unit
    mass = [n*(w//unit) for w in weights]
    small = [j for j, x in enumerate(mass) if x < total]
    large = [j for j, x in enumerate(mass) if x >= total]
    cutoffs, aliases = [total]*n, list(range(n))
    while small and large:
        j, donor = small.pop(), large.pop()
        cutoffs[j], aliases[j] = mass[j], donor
        mass[donor] -= total-mass[j]
        (small if mass[donor] < total else large).append(donor)
    # Exact integer arithmetic implies every remaining column is full.
    for j in small+large:
        if mass[j] != total:
            raise ArithmeticError("alias construction mass invariant failed")
    return AliasTable(weights, tuple(cutoffs), tuple(aliases))
