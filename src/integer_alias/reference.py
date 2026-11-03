"""Independent integer CDF decoder and finite-state reference experiments."""

from collections import Counter
from fractions import Fraction
from itertools import accumulate


def cdf_decode(weights, ticket):
    """Independent binary search on the original (unreduced) cumulative mass.

    Returns output index and the actual number of CDF comparisons.
    A different ticket space means draws are not coupled to alias draws.
    """
    if not weights or any(type(w) is not int or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("invalid integer weights")
    cumulative = tuple(accumulate(weights))
    return cdf_lookup(cumulative, ticket)


def cdf_lookup(cumulative, ticket):
    if type(ticket) is not int or not 0 <= ticket < cumulative[-1]:
        raise ValueError("ticket outside CDF")
    left, right, comparisons = 0, len(cumulative), 0
    while left < right:
        middle = (left+right)//2
        comparisons += 1
        if ticket < cumulative[middle]:
            right = middle
        else:
            left = middle+1
    return left, comparisons


def enumerate_cdf(weights, limit=100_000):
    total = sum(weights)
    if total > limit:
        raise ValueError("reference enumeration limit exceeded")
    counts = Counter(cdf_decode(weights, ticket)[0] for ticket in range(total))
    return tuple(Fraction(counts[j], total) for j in range(len(weights)))


def one_word_laws(table, bits):
    """Exhaustive modulo law and accepted-word rejection law, up to 20 bits."""
    if type(bits) is not int or not 0 <= bits <= 20 or (1 << bits) < table.cells:
        raise ValueError("need 0..20 bits with enough words to cover the lattice")
    modulo, accepted = Counter(), Counter()
    for word in range(1 << bits):
        modulo[table.decode(word % table.cells)] += 1
        if word < table.cells:
            accepted[table.decode(word)] += 1
    n = len(table.weights)
    return {"modulo": tuple(Fraction(modulo[j], 1 << bits) for j in range(n)),
            "rejection_given_acceptance": tuple(Fraction(accepted[j], table.cells) for j in range(n)),
            "acceptance": Fraction(table.cells, 1 << bits)}
