from collections import Counter
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import product
import random

import pytest

from integer_alias import AliasTable, SamplingError, build, uniform_below
from integer_alias.reference import cdf_decode, enumerate_cdf, one_word_laws


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_small_integer_weight_vector_against_independent_cdf(n):
    for weights in product(range(4), repeat=n):
        if not any(weights):
            continue
        table = build(weights)
        counts = Counter(table.decode(t) for t in range(table.cells))
        law = tuple(Fraction(counts[j], table.cells) for j in range(n))
        assert law == enumerate_cdf(weights)
        assert law == tuple(Fraction(w, sum(weights)) for w in weights)
        assert tuple(counts[j] for j in range(n)) == table.cell_counts()


def test_known_alias_geometry():
    table = build([1, 2, 3])
    assert table.cutoffs == (3, 6, 6)
    assert table.aliases == (2, 1, 2)
    assert [table.decode(t) for t in [0, 2, 3, 5, 6, 11, 12, 17]] == [0, 0, 2, 2, 1, 1, 2, 2]


def test_zero_mass_categories_are_never_selected():
    table = build([0, 2, 0, 1, 0])
    assert set(table.decode(t) for t in range(table.cells)) == {1, 3}
    assert [cdf_decode([0, 2, 0, 1, 0], t)[0] for t in range(3)] == [1, 1, 3]


def test_huge_weights_keep_rare_support_without_float_conversion():
    table = build([2**2048, 1, 0, 3])
    assert table.cell_counts() == (4*2**2048, 4, 0, 12)
    assert table.decode(table.total) == 1
    assert table.decode(3*table.total) == 3
    assert table.probabilities()[1] == Fraction(1, 2**2048+4)


@pytest.mark.parametrize("seed", range(10))
def test_random_large_masses_and_scaling(seed):
    rng = random.Random(seed)
    weights = [rng.randrange(10**15) for _ in range(25)]
    t = build(weights)
    scaled = build([37*w for w in weights])
    assert (t.total, t.cutoffs, t.aliases) == (scaled.total, scaled.cutoffs, scaled.aliases)
    assert t.probabilities() == tuple(Fraction(w, sum(weights)) for w in weights)


def test_singleton_draw_does_not_consume_bits():
    def forbidden(bits):
        raise AssertionError("no random bits needed")
    assert build([999999]).draw(forbidden) == 0


def test_immutable_snapshot_of_input_arrays():
    weights, cuts, aliases = [1, 1], [2, 2], [0, 1]
    table = AliasTable(weights, cuts, aliases)
    weights[0], cuts[0], aliases[0] = 999, 0, 1
    assert table.weights == (1, 1) and table.cutoffs == (2, 2)
    with pytest.raises(FrozenInstanceError):
        table.total = 1


@pytest.mark.parametrize("bound", [1, 2, 3, 5, 16, 17])
def test_one_attempt_uniform_below_enumerates_all_binary_words(bound):
    bits = (bound-1).bit_length()
    accepted = []
    for word in range(1 << bits):
        if word < bound:
            accepted.append(uniform_below(bound, lambda k: word, max_attempts=1))
        else:
            with pytest.raises(SamplingError):
                uniform_below(bound, lambda k: word, max_attempts=1)
    assert accepted == list(range(bound))


def test_rejection_consumes_new_words_and_has_no_modulo_fallback():
    words = iter([7, 6, 3])
    assert uniform_below(5, lambda k: next(words), max_attempts=3) == 3
    with pytest.raises(SamplingError, match="budget"):
        uniform_below(5, lambda k: 7, max_attempts=3)


def test_modulo_bias_and_rejection_conditional_law():
    table = build([1, 2, 3])
    law = one_word_laws(table, 5)
    assert law["modulo"] == (Fraction(3, 16), Fraction(3, 8), Fraction(7, 16))
    assert law["acceptance"] == Fraction(9, 16)
    assert law["rejection_given_acceptance"] == table.probabilities()
    assert sum(abs(p-q) for p, q in zip(law["modulo"], table.probabilities()))/2 == Fraction(1, 16)


def test_valid_table_does_not_certify_a_biased_bit_source():
    table = build([1, 1])
    assert table.probabilities() == (Fraction(1, 2),)*2
    assert {table.draw(lambda k: 0) for _ in range(10)} == {0}


@pytest.mark.parametrize("word", [-1, 8, True, 1.5, "1"])
def test_bad_bit_source_values(word):
    with pytest.raises(SamplingError):
        uniform_below(5, lambda k: word)


@pytest.mark.parametrize("weights", [[], [0, 0], [-1, 2], [1.0], [True], [2**4096]])
def test_bad_weights(weights):
    with pytest.raises(ValueError):
        build(weights)


def test_infinite_weight_iterator_is_bounded():
    from itertools import repeat
    with pytest.raises(ValueError, match="at most"):
        build(repeat(1))


@pytest.mark.parametrize("ticket", [-1, 18, 1.0, True])
def test_bad_tickets(ticket):
    with pytest.raises(ValueError):
        build([1, 2, 3]).decode(ticket)


@pytest.mark.parametrize("bound", [0, -1, True, 1.0])
def test_bad_bounds(bound):
    with pytest.raises(ValueError):
        uniform_below(bound, random.Random(1).getrandbits)


def test_bad_attempt_budget():
    with pytest.raises(ValueError):
        uniform_below(5, random.Random(1).getrandbits, max_attempts=0)
