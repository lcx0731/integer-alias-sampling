"""Synthetic exact-law examples and deterministic lookup/bit-cost diagnostics."""

import argparse
from fractions import Fraction
from itertools import accumulate
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import build
from .codec import dumps
from .reference import cdf_lookup, one_word_laws


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    table = build([1, 2, 3])
    laws = one_word_laws(table, 5)
    intended = table.probabilities()
    tv = sum(abs(p-q) for p, q in zip(intended, laws["modulo"]))/2
    modulo = {"weights": [1, 2, 3], "bits": 5,
              "intended": list(map(str, intended)),
              "modulo": list(map(str, laws["modulo"])),
              "rejection_given_acceptance": list(map(str, laws["rejection_given_acceptance"])),
              "acceptance_probability": str(laws["acceptance"]),
              "expected_attempts_without_cap": str(1/laws["acceptance"]),
              "total_variation_modulo": str(tv)}

    huge = build([2**60, 1, 1])
    float_cdf = list(accumulate(map(float, huge.weights)))
    float_widths = [float_cdf[0]]+[b-a for a, b in zip(float_cdf, float_cdf[1:])]
    rare_witnesses = [j*huge.total for j in (1, 2)]
    assert [huge.decode(t) for t in rare_witnesses] == [1, 2]
    rare = {"weights": list(map(str, huge.weights)),
            "probabilities": list(map(str, huge.probabilities())),
            "cell_counts": list(map(str, huge.cell_counts())),
            "float_cdf": list(map(str, float_cdf)),
            "float_cdf_increment_widths": list(map(str, float_widths)),
            "rare_output_tickets": list(map(str, rare_witnesses)),
            "intended_rare_probability": str(sum(huge.probabilities()[1:])),
            "expected_draws_until_either_rare_under_iid": str(1/sum(huge.probabilities()[1:]))}

    # Algorithmic work only. No wall-clock speed claim: the random ticket ranges
    # and Python integer operation costs differ between the two decoders.
    costs = []
    for n in [16, 64, 256, 1024, 4096]:
        weights = [1+(j*37+11)%97 for j in range(n)]
        candidate = build(weights)
        cumulative = tuple(accumulate(weights))
        comparisons = []
        for j in range(1024):
            ticket = (j*cumulative[-1])//1024
            _, work = cdf_lookup(cumulative, ticket)
            comparisons.append(work)
        cdf_bits = (cumulative[-1]-1).bit_length()
        alias_bits = (candidate.cells-1).bit_length()
        costs.append({"n": n, "queries": 1024,
                      "cdf_mean_comparisons": sum(comparisons)/len(comparisons),
                      "cdf_max_comparisons": max(comparisons),
                      "alias_cutoff_comparisons_per_decode": 1,
                      "cdf_bits_per_attempt": cdf_bits,
                      "alias_bits_per_attempt": alias_bits,
                      "cdf_expected_bits_without_cap": str(Fraction(cdf_bits*(1 << cdf_bits), cumulative[-1])),
                      "alias_expected_bits_without_cap": str(Fraction(alias_bits*(1 << alias_bits), candidate.cells))})

    # A reproducible smoke sample, explicitly not evidence for tiny probabilities.
    rng = random.Random(731)
    counts = [0, 0, 0]
    for _ in range(10000):
        counts[table.draw(rng.getrandbits)] += 1
    summary = {"data": "Synthetic integer weights; deterministic enumerations and a seed-731 smoke sample.",
               "modulo_example": modulo, "rare_support_example": rare,
               "cost_diagnostics": costs,
               "smoke_sample": {"seed": 731, "draws": 10000, "counts": counts}}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"weights.json").write_text('{"weights": ["1", "2", "3"]}\n')
    (output/"table.json").write_text(dumps(table))
    (output/"rare-table.json").write_text(dumps(huge))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    axs[0].bar([-.18, .82, 1.82], list(map(float, intended)), width=.36,
               label="Intended / rejection", color="#286f86")
    axs[0].bar([.18, 1.18, 2.18], list(map(float, laws["modulo"])), width=.36,
               label="5-bit word modulo 18", color="#cf7138")
    axs[0].set(xticks=[0, 1, 2], xticklabels=["0", "1", "2"], ylim=(0, .62),
               xlabel="Category", ylabel="Exact probability", title="Modulo changes the output law")
    axs[0].legend(fontsize=8, loc="upper left")
    axs[0].text(.96, .96, "TV = 1/16", transform=axs[0].transAxes, ha="right", va="top")
    sizes = [r["n"] for r in costs]
    axs[1].plot(sizes, [r["cdf_mean_comparisons"] for r in costs], "o-",
                label="Integer CDF: mean comparisons", color="#cf7138")
    axs[1].plot(sizes, [1]*len(costs), "o-", label="Alias: one cutoff comparison", color="#286f86")
    axs[1].set(xscale="log", xlabel="Number of categories", ylabel="Comparisons per decode",
               title="Less lookup work after preprocessing")
    axs[1].legend(fontsize=8)
    axs[2].plot(sizes, [r["cdf_bits_per_attempt"] for r in costs], "o-",
                label="Integer CDF ticket", color="#cf7138")
    axs[2].plot(sizes, [r["alias_bits_per_attempt"] for r in costs], "o-",
                label="Combined alias ticket", color="#286f86")
    axs[2].set(xscale="log", xlabel="Number of categories", ylabel="Requested bits per attempt",
               title="The larger ticket space costs random bits")
    axs[2].legend(fontsize=8)
    fig.savefig(output/"sampling-diagnostics.png", dpi=160, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(json.dumps({"output": str(output), "modulo_total_variation": str(tv),
                      "rare_probability": rare["intended_rare_probability"],
                      "smoke_counts": counts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    generate(parser.parse_args().output)


if __name__ == "__main__":
    main()
