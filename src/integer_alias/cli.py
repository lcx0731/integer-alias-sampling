"""Build, audit, and sample integer alias tables. Files are read-only inputs."""

import argparse
import json
from pathlib import Path
import random
import sys

from . import SamplingError, build
from .codec import MAX_JSON_BYTES, dumps, load_weights, loads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["build", "audit", "sample"]:
        p = commands.add_parser(name)
        p.add_argument("input", type=Path)
        if name == "sample":
            p.add_argument("--count", type=int, default=1000)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES+1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("input exceeds 16 MiB")
        text = raw.decode("utf-8")
        if args.command == "build":
            print(dumps(build(load_weights(text))), end="")
            return 0
        table = loads(text)
        if args.command == "audit":
            output = {"valid": True, "n": len(table.weights), "gcd_unit": str(table.unit),
                      "column_height": str(table.total), "lattice_cells": str(table.cells),
                      "output_cell_counts": list(map(str, table.cell_counts())),
                      "probabilities": list(map(str, table.probabilities())),
                      "scope": "Exact table masses; bit-source quality is not certified."}
        else:
            if not 0 <= args.count <= 1_000_000:
                raise ValueError("sample count must be in [0, 1000000]")
            rng = random.Random(args.seed)
            counts = [0]*len(table.weights)
            for _ in range(args.count):
                counts[table.draw(rng.getrandbits)] += 1
            output = {"draws": args.count, "seed": args.seed, "counts": counts,
                      "rng": "Python random.Random; pseudorandom, not cryptographic"}
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 1
    except SamplingError as exc:
        print(f"sampling error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
