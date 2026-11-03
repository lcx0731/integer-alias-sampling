"""Strict portable JSON: large integers are decimal strings, not JSON numbers."""

import json
import re

from .core import AliasTable, MAX_ITEMS

FORMAT = "integer-alias-v1"
MAX_JSON_BYTES = 16*1024*1024


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def parse_json(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("JSON must be a string no larger than 16 MiB")
    return json.loads(text, object_pairs_hook=_unique)


def _decimal(text):
    if not isinstance(text, str) or not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise ValueError("weights and cutoffs must be canonical nonnegative decimal strings")
    # 4096-bit weights plus ceil(log2(MAX_ITEMS)) bits for their sum.
    if len(text) > 1240:
        raise ValueError("decimal integer exceeds implementation limit")
    return int(text)


def load_weights(text):
    value = parse_json(text)
    if not isinstance(value, dict) or set(value) != {"weights"}:
        raise ValueError("weights input must contain only the weights field")
    if not isinstance(value["weights"], list) or not 1 <= len(value["weights"]) <= MAX_ITEMS:
        raise ValueError("weights must be a nonempty bounded array")
    # Core validation handles positivity and bit limits after conversion.
    return tuple(_decimal(x) for x in value["weights"])


def dumps(table):
    return json.dumps({"format": FORMAT, "weights": list(map(str, table.weights)),
                       "cutoffs": list(map(str, table.cutoffs)),
                       "aliases": list(table.aliases)}, indent=2)+"\n"


def loads(text):
    value = parse_json(text)
    fields = {"format", "weights", "cutoffs", "aliases"}
    if not isinstance(value, dict) or set(value) != fields or value["format"] != FORMAT:
        raise ValueError("unsupported or malformed table format")
    if not all(isinstance(value[name], list) for name in fields-{"format"}):
        raise ValueError("table fields must be arrays")
    if not 1 <= len(value["weights"]) <= MAX_ITEMS:
        raise ValueError("table size outside implementation limit")
    if len(value["cutoffs"]) != len(value["weights"]) or len(value["aliases"]) != len(value["weights"]):
        raise ValueError("table array lengths differ")
    return AliasTable(tuple(_decimal(x) for x in value["weights"]),
                      tuple(_decimal(x) for x in value["cutoffs"]), tuple(value["aliases"]))
