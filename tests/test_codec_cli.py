import json
import subprocess
import sys

import pytest

from integer_alias import build
from integer_alias.codec import dumps, loads, load_weights


def test_large_integer_round_trip_and_canonical_dump():
    table = build([2**2048, 1, 3])
    encoded = dumps(table)
    assert loads(encoded) == table
    assert dumps(loads(encoded)) == encoded
    assert isinstance(json.loads(encoded)["weights"][0], str)


@pytest.mark.parametrize("field,value", [
    ("format", "unknown"), ("weights", ["0", "0", "0"]),
    ("weights", ["2", "2", "3"]), ("weights", [1, 2, 3]),
    ("weights", ["01", "2", "3"]), ("weights", ["-1", "2", "3"]),
    ("cutoffs", ["4", "6", "6"]), ("cutoffs", ["7", "6", "6"]),
    ("cutoffs", ["3", "6"]), ("aliases", [3, 1, 2]),
    ("aliases", [True, 1, 2]), ("aliases", [0, 1, 2]),
])
def test_corrupt_or_incompatible_table_rejected(field, value):
    obj = json.loads(dumps(build([1, 2, 3])))
    obj[field] = value
    with pytest.raises(ValueError):
        loads(json.dumps(obj))


def test_duplicate_and_unknown_fields_rejected():
    text = dumps(build([1]))
    with pytest.raises(ValueError, match="duplicate"):
        loads(text.replace('"format":', '"format": "integer-alias-v1", "format":'))
    obj = json.loads(text)
    obj["unexpected"] = 1
    with pytest.raises(ValueError):
        loads(json.dumps(obj))


@pytest.mark.parametrize("text", ['[]', '{}', '{"weights": []}', '{"weights": ["+1"]}',
                                  '{"weights": ["1.0"]}', '{"weights": [true]}'])
def test_weights_json_schema(text):
    with pytest.raises(ValueError):
        load_weights(text)


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "integer_alias.cli", *map(str, args)],
                          capture_output=True, text=True)


def test_cli_build_audit_sample_and_input_immutability(tmp_path):
    path = tmp_path/"weights.json"
    original = '{"weights": ["1", "2", "3"]}'
    path.write_text(original)
    built = run_cli("build", path)
    assert built.returncode == 0
    assert path.read_text() == original
    table_path = tmp_path/"table.json"
    table_path.write_text(built.stdout)
    verified = run_cli("audit", table_path)
    assert verified.returncode == 0
    assert json.loads(verified.stdout)["output_cell_counts"] == ["3", "6", "9"]
    first = run_cli("sample", table_path, "--count", 1000, "--seed", 731)
    second = run_cli("sample", table_path, "--count", 1000, "--seed", 731)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert sum(json.loads(first.stdout)["counts"]) == 1000
    assert table_path.read_text() == built.stdout


def test_cli_invalid_count_and_invalid_utf8(tmp_path):
    path = tmp_path/"table.json"
    path.write_text(dumps(build([1])))
    invalid = run_cli("sample", path, "--count", -1)
    assert invalid.returncode == 1 and "input error" in invalid.stderr
    path.write_bytes(b'\xff')
    invalid = run_cli("audit", path)
    assert invalid.returncode == 1


def test_cli_no_draws(tmp_path):
    path = tmp_path/"table.json"
    path.write_text(dumps(build([0, 1])))
    result = run_cli("sample", path, "--count", 0)
    assert result.returncode == 0
    assert json.loads(result.stdout)["counts"] == [0, 0]
