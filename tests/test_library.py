import csv
import json

import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from polyevo.cli import main
from polyevo.encoding.library import build_library, decompose_polymer


POLYESTER = "*CCOC(=O)c1ccc(C(=O)OCC*)cc1"
BRANCHED = "*CCN(CC)C(=O)CC*"


def write_csv(path, smiles, column="smiles"):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([column])
        writer.writerows([s] for s in smiles)
    return path


def test_polyester_exact_fragments_and_endpoint_identity():
    result = decompose_polymer(POLYESTER)
    assert len(result.bonds) == 6
    assert result.fragments == tuple(sorted([
        "*CC[4*]", "*CC[4*]", "[1*]C([6*])=O", "[1*]C([6*])=O",
        "[16*]c1ccc([16*])cc1", "[3*]O[3*]", "[3*]O[3*]",
    ]))
    fragments = [Chem.MolFromSmiles(s) for s in result.fragments]
    assert sum(m.GetNumHeavyAtoms() for m in fragments) == Chem.MolFromSmiles(POLYESTER).GetNumHeavyAtoms()
    ports = [a for m in fragments for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    assert sum(a.GetIsotope() == 0 for a in ports) == 2
    assert sum(a.GetIsotope() > 0 for a in ports) == 2 * len(result.bonds)
    original = Chem.MolFromSmiles(result.smiles)
    assert all(original.GetBondBetweenAtoms(*indices) is not None for indices, labels in result.bonds)


def test_no_brics_bonds_preserves_repeat_unit_and_stereo():
    for smiles in ("*CC*", "*C[C@H](F)CC*"):
        result = decompose_polymer(smiles)
        assert result.bonds == ()
        assert result.fragments == (Chem.MolToSmiles(Chem.MolFromSmiles(smiles)),)
    assert "@" in decompose_polymer("*C[C@H](F)CC*").fragments[0]


def test_single_and_multiport_fragments_are_retained():
    result = decompose_polymer(BRANCHED)
    assert "[5*]N([5*])[5*]" in result.fragments
    assert "[4*]CC" in result.fragments
    assert any(s.startswith("*") for s in result.fragments)


@pytest.mark.parametrize("smiles", [None, "", "not_smiles", "CC", "*CC", "*C(*)*", "**",
                                    "*CC*.[Na+]", "[16*]CC[16*]", "*=CC*"])
def test_invalid_polymer_inputs(smiles):
    with pytest.raises(ValueError):
        decompose_polymer(smiles)


def test_dataset_deduplication_counts_and_metadata(tmp_path):
    equivalent = Chem.MolToSmiles(Chem.MolFromSmiles(POLYESTER), rootedAtAtom=5, canonical=False)
    source = write_csv(tmp_path / "input.csv", [POLYESTER, equivalent, "CC", "", "*CC*"])
    output = tmp_path / "nested" / "library.parquet"
    report = build_library(source, output)
    assert report["input_rows"] == 5
    assert report["invalid_rows"] == 2
    assert report["duplicate_polymers"] == 1
    assert report["unique_polymers"] == 2
    assert report["polymers_without_cuts"] == 1
    assert report["brics_bonds_cut"] == 6
    assert report["fragment_occurrences"] == 8
    assert report["unique_fragments"] == 5
    assert report["legacy_two_port_compatible"] == 3
    table = pq.read_table(output)
    rows = {row["smiles"]: row for row in table.to_pylist()}
    assert rows["*CC[4*]"] == {
        "smiles": "*CC[4*]", "weight": 1.0, "heavy_atoms": 2,
        "n_attachment": 1, "n_brics_ports": 1, "brics_labels": [4],
        "occurrences": 2, "source_count": 1,
    }
    assert rows["*CC*"]["n_brics_ports"] == 0
    assert json.loads(table.schema.metadata[b"polyevo.build_report"])["invalid_rows"] == 2
    assert report["invalid_examples"][0]["line"] == 4


def test_counts_across_distinct_parents(tmp_path):
    source = write_csv(tmp_path / "input.csv", [POLYESTER, "*CCOCC*"])
    out = tmp_path / "library.parquet"
    build_library(source, out)
    rows = {r["smiles"]: r for r in pq.read_table(out).to_pylist()}
    assert rows["*CC[4*]"]["occurrences"] == 4
    assert rows["*CC[4*]"]["source_count"] == 2


@pytest.mark.parametrize("options,removed", [
    ({"min_heavy_atoms": 2}, {"[3*]O[3*]"}),
    ({"max_heavy_atoms": 2}, {"[16*]c1ccc([16*])cc1"}),
    ({"allowed_elements": ["C"]}, {"[3*]O[3*]", "[1*]C([6*])=O"}),
    ({"forbidden_smarts": ["C=O"]}, {"[1*]C([6*])=O"}),
])
def test_fragment_filters(tmp_path, options, removed):
    source = write_csv(tmp_path / "input.csv", [POLYESTER])
    out = tmp_path / "library.parquet"
    report = build_library(source, out, **options)
    kept = set(pq.read_table(out).column("smiles").to_pylist())
    assert set(decompose_polymer(POLYESTER).fragments) - kept == removed
    assert report["filtered_occurrences"] > 0
    assert sum(report["filter_counts"].values()) == report["filtered_occurrences"]


@pytest.mark.parametrize("options", [
    {"min_heavy_atoms": 0}, {"max_heavy_atoms": 0},
    {"allowed_elements": ["NotAnElement"]}, {"forbidden_smarts": ["["]},
])
def test_invalid_filters_do_not_write(tmp_path, options):
    source = write_csv(tmp_path / "input.csv", [POLYESTER])
    out = tmp_path / "library.parquet"
    with pytest.raises(ValueError):
        build_library(source, out, **options)
    assert not out.exists()


def test_no_overwrite_and_empty_output_is_error(tmp_path):
    source = write_csv(tmp_path / "input.csv", [POLYESTER])
    out = tmp_path / "library.parquet"
    build_library(source, out)
    content = out.read_bytes()
    with pytest.raises(FileExistsError):
        build_library(source, out)
    assert out.read_bytes() == content
    for options in ({"min_heavy_atoms": 1000}, {"allowed_elements": ["He"]}):
        with pytest.raises(ValueError, match="No fragments retained"):
            build_library(source, tmp_path / "empty.parquet", **options)
        assert not (tmp_path / "empty.parquet").exists()


def test_malformed_csv_and_missing_column(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("wrong\n*CC*\n")
    with pytest.raises(ValueError, match="column"):
        build_library(source, tmp_path / "out.parquet")
    source.write_text("smiles,smiles\n*CC*,*CC*\n")
    with pytest.raises(ValueError, match="duplicate column"):
        build_library(source, tmp_path / "out.parquet")
    source.write_text("smiles\n*CC*,extra\n*CC*\n")
    assert build_library(source, tmp_path / "out.parquet")["invalid_rows"] == 1


def test_cli_custom_column_filters_and_reproducibility(tmp_path, capsys):
    source = write_csv(tmp_path / "input.csv", [POLYESTER, BRANCHED], column="repeat_unit")
    paths = [tmp_path / "one.parquet", tmp_path / "two.parquet"]
    for out in paths:
        assert main(["build-library", "--input", str(source), "--out", str(out),
                     "--smiles-column", "repeat_unit", "--min-heavy-atoms", "2",
                     "--allowed-elements", "C", "N", "O", "--forbid-smarts", "[Cl]"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["unique_polymers"] == 2
        assert report["unique_fragments"] > 0
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert main(["build-library", "--input", str(source), "--out", str(paths[0])]) == 2
    assert "overwrite" in capsys.readouterr().err
