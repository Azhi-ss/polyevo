"""Polymer-aware BRICS decomposition and dataset-to-library construction."""

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import BRICS

from polyevo.domain.connectivity import ConnectivitySpec, canonicalize


@dataclass(frozen=True)
class PolymerDecomposition:
    smiles: str
    # Atom indices refer to MolFromSmiles(smiles), not the input atom ordering.
    bonds: tuple
    fragments: tuple[str, ...]


def decompose_polymer(smiles: str) -> PolymerDecomposition:
    """Cut every RDKit BRICS bond once, retaining multiplicity and original '*'.

    This is a single full cut, not recursive enumeration of overlapping fragments.
    Original polymer ports have isotope 0; BRICS-generated ports have isotope > 0.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Expected a nonempty polymer SMILES")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError("Cannot parse polymer SMILES")
    if any(a.GetAtomicNum() == 0 and a.GetIsotope() != 0 for a in mol.GetAtoms()):
        raise ValueError("Input polymer endpoints must be unlabeled '*', not BRICS ports")
    canonical = canonicalize(smiles.strip(), ConnectivitySpec())
    if canonical is None:
        raise ValueError("Expected one connected repeat unit with exactly two single-bonded '*' endpoints")
    mol = Chem.MolFromSmiles(canonical)
    bonds = tuple(sorted(BRICS.FindBRICSBonds(mol)))
    # Avoid BreakBRICSBonds' empty-list auto-detection path when there are no cuts.
    broken = BRICS.BreakBRICSBonds(mol, bonds=list(bonds)) if bonds else mol
    fragments = tuple(sorted(Chem.MolToSmiles(f, canonical=True)
                             for f in Chem.GetMolFrags(broken, asMols=True)))
    return PolymerDecomposition(canonical, bonds, fragments)


_SCHEMA = pa.schema([
    ("smiles", pa.string()), ("weight", pa.float64()),
    ("heavy_atoms", pa.int64()), ("n_attachment", pa.int64()),
    ("n_brics_ports", pa.int64()), ("brics_labels", pa.list_(pa.int64())),
    ("occurrences", pa.int64()), ("source_count", pa.int64()),
])


def build_library(input_path, out, *, smiles_column="smiles", min_heavy_atoms=1,
                  max_heavy_atoms=None, forbidden_smarts=(), allowed_elements=None):
    """Build a deduplicated library; invalid rows are skipped and reported.

    Duplicate canonical polymers do not inflate fragment occurrence/source counts.
    All port topologies are retained unless explicitly filtered by the caller.
    """
    if isinstance(min_heavy_atoms, bool) or not isinstance(min_heavy_atoms, int) or min_heavy_atoms < 1:
        raise ValueError("min_heavy_atoms must be a positive integer")
    if max_heavy_atoms is not None and (
            isinstance(max_heavy_atoms, bool) or not isinstance(max_heavy_atoms, int)
            or max_heavy_atoms < min_heavy_atoms):
        raise ValueError("max_heavy_atoms must be an integer >= min_heavy_atoms")
    patterns = []
    for smarts in forbidden_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"Invalid forbidden SMARTS: {smarts}")
        patterns.append(pattern)
    elements = None
    if allowed_elements is not None:
        elements = set(allowed_elements)
        valid_symbols = {Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)}
        if not elements or not elements <= valid_symbols:
            raise ValueError("allowed_elements must contain valid element symbols; '*' ports are always allowed")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing library: {out}")
    report = {
        "input_rows": 0, "invalid_rows": 0, "duplicate_polymers": 0,
        "unique_polymers": 0, "polymers_without_cuts": 0, "brics_bonds_cut": 0,
        "fragment_occurrences": 0, "filtered_occurrences": 0,
        "filter_counts": {"heavy_atoms": 0, "elements": 0, "smarts": 0},
        "invalid_examples": [],
    }
    seen_polymers, records = set(), {}
    with Path(input_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or smiles_column not in reader.fieldnames:
            raise ValueError(f"CSV must have a '{smiles_column}' column")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV has duplicate column names")
        for row in reader:
            report["input_rows"] += 1
            try:
                if None in row:
                    raise ValueError("CSV row has extra fields; quote SMILES containing commas")
                decomposition = decompose_polymer(row.get(smiles_column))
            except ValueError as exc:
                report["invalid_rows"] += 1
                if len(report["invalid_examples"]) < 20:
                    report["invalid_examples"].append({"line": reader.line_num, "reason": str(exc)})
                continue
            if decomposition.smiles in seen_polymers:
                report["duplicate_polymers"] += 1
                continue
            seen_polymers.add(decomposition.smiles)
            report["unique_polymers"] += 1
            report["polymers_without_cuts"] += int(not decomposition.bonds)
            report["brics_bonds_cut"] += len(decomposition.bonds)
            source_fragments = set()
            for fragment in decomposition.fragments:
                report["fragment_occurrences"] += 1
                mol = Chem.MolFromSmiles(fragment)
                heavy = mol.GetNumHeavyAtoms()
                reason = None
                if heavy < min_heavy_atoms or (max_heavy_atoms is not None and heavy > max_heavy_atoms):
                    reason = "heavy_atoms"
                elif elements is not None and any(a.GetAtomicNum() != 0 and a.GetSymbol() not in elements
                                                   for a in mol.GetAtoms()):
                    reason = "elements"
                elif any(mol.HasSubstructMatch(pattern) for pattern in patterns):
                    reason = "smarts"
                if reason:
                    report["filtered_occurrences"] += 1
                    report["filter_counts"][reason] += 1
                    continue
                if fragment not in records:
                    ports = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
                    labels = sorted(a.GetIsotope() for a in ports if a.GetIsotope() > 0)
                    records[fragment] = {
                        "smiles": fragment, "weight": 1.0, "heavy_atoms": heavy,
                        "n_attachment": sum(a.GetIsotope() == 0 for a in ports),
                        "n_brics_ports": len(labels), "brics_labels": labels,
                        "occurrences": 0, "source_count": 0,
                    }
                records[fragment]["occurrences"] += 1
                if fragment not in source_fragments:
                    records[fragment]["source_count"] += 1
                    source_fragments.add(fragment)
    if not records:
        raise ValueError(f"No fragments retained; no output written. Report: {json.dumps(report)}")
    rows = [records[key] for key in sorted(records)]
    report.update(unique_fragments=len(rows),
                  fragments_with_polymer_endpoints=sum(r["n_attachment"] > 0 for r in rows),
                  fragments_with_multiple_brics_ports=sum(r["n_brics_ports"] > 2 for r in rows),
                  legacy_two_port_compatible=sum(r["n_attachment"] == 0 and r["n_brics_ports"] == 2
                                                 for r in rows))
    metadata = {
        b"polyevo.library_format": b"polymer_brics_v1",
        b"polyevo.rdkit_version": rdBase.rdkitVersion.encode(),
        b"polyevo.build_report": json.dumps(report, sort_keys=True).encode(),
        b"polyevo.build_config": json.dumps({
            "smiles_column": smiles_column, "min_heavy_atoms": min_heavy_atoms,
            "max_heavy_atoms": max_heavy_atoms, "forbidden_smarts": list(forbidden_smarts),
            "allowed_elements": sorted(elements) if elements is not None else None,
            "decomposition": "all_brics_bonds_single_cut",
        }, sort_keys=True).encode(),
    }
    table = pa.Table.from_pylist(rows, schema=_SCHEMA.with_metadata(metadata))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file without replacing an existing destination, even in a race.
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".parquet") as tmp:
        pq.write_table(table, tmp.name)
        os.link(tmp.name, out)
    return {"out": str(out), **report}
