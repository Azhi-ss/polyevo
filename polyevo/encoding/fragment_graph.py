"""Topological fragment assembly: multi-port fragments, side chains, new chain ends."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator

from polyevo.core.registry import register
from polyevo.domain.connectivity import ConnectivitySpec
from polyevo.encoding.base import Encoding


_BONDS = {}
for group in BRICS.reactionDefs:
    for left, right, bond in group:
        a, b = int(left.rstrip("ab")), int(right.rstrip("ab"))
        _BONDS[a, b] = _BONDS[b, a] = Chem.BondType.DOUBLE if bond == "=" else Chem.BondType.SINGLE

_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


@dataclass(frozen=True)
class FragmentInstance:
    smiles: str
    instance_id: int


@dataclass(frozen=True)
class FragmentGraph:
    instances: tuple[FragmentInstance, ...]
    # edge: ((inst_id, port_idx), (inst_id, port_idx)) — both are BRICS port atom indices in the fragment
    edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()


def _mol_ports(mol):
    orig = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIsotope() == 0]
    brics = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIsotope() > 0]
    return orig, brics


def _bond_type(a, b):
    return _BONDS.get((a, b))


@register("encoding", "fragment_graph")
class FragmentGraphEncoding(Encoding):
    name = "fragment_graph"

    def __init__(self, library, spec: ConnectivitySpec, max_instances=8, max_attempts=50):
        self.library = library
        self.spec = spec
        self.max_instances = max_instances
        self.max_attempts = max_attempts
        self.crossover_ops = {"subgraph_swap": subgraph_swap}
        self.mutation_ops = {
            "substitute_fragment": substitute_fragment,
            "insert_fragment": insert_fragment,
            "remove_fragment": remove_fragment,
            "rewire_edge": rewire_edge,
        }
        self._frag_meta = {}
        for smi in library.smiles:
            mol = Chem.MolFromSmiles(smi)
            orig, brics = _mol_ports(mol)
            labels = tuple(sorted(Chem.MolFromSmiles(smi).GetAtomWithIdx(i).GetIsotope() for i in brics))
            self._frag_meta[smi] = (len(orig), labels, tuple(brics))

    def random(self, ctx):
        for _ in range(self.max_attempts):
            g = self._assemble(ctx)
            if g is not None:
                smi = self.decode(g)
                if smi is not None:
                    return g
        raise ValueError("Failed to assemble a valid fragment graph")

    def _assemble(self, ctx):
        # Pick a start fragment with exactly 1 original port and >=1 BRICS port
        starts = [s for s, (o, b, _) in self._frag_meta.items() if o == 1 and len(b) >= 1]
        if not starts:
            return None
        start_smi = str(ctx.rng.choice(starts))
        instances = [FragmentInstance(start_smi, 0)]
        edges = []
        used = {0: set()}  # inst_id -> set of used BRICS port atom indices
        n = int(ctx.rng.integers(1, self.max_instances + 1))
        for i in range(1, n):
            frag_smi = self.library.sample(ctx.rng)
            inst = FragmentInstance(frag_smi, i)
            instances.append(inst)
            used[i] = set()
            # Find all possible (existing_inst, existing_port, new_port) with compatible bond
            options = []
            new_mol = Chem.MolFromSmiles(frag_smi)
            new_brics = [a.GetIdx() for a in new_mol.GetAtoms()
                         if a.GetAtomicNum() == 0 and a.GetIsotope() > 0]
            for j in range(i):
                j_mol = Chem.MolFromSmiles(instances[j].smiles)
                for p in [a.GetIdx() for a in j_mol.GetAtoms()
                          if a.GetAtomicNum() == 0 and a.GetIsotope() > 0 and a.GetIdx() not in used[j]]:
                    for q in new_brics:
                        if _bond_type(j_mol.GetAtomWithIdx(p).GetIsotope(),
                                      new_mol.GetAtomWithIdx(q).GetIsotope()) is not None:
                            options.append((j, p, q))
            if not options:
                return None
            j, p, q = options[int(ctx.rng.integers(len(options)))]
            edges.append(((j, p), (i, q)))
            used[j].add(p)
            used[i].add(q)
        # Validate: total original ports <= 2, remaining BRICS ports can be ends
        total_orig = sum(self._frag_meta[i.smiles][0] for i in instances)
        if total_orig > 2:
            return None
        # Check remaining BRICS ports are valid chain ends (single bond, not adjacent to another dummy)
        for inst in instances:
            mol = Chem.MolFromSmiles(inst.smiles)
            for a in mol.GetAtoms():
                if a.GetAtomicNum() == 0 and a.GetIsotope() > 0 and a.GetIdx() not in used[inst.instance_id]:
                    if a.GetDegree() != 1 or a.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
                        return None
        return FragmentGraph(tuple(instances), tuple(edges))

    def decode(self, genotype: FragmentGraph) -> str | None:
        try:
            combined = Chem.RWMol()
            offset = 0
            port_map = {}
            for inst in genotype.instances:
                mol = Chem.MolFromSmiles(inst.smiles)
                for a in mol.GetAtoms():
                    combined.AddAtom(Chem.Atom(a))
                for b in mol.GetBonds():
                    combined.AddBond(b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset, b.GetBondType())
                for idx in _mol_ports(mol)[0] + _mol_ports(mol)[1]:
                    port_map[(inst.instance_id, idx)] = offset + idx
                offset += mol.GetNumAtoms()
            to_remove = []
            for (i1, p1), (i2, p2) in genotype.edges:
                a1 = combined.GetAtomWithIdx(port_map[(i1, p1)])
                a2 = combined.GetAtomWithIdx(port_map[(i2, p2)])
                bt = _bond_type(a1.GetIsotope(), a2.GetIsotope())
                if bt is None:
                    return None
                n1 = a1.GetNeighbors()[0].GetIdx()
                n2 = a2.GetNeighbors()[0].GetIdx()
                combined.AddBond(n1, n2, bt)
                to_remove.extend([port_map[(i1, p1)], port_map[(i2, p2)]])
            for idx in sorted(to_remove, reverse=True):
                combined.RemoveAtom(idx)
            mol = combined.GetMol()
            # Count remaining ports: original + unconnected BRICS
            orig = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIsotope() == 0]
            brics = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIsotope() > 0]
            # Chain ends = original ports + enough BRICS ports to reach 2 total
            n_ends_needed = 2 - len(orig)
            if n_ends_needed < 0 or n_ends_needed > len(brics):
                return None
            # Convert first n_ends_needed BRICS ports to chain ends; remove the rest
            for a in brics[:n_ends_needed]:
                a.SetIsotope(0)
            for a in brics[n_ends_needed:]:
                # Remove unneeded BRICS port and its bond
                idx = a.GetIdx()
                rw = Chem.RWMol(mol)
                rw.RemoveAtom(idx)
                mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def distance(self, a, b):
        sa, sb = self.decode(a), self.decode(b)
        if sa is None or sb is None:
            return 1.0
        return 1.0 - DataStructs.TanimotoSimilarity(
            _FP.GetFingerprint(Chem.MolFromSmiles(sa)),
            _FP.GetFingerprint(Chem.MolFromSmiles(sb)))


# --- Operators ---

def _remap_edges(edges, removed_id=None, id_map=None):
    result = []
    for (i1, p1), (i2, p2) in edges:
        if removed_id is not None and removed_id in (i1, i2):
            continue
        ni1 = id_map.get(i1, i1) if id_map else i1
        ni2 = id_map.get(i2, i2) if id_map else i2
        result.append(((ni1, p1), (ni2, p2)))
    return tuple(result)


def subgraph_swap(a: FragmentGraph, b: FragmentGraph, ctx) -> FragmentGraph:
    if not a.instances or not b.instances:
        return a
    idx = int(ctx.rng.integers(len(a.instances)))
    donor = b.instances[int(ctx.rng.integers(len(b.instances)))]
    new_instances = list(a.instances)
    new_instances[idx] = FragmentInstance(donor.smiles, a.instances[idx].instance_id)
    return FragmentGraph(tuple(new_instances), _remap_edges(a.edges, removed_id=a.instances[idx].instance_id))


def substitute_fragment(g: FragmentGraph, ctx) -> FragmentGraph:
    if not g.instances:
        return g
    idx = int(ctx.rng.integers(len(g.instances)))
    new_smi = ctx.library.sample(ctx.rng)
    new_instances = list(g.instances)
    new_instances[idx] = FragmentInstance(new_smi, g.instances[idx].instance_id)
    return FragmentGraph(tuple(new_instances), _remap_edges(g.edges, removed_id=g.instances[idx].instance_id))


def insert_fragment(g: FragmentGraph, ctx) -> FragmentGraph:
    if len(g.instances) >= 8:  # hard limit; could be configurable
        return g
    new_id = max(i.instance_id for i in g.instances) + 1 if g.instances else 0
    new_smi = ctx.library.sample(ctx.rng)
    return FragmentGraph(g.instances + (FragmentInstance(new_smi, new_id),), g.edges)


def remove_fragment(g: FragmentGraph, ctx) -> FragmentGraph:
    if len(g.instances) <= 1:
        return g
    idx = int(ctx.rng.integers(len(g.instances)))
    removed = g.instances[idx].instance_id
    new_instances = tuple(i for i in g.instances if i.instance_id != removed)
    return FragmentGraph(new_instances, _remap_edges(g.edges, removed_id=removed))


def rewire_edge(g: FragmentGraph, ctx) -> FragmentGraph:
    if not g.edges:
        return g
    idx = int(ctx.rng.integers(len(g.edges)))
    new_edges = list(g.edges)
    new_edges.pop(idx)
    return FragmentGraph(g.instances, tuple(new_edges))
