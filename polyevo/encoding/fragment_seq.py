from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator

from polyevo.core.registry import register


# RDKit is the source of truth for BRICS compatibility, including the 7a/7b bond.
_BONDS = {}
for group in BRICS.reactionDefs:
    for left, right, bond in group:
        a, b = int(left.rstrip("ab")), int(right.rstrip("ab"))
        kind = Chem.BondType.DOUBLE if bond == "=" else Chem.BondType.SINGLE
        _BONDS[a, b] = _BONDS[b, a] = kind
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


@dataclass(frozen=True)
class FragmentLibrary:
    smiles: tuple[str, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_entries(cls, entries, require_two_brics=True):
        unique = {}
        for entry in entries:
            smi = entry if isinstance(entry, str) else entry["smiles"]
            weight = 1.0 if isinstance(entry, str) else float(entry.get("weight", 1.0))
            mol = Chem.MolFromSmiles(smi)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError(f"Invalid fragment: {smi}")
            ports = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            if require_two_brics:
                if len(ports) != 2 or any(a.GetDegree() != 1 or a.GetIsotope() == 0
                                          or a.GetNeighbors()[0].GetAtomicNum() == 0 for a in ports):
                    raise ValueError(f"M1 fragment_seq requires two labeled BRICS ports: {smi}")
            else:
                for a in ports:
                    if a.GetDegree() != 1 or a.GetNeighbors()[0].GetAtomicNum() == 0:
                        raise ValueError(f"Invalid port in fragment: {smi}")
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("Fragment weights must be positive and finite")
            unique.setdefault(Chem.MolToSmiles(mol), weight)
        if not unique:
            raise ValueError("Fragment library must not be empty")
        return cls(tuple(unique), tuple(unique.values()))

    def sample(self, rng):
        weights = np.asarray(self.weights)
        return self.smiles[int(rng.choice(len(weights), p=weights / weights.sum()))]


def _join(left, right):
    offset = left.GetNumAtoms()
    for a in left.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        for b in right.GetAtoms():
            if b.GetAtomicNum() != 0:
                continue
            bond = _BONDS.get((a.GetIsotope(), b.GetIsotope()))
            if bond is None:
                continue
            combined = Chem.RWMol(Chem.CombineMols(left, right))
            combined.AddBond(a.GetNeighbors()[0].GetIdx(),
                             offset + b.GetNeighbors()[0].GetIdx(), bond)
            for idx in sorted((a.GetIdx(), offset + b.GetIdx()), reverse=True):
                combined.RemoveAtom(idx)
            result = combined.GetMol()
            Chem.SanitizeMol(result)
            return result
    return None


@register("encoding", "fragment_seq")
class FragmentSequence:
    name = "fragment_seq"

    def __init__(self, library, spec, n_fragments=(2, 4)):
        self.library, self.spec = library, spec
        if len(n_fragments) != 2 or not 1 <= n_fragments[0] <= n_fragments[1]:
            raise ValueError("n_fragments must be [minimum, maximum], minimum >= 1")
        self.n_fragments = tuple(n_fragments)
        self.crossover_ops = {"brics_single_point": single_point}
        self.mutation_ops = {"substitute": substitute, "insert": insert,
                             "delete": delete, "swap_order": swap_order}

    def random(self, ctx):
        n = int(ctx.rng.integers(self.n_fragments[0], self.n_fragments[1] + 1))
        return tuple(self.library.sample(ctx.rng) for _ in range(n))

    def decode(self, genotype):
        if not genotype:
            return None
        try:
            mol = Chem.MolFromSmiles(genotype[0])
            if mol is None:
                return None
            for fragment in genotype[1:]:
                other = Chem.MolFromSmiles(fragment)
                if other is None:
                    return None
                mol = _join(mol, other)
                if mol is None:
                    return None
            stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            if self.spec.n_attachment == 0:
                for idx in stars:
                    atom = mol.GetAtomWithIdx(idx)
                    atom.SetAtomicNum(1)
                    atom.SetIsotope(0)
                mol = Chem.RemoveHs(mol)
            else:
                for idx in stars:
                    mol.GetAtomWithIdx(idx).SetIsotope(0)
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except (ValueError, RuntimeError):
            return None

    def distance(self, a, b):
        sa, sb = self.decode(a), self.decode(b)
        if sa is None or sb is None:
            raise ValueError("Cannot measure distance of invalid genotypes")
        return 1.0 - DataStructs.TanimotoSimilarity(
            _FP.GetFingerprint(Chem.MolFromSmiles(sa)),
            _FP.GetFingerprint(Chem.MolFromSmiles(sb)))


@register("crossover", "brics_single_point")
def single_point(a, b, ctx):
    i = int(ctx.rng.integers(1, len(a) + 1))
    j = int(ctx.rng.integers(0, len(b)))
    return a[:i] + b[j:]


@register("mutation", "substitute")
def substitute(g, ctx):
    i = int(ctx.rng.integers(len(g)))
    return g[:i] + (ctx.library.sample(ctx.rng),) + g[i + 1:]


@register("mutation", "insert")
def insert(g, ctx):
    i = int(ctx.rng.integers(len(g) + 1))
    return g[:i] + (ctx.library.sample(ctx.rng),) + g[i:]


@register("mutation", "delete")
def delete(g, ctx):
    if len(g) == 1:
        return g
    i = int(ctx.rng.integers(len(g)))
    return g[:i] + g[i + 1:]


@register("mutation", "swap_order")
def swap_order(g, ctx):
    if len(g) < 2:
        return g
    i, j = ctx.rng.choice(len(g), size=2, replace=False)
    result = list(g)
    result[i], result[j] = result[j], result[i]
    return tuple(result)
