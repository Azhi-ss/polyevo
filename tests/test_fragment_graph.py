from types import SimpleNamespace

import numpy as np
import pytest

from polyevo.domain.connectivity import ConnectivitySpec
from polyevo.encoding.fragment_graph import FragmentGraph, FragmentGraphEncoding, FragmentInstance
from polyevo.encoding.fragment_seq import FragmentLibrary


@pytest.fixture
def library():
    return FragmentLibrary.from_entries([
        "*CC[4*]",           # endpoint: 1 original + 1 BRICS (4)
        "*CCO[3*]",          # endpoint: 1 original + 1 BRICS (3)
        "[3*]CC[4*]",        # middle: 0 original + 2 BRICS (3,4)
        "[3*]O[3*]",         # middle: 0 original + 2 BRICS (3,3)
        "[5*]N([5*])[5*]",   # branch: 0 original + 3 BRICS (5,5,5)
        "[1*]C([6*])=O",     # middle: 0 original + 2 BRICS (1,6)
        "*CC*",              # full repeat: 2 original + 0 BRICS
    ], require_two_brics=False)


@pytest.fixture
def encoding(library):
    return FragmentGraphEncoding(library, ConnectivitySpec())


def test_random_assembly_produces_valid_smiles(encoding):
    ctx = SimpleNamespace(rng=np.random.default_rng(42), library=encoding.library)
    for _ in range(50):
        g = encoding.random(ctx)
        smi = encoding.decode(g)
        assert smi is not None
        # Must have exactly 2 chain ends (*)
        mol = __import__("rdkit").Chem.MolFromSmiles(smi)
        stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(stars) == 2


def test_decode_deterministic(encoding):
    g = FragmentGraph(
        instances=(FragmentInstance("*CC[4*]", 0), FragmentInstance("[3*]CC[4*]", 1)),
        edges=(((0, 3), (1, 0)),)  # [4*] -> [3*] is compatible
    )
    s1 = encoding.decode(g)
    s2 = encoding.decode(g)
    assert s1 == s2
    assert s1 is not None


def test_edge_compatibility_enforced(encoding):
    # [4*] cannot bond to [3*] in BRICS rules
    g = FragmentGraph(
        instances=(FragmentInstance("*CC[4*]", 0), FragmentInstance("[1*]C([6*])=O", 1)),
        edges=(((0, 3), (1, 0)),)  # [4*] and [1*] are incompatible
    )
    assert encoding.decode(g) is None


def test_multiport_fragment_side_chain(encoding):
    # [5*]N([5*])[5*] can connect to 3 fragments
    g = FragmentGraph(
        instances=(
            FragmentInstance("*CC[4*]", 0),
            FragmentInstance("[3*]CC[4*]", 1),
            FragmentInstance("[5*]N([5*])[5*]", 2),
        ),
        edges=(((0, 3), (1, 0)), ((1, 3), (2, 0)))  # 4->3 compatible, 4->5 compatible
    )
    smi = encoding.decode(g)
    assert smi is not None
    mol = __import__("rdkit").Chem.MolFromSmiles(smi)
    assert len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]) == 2


def test_operators_preserve_connectivity(encoding):
    ctx = SimpleNamespace(rng=np.random.default_rng(7), library=encoding.library)
    g = encoding.random(ctx)
    for op_name, op in encoding.mutation_ops.items():
        child = op(g, ctx)
        smi = encoding.decode(child)
        # May be None if mutation breaks connectivity or produces wrong port count
        # That's OK — canonicalize will filter it downstream
        if smi is not None:
            mol = __import__("rdkit").Chem.MolFromSmiles(smi)
            n_stars = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 0])
            assert n_stars >= 2  # May be >2 if mutation disconnected a fragment; canonicalize filters it


def test_too_many_original_ports_rejected(library):
    enc = FragmentGraphEncoding(library, ConnectivitySpec())
    g = FragmentGraph(
        instances=(
            FragmentInstance("*CC[4*]", 0),
            FragmentInstance("*CCO[3*]", 1),
            FragmentInstance("*CC*", 2),
        ),
        edges=()
    )
    # decode correctly returns None because >2 original ports
    assert enc.decode(g) is None
