from types import SimpleNamespace

import numpy as np
import pytest

from polyevo.domain.connectivity import ConnectivitySpec, canonicalize
from polyevo.encoding.fragment_seq import FragmentLibrary, FragmentSequence


@pytest.mark.parametrize("domain,n", [("polymer_linear", 2), ("molecule", 0)])
def test_fragment_operators_1000_trials(config, domain, n):
    library = FragmentLibrary.from_entries(config["library"])
    spec = ConnectivitySpec(domain, n)
    encoding = FragmentSequence(library, spec)
    ctx = SimpleNamespace(library=library, rng=np.random.default_rng(123))
    for name, op in {**encoding.mutation_ops, **encoding.crossover_ops}.items():
        invalid = 0
        for _ in range(1000):
            a, b = encoding.random(ctx), encoding.random(ctx)
            child = op(a, b, ctx) if name in encoding.crossover_ops else op(a, ctx)
            invalid += canonicalize(encoding.decode(child), spec) is None
        assert invalid / 1000 < 0.3, (name, invalid)


def test_incompatible_brics_labels_are_not_arbitrarily_joined():
    library = FragmentLibrary.from_entries(["[3*]O[3*]"])
    encoding = FragmentSequence(library, ConnectivitySpec())
    assert encoding.decode((library.smiles[0], library.smiles[0])) is None


def test_invalid_fragment_library():
    for entries in ([], ["CC"], ["*CC*"], [{"smiles": "[16*]c1ccc([16*])cc1", "weight": -1}]):
        with pytest.raises(ValueError):
            FragmentLibrary.from_entries(entries)
