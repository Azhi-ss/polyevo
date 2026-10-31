from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors


@dataclass(frozen=True)
class ConnectivitySpec:
    name: str = "polymer_linear"
    n_attachment: int = 2
    attachment_symbol: str = "*"
    forbid_adjacent: bool = True
    remove_stereo: bool = False

    def __post_init__(self):
        expected = {"polymer_linear": 2, "molecule": 0}
        if self.name not in expected or self.n_attachment != expected[self.name]:
            raise ValueError("Supported domains: polymer_linear (2 attachments), molecule (0)")
        if self.attachment_symbol != "*":
            raise ValueError("Only '*' attachment symbols are supported")


def canonicalize(smiles: str | None, spec: ConnectivitySpec,
                 constraints=()) -> str | None:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            return None
        stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(stars) != spec.n_attachment:
            return None
        for atom in stars:
            if atom.GetDegree() != 1 or atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
                return None
            if spec.forbid_adjacent and atom.GetNeighbors()[0].GetAtomicNum() == 0:
                return None
            atom.SetIsotope(0)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        if spec.remove_stereo:
            Chem.RemoveStereochemistry(mol)
        for rule in constraints:
            if "smarts_forbidden" in rule:
                if any(mol.HasSubstructMatch(Chem.MolFromSmarts(s))
                       for s in rule["smarts_forbidden"]):
                    return None
            else:
                value = getattr(Descriptors, rule["property"])(mol)
                lo, hi = rule["range"]
                if not lo <= value <= hi:
                    return None
        return Chem.MolToSmiles(mol, canonical=True)
    except (ValueError, RuntimeError):
        return None
