"""SDF conversion using the original aromatic vocabulary and real bonds."""
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

VOCAB = {
    token: i for i, token in enumerate(
        Path(__file__).with_name("smiles_vocab_with_aro.txt").read_text().splitlines(), 1
    )
}


def molecule_to_graph(mol):
    if mol is None:
        raise ValueError("RDKit could not parse this SDF record.")
    mol = Chem.RemoveHs(mol)
    if not mol.GetNumAtoms():
        raise ValueError("Molecule has no atoms after removing hydrogens.")
    if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError("A 3D conformer is required; 2D coordinates are not a SYNC input.")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite atom coordinates.")
    tokens = [a.GetSymbol() + ("pi" if a.GetIsAromatic() else "") for a in mol.GetAtoms()]
    missing = sorted(set(tokens).difference(VOCAB))
    if missing:
        raise ValueError(f"Unsupported atom tokens: {missing}")
    edges = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(a, b), (b, a)])
    return {
        "atom_types": torch.tensor([VOCAB[t] for t in tokens], dtype=torch.long),
        "positions": torch.from_numpy(coords.copy()),
        "edge_index": torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        "smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
    }


def collate_graphs(graphs, device="cpu"):
    if not graphs:
        raise ValueError("Empty graph batch.")
    nodes, positions, edges, batches = [], [], [], []
    offset = 0
    for i, graph in enumerate(graphs):
        n = len(graph["atom_types"])
        if n < 1 or graph["positions"].shape != (n, 3):
            raise ValueError("Atom tokens and 3D coordinates must have matching, nonzero lengths.")
        edge_index = graph["edge_index"]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n):
            raise ValueError("Graph contains out-of-range edges.")
        nodes.append(graph["atom_types"])
        positions.append(graph["positions"])
        edges.append(graph["edge_index"] + offset)
        batches.append(torch.full((n,), i, dtype=torch.long))
        offset += n
    return tuple(x.to(device) for x in (
        torch.cat(nodes), torch.cat(positions), torch.cat(edges, dim=1), torch.cat(batches)
    ))


def sdf_records(path):
    path = Path(path)
    files = sorted(path.rglob("*.sdf")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No SDF files found in {path}")
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
        for index, mol in enumerate(Chem.SDMolSupplier(str(file), removeHs=True)):
            yield file, index, mol
