import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from sync_sbdd.data import collate_graphs, molecule_to_graph
from sync_sbdd.model import load_model, save_model

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


@pytest.fixture
def model():
    return load_model(ROOT / "checkpoints/sync_noedge.pt")[0]


@pytest.fixture
def graphs():
    return [molecule_to_graph(m) for m in Chem.SDMolSupplier(str(ROOT / "examples/molecules.sdf"))]


def test_original_checkpoint_predictions(model, graphs):
    expected = json.loads((ROOT / "tests/reference_logits.json").read_text())
    with torch.no_grad():
        actual = model(*collate_graphs(graphs))
    torch.testing.assert_close(actual, torch.tensor(expected["logits"]), atol=2e-5, rtol=2e-5)


def test_translation_rotation_invariance(model, graphs):
    features = list(collate_graphs(graphs))
    with torch.no_grad():
        baseline = model(*features)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        features[1] = features[1] @ rotation + torch.tensor([4., -7., 2.])
        torch.testing.assert_close(model(*features), baseline, atol=3e-5, rtol=3e-5)


def test_permutation_invariance(model, graphs):
    g = graphs[-1]
    torch.manual_seed(10)
    order = torch.randperm(len(g["atom_types"]))
    inverse = torch.argsort(order)
    permuted = dict(atom_types=g["atom_types"][order], positions=g["positions"][order],
                    edge_index=inverse[g["edge_index"]])
    with torch.no_grad():
        torch.testing.assert_close(model(*collate_graphs([g])),
                                   model(*collate_graphs([permuted])), atol=2e-5, rtol=2e-5)


def test_batch_independence(model, graphs):
    with torch.no_grad():
        individual = torch.cat([model(*collate_graphs([g])) for g in graphs])
        together = model(*collate_graphs(graphs))
    torch.testing.assert_close(individual, together, atol=2e-5, rtol=2e-5)


def test_differentiable_coordinates_and_training(model, graphs):
    features = list(collate_graphs(graphs))
    features[1].requires_grad_(True)
    before = model.head.mlp[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = torch.nn.functional.cross_entropy(model(*features), torch.tensor([0, 1, 0]))
    loss.backward()
    assert torch.isfinite(features[1].grad).all()
    assert features[1].grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(model.head.mlp[-1].weight, before)


def test_state_checkpoint_roundtrip(model, graphs, tmp_path):
    path = tmp_path / "roundtrip.pt"
    save_model(path, model)
    loaded, _ = load_model(path)
    with torch.no_grad():
        torch.testing.assert_close(model(*collate_graphs(graphs)), loaded(*collate_graphs(graphs)))


def test_aromatic_token_mapping(graphs):
    assert graphs[0]["atom_types"].tolist() == [2, 2, 6]
    assert graphs[1]["atom_types"].tolist() == [3] * 6


def test_single_atom_no_edges(model):
    mol = Chem.MolFromSmiles("[Na+]")
    conformer = Chem.Conformer(1)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    graph = molecule_to_graph(mol)
    assert graph["edge_index"].shape == (2, 0)
    assert torch.isfinite(model(*collate_graphs([graph]))).all()


def test_invalid_molecule_rejected():
    with pytest.raises(ValueError):
        molecule_to_graph(None)


def test_malformed_legacy_graph_rejected(graphs):
    malformed = dict(graphs[0])
    malformed["positions"] = malformed["positions"][:-1]
    with pytest.raises(ValueError, match="matching"):
        collate_graphs([malformed])


def test_2d_coordinates_rejected():
    mol = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(mol)
    with pytest.raises(ValueError, match="3D"):
        molecule_to_graph(mol)


def test_cli_reports_every_record(tmp_path):
    output = tmp_path / "scores.csv"
    subprocess.run([sys.executable, "-m", "sync_sbdd.cli", "--input",
                    str(ROOT / "examples/molecules.sdf"), "--checkpoint",
                    str(ROOT / "checkpoints/sync_noedge.pt"), "--output", str(output),
                    "--batch-size", "2"], check=True)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["status"] == "ok" for row in rows)
    assert json.loads(output.with_suffix(".summary.json").read_text())["failed"] == 0
