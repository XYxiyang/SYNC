"""Run SYNC-guided TargetDiff on a tensor-format protein pocket."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'legacy/targetdiff'))

import numpy as np
import torch
import yaml
from easydict import EasyDict
from rdkit import Chem
from datasets.pl_data import ProteinLigandData
from models.molopt_score_model import ScorePosNet3D
from synsbdd.models.syn3d_lin import Syn3dlin
from scripts.sample_diffusion import sample_diffusion_ligand
from utils import reconstruct, transforms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pocket', type=Path, default=ROOT/'examples/1wn6_pocket.pt')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'legacy/targetdiff/pretrained_models/pretrained_diffusion.pt')
    p.add_argument('--classifier', type=Path, default=ROOT/'checkpoints/sync_noedge.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-samples', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--guide', choices=['guide', 'origin'], default='guide')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        p.error('num-samples and batch-size must be positive')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    pocket = torch.load(args.pocket, map_location='cpu', weights_only=True)
    data = ProteinLigandData(**pocket['data'])
    with torch.serialization.safe_globals([EasyDict]):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ScorePosNet3D(checkpoint['config'].model, data.protein_atom_feature.shape[1], 13)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(args.device).eval()
    config = EasyDict(yaml.safe_load((ROOT/'legacy/targetdiff/configs/sampling.yml').read_text()))
    config.classifier.encoder.device = args.device
    classifier = Syn3dlin(config.classifier)
    state = torch.load(args.classifier, map_location='cpu', weights_only=True)
    if state.get('representation') != 'aromatic-start1-bonds-noedge':
        raise ValueError('Use the SYNC no-edge classifier checkpoint.')
    classifier.load_state_dict(state['state_dict'], strict=True)
    classifier = classifier.to(args.device).eval()
    # Seed after model construction, matching the validation run.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    start = time.time()
    result = sample_diffusion_ligand(model, classifier, data, args.num_samples,
        batch_size=args.batch_size, device=args.device, num_steps=1000, pos_only=False,
        center_pos_mode='protein', sample_num_atoms='prior', guide=args.guide, params={})
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'positions':[torch.tensor(x) for x in result[0]],
                'atom_types':[torch.tensor(x) for x in result[1]],
                'position_trajectories':[torch.tensor(x) for x in result[2]],
                'atom_type_trajectories':[torch.tensor(x) for x in result[3]]}, args.output/'samples.pt')
    records=[]
    with Chem.SDWriter(str(args.output/'samples.sdf')) as writer:
        for i,(positions,types) in enumerate(zip(result[0], result[1])):
            record={'sample_index':i,'atoms':len(types)}
            try:
                mol=reconstruct.reconstruct_from_generated(positions,
                    transforms.get_atomic_number_from_index(types,mode='add_aromatic'),
                    transforms.is_aromatic_from_index(types,mode='add_aromatic'),False)
                mol.SetProp('_Name',f'SYNC_{i}')
                writer.write(mol)
                record.update(status='ok',smiles=Chem.MolToSmiles(mol),connected=len(Chem.GetMolFrags(mol))==1)
            except reconstruct.MolReconsError as error:
                record.update(status='reconstruction_failed',error=str(error))
            records.append(record)
    summary={'seed':args.seed,'guide':args.guide,'diffusion_steps':1000,
             'elapsed_seconds':time.time()-start,'records':records}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
