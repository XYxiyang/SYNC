import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch

from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict
from utils import transforms, reconstruct
from rdkit import Chem



class PairDataset(Dataset):

    def __init__(self, raw_path, transform=None, mode='full',
                 include_dummy_atoms=False, kekulize=True, version='v1'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.mode = mode  # ['arms', 'scaffold', 'full']
        self.include_dummy_atoms = include_dummy_atoms
        self.kekulize = kekulize
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_{mode}_{version}.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + f'_{mode}_{version}_name2id.pt')

        self.transform = transform
        self.mode = mode
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        print('Load dataset from %s' % self.processed_path)
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        #print(data)
        #ligand_path = os.path.join("outputs_pairwise", f"result_{idx}.pt")
        #ligand = torch.load(ligand_path)
        #print(ligand[0][0])
        #data.ligand_w_pos = ligand[0][0]['pred_pos']
        #data.ligand_w_v = ligand[0][0]['pred_v']
        #data.ligand_w_bond_index = ligand[0][0]['pred_bond_index']
        #data.ligand_w_bond_type = ligand[0][0]['pred_bond_type']
        #data.ligand_w_decomp_mask = ligand[0][0]['decomp_mask']
        #data.ligand_l_pos = ligand[1][0]['pred_pos']
        #data.ligand_l_v = ligand[1][0]['pred_v']
        #data.ligand_l_bond_index = ligand[1][0]['pred_bond_index']
        #data.ligand_l_bond_type = ligand[1][0]['pred_bond_type']
        #data.ligand_l_decomp_mask = ligand[1][0]['decomp_mask']
        
        #data.ligand_w_pos = torch.tensor(ligand['pred_ligand_pos_traj_w'][0][-1], dtype=torch.float32)
        #data.ligand_w_atom_feature_full = torch.tensor(ligand['pred_ligand_v_traj_w'][0][-1], dtype=torch.int64)
        #data.ligand_l_pos = torch.tensor(ligand['pred_ligand_pos_traj_l'][0][-1], dtype=torch.float32)
        #data.ligand_l_atom_feature_full = torch.tensor(ligand['pred_ligand_v_traj_l'][0][-1], dtype=torch.int64)
        
        return data


    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        if self.transform is not None:
            data = self.transform(data)
        return data
        
def smiles(pred_v, pred_pos):
    pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
    pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode='add_aromatic')
    try:
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
    except:
        pass
    print("neg:", Chem.MolToSmiles(mol))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
