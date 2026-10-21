# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import argparse
import multiprocessing as mp
import os.path
import pickle
from functools import partial

import alphaspace2 as al
import mdtraj
from rdkit import Chem

from utils.data import convert_sdf_to_pdb, PDBProtein
from utils.misc import *
from utils.preprocess import extract_subpockets, extract_submols, union_pocket_residues, mark_in_range
from utils.data import PDBProtein
from utils.data import ProteinLigandData, torchify_dict, parse_sdf_file
from utils.prior import compute_golden_prior_from_data
import utils.transforms as trans
import copy
from torch_geometric.transforms import Compose

KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}


def parse_pdbbind_index_file(raw_path, subset='refined'):
    all_index = []
    version = int(raw_path.rstrip('/')[-4:])
    assert version >= 2016
    if subset == 'refined':
        data_path = os.path.join(raw_path, f'refined-set')
        index_path = os.path.join(data_path, 'index', f'INDEX_refined_data.{version}')
    elif subset == 'general':
        data_path = os.path.join(raw_path, f'general-set-except-refined')
        index_path = os.path.join(data_path, 'index', f'INDEX_general_PL_data.{version}')
    else:
        raise ValueError(subset)

    all_files = os.listdir(data_path)
    with open(index_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        index, res, year, pka, kv = line.split('//')[0].strip().split()
        kind = [v for k, v in KMAP.items() if k in kv]
        assert len(kind) == 1
        if index in all_files:
            all_index.append({
                'pdb_index': index,
                'src_protein_filename': os.path.join(index, f'{index}_protein.pdb'),
                'src_ligand_filename': os.path.join(index, f'{index}_ligand.sdf'),
                'resolution': res,
                'pka': pka,
                'pkd_type': kind[0],
            })

    return sorted(all_index, key=lambda x: x['pdb_index'])


def parse_crossdocked_index_file(split_fn):
    all_index = []
    split_index = torch.load(split_fn)
    for k, v in split_index.items():
        for protein_fn, ligand_fn in v:
            protein_fn = os.path.join(os.path.dirname(protein_fn),
                                      os.path.basename(protein_fn)[:10] + '.pdb')
            all_index.append({
                'src_protein_filename': protein_fn,
                'src_ligand_filename': ligand_fn
            })
    return all_index


# @profile
def extract_subcomplex(config, protein_path, ligand_sdf_path,
                       save_snapshot_path=None, save_path=None, verbose=False):

    tmp_ligand_pdb_path = ligand_sdf_path.replace('.sdf', '.pdb')
    tmp_ligand_pdb_path = os.path.join('./tmp', os.path.basename(tmp_ligand_pdb_path))
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    try:
        # Convert sdf format ligand file to pdb format
        convert_sdf_to_pdb(ligand_sdf_path, tmp_ligand_pdb_path)
        # Load in receptor and ligand seperately using mdtraj
        receptor = mdtraj.load(protein_path)
        binder = mdtraj.load(tmp_ligand_pdb_path)
        os.remove(tmp_ligand_pdb_path)
        # Initialize a snapshot object, this will contain the receptor and the binder informations
        ss = al.Snapshot()
        for k, v in config.alphaspace.items():
            setattr(ss, k, v)
        # Run the snapshot object by feeding it receptor and binder mdtraj objects.
        ss.run(receptor=receptor, binder=binder)
    except:
        raise AlphaSpaceError

    pocket_list = [p for p in sorted(ss.pockets, key=lambda i: i.nonpolar_space, reverse=True) if p.isContact]
    if len(pocket_list) == 0:
        raise ExtractPocketError

    all_pocket_occupancy = []
    for p_idx, p in enumerate(pocket_list):
        all_pocket_occupancy.append(p.occupancy_nonpolar)
        if verbose:
            print(
                "Pocket {} has alpha-space of {} A3, BetaScore of {:.1f} kcal/mol and is {:.0f}% occupied".format(
                    p_idx, round(p.nonpolar_space), p.score, (p.occupancy_nonpolar * 100)
                )
            )
        # print(p.occupiedNonpolarSpace, p.nonpolar_space)
    if save_snapshot_path:
        al.write_snapshot(folder_path=save_snapshot_path, snapshot=ss, receptor=receptor, binder=binder)

    # ligand = parse_sdf_file(ligand_sdf_path)
    rdmol = next(iter(Chem.SDMolSupplier(ligand_sdf_path, removeHs=True, sanitize=True)))
    if rdmol is None:
        ligand_mol2_path = ligand_sdf_path.replace('.sdf', '.mol2')
        rdmol = Chem.MolFromMol2File(ligand_mol2_path)
    if rdmol is None:
        raise SDFParsingError

    # Extract sub-molecules
    all_frag_centroid, assignment, all_submol_atom_idx, all_submols, valid_pocket_id = \
        extract_submols(rdmol, pocket_list)

    # Extract pocket protein atoms / residues
    all_pocket_atom_serial, all_pocket_residues = [], []
    all_pocket_occupancy_by_submol = []
    protein = PDBProtein(protein_path)
    for i, pocket_id in enumerate(valid_pocket_id):
        pocket = pocket_list[pocket_id]
        selected_atom_serial, selected_residues = extract_subpockets(
            protein, pocket, method=config.pocket.protein_ext_method,
            submol=all_submols[i],
            mdtraj_protein=receptor, protein_radius=config.pocket.protein_radius,
        )

        all_pocket_atom_serial.append(selected_atom_serial)
        all_pocket_residues.append(selected_residues)

        centers = [a.centroid for a in pocket.alphas]
        contact_mask = mark_in_range(centers, all_submols[i].GetConformer(0).GetPositions())
        occupied_space = np.sum(np.array([alpha.nonpolar_space for alpha in pocket.alphas]) * contact_mask)
        occupancy = occupied_space / pocket.nonpolar_space
        all_pocket_occupancy_by_submol.append(occupancy)

    sub_pocket_dest_files, sub_ligand_dest = [], None
    ori_protein_fn = os.path.basename(protein_path)
    ori_ligand_fn = os.path.basename(ligand_sdf_path)
    union_pocket_dest, union_ligand_dest = None, None

    if save_path:
        if config.dataset.name == 'pdbbind':
            union_pocket_dest = os.path.join(save_path, ori_protein_fn)[:-4] + '_pocket.pdb'
        elif config.dataset.name == 'crossdocked':
            union_pocket_dest = os.path.join(save_path, ori_ligand_fn)[:-4] + '_pocket.pdb'
        else:
            raise ValueError
        union_ligand_dest = os.path.join(save_path, ori_ligand_fn)

        # write overall pocket and molecule
        os.makedirs(save_path, exist_ok=True)
        # shutil.copyfile(
        #     src=ligand_sdf_path,
        #     dst=union_ligand_dest
        # )
        w = Chem.SDWriter(union_ligand_dest)
        w.write(rdmol)
        w.close()
        if config.pocket.protein_ext_method == 'submol_radius':
            _, union_residues = protein.query_residues_centers(rdmol.GetConformer(0).GetPositions(),
                                                               config.pocket.protein_radius)
        else:
            union_residues = union_pocket_residues(all_pocket_residues)
        pdb_block_pocket = protein.residues_to_pdb_block(union_residues)
        with open(union_pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)

        # write sub-pocket
        for idx, pocket_residues in enumerate(all_pocket_residues):
            pocket_dest = union_pocket_dest[:-4] + '_%d.pdb' % idx
            # pocket_dest = os.path.join(save_path, pocket_fn)
            pdb_block_pocket = protein.residues_to_pdb_block(pocket_residues)
            with open(pocket_dest, 'w') as f:
                f.write(pdb_block_pocket)
            sub_pocket_dest_files.append(pocket_dest)

        # write sub-molecule
        ligand_fn = ori_ligand_fn[:-4] + '_frags.sdf'
        sub_ligand_dest = os.path.join(save_path, ligand_fn)
        w = Chem.SDWriter(sub_ligand_dest)
        for m in all_submols:
            if m is not None:
                w.write(m)
        w.close()

    return {
        'al_snapshot': ss,
        'all_pockets': [pocket_list[pocket_id] for pocket_id in valid_pocket_id],
        'all_pocket_atom_serial': all_pocket_atom_serial,
        'all_pocket_residues': all_pocket_residues,
        'all_submols': all_submols,
        'all_submol_atom_idx': all_submol_atom_idx,
        'protein_file': union_pocket_dest,
        'ligand_file': union_ligand_dest,
        'sub_pocket_files': sub_pocket_dest_files,
        'sub_ligand_file': sub_ligand_dest,

        'num_pockets': len(pocket_list),
        'num_frags': len(all_frag_centroid),
        'num_arms': len(valid_pocket_id),
        'num_scaffold': 1 if all_submols[-1] is not None else 0,
        'pocket_occupancies_by_mol': all_pocket_occupancy,
        'pocket_occupancies_by_submol': all_pocket_occupancy_by_submol
    }


# @profile
def process_item(item, config):
    if config.dataset.name == 'pdbbind':
        # pdb_idx, res, year, pka, kind = item
        pdb_idx = item['pdb_index']
        if config.dataset.split == 'refined':
            pdb_path = os.path.join(config.dataset.path, 'refined-set', pdb_idx)
        elif config.dataset.split == 'general':
            pdb_path = os.path.join(config.dataset.path, 'general-set-except-refined', pdb_idx)
        else:
            raise ValueError(args.subset)
        protein_path = os.path.join(pdb_path, f'{pdb_idx}_protein.pdb')
        ligand_path = os.path.join(pdb_path, f'{pdb_idx}_ligand.sdf')
        save_path = os.path.join(args.dest, pdb_idx)

    elif config.dataset.name == 'crossdocked':
        protein_path = os.path.join(config.dataset.path, item['src_protein_filename'])
        ligand_path = os.path.join(config.dataset.path, item['src_ligand_filename'])
        save_path = os.path.join(args.dest, os.path.dirname(item['src_ligand_filename']))
        pdb_idx = '%s + %s' % (os.path.basename(protein_path), os.path.basename(ligand_path))

    else:
        raise NotImplementedError

    try:
        # print(f'begin process PDB {pdb_idx}')
        r = extract_subcomplex(config, protein_path, ligand_path, save_path=save_path)
        print(f"PDB {pdb_idx}: num_pockets: {r['num_pockets']} num_frags: {r['num_frags']} "
              f"num_arms: {r['num_arms']} num_scaffold: {r['num_scaffold']}")
    except AlphaSpaceError:
        print(f"{pdb_idx}: AlphaSpace run fail!")
        r = 'al_fail'
    except SDFParsingError:
        print(f"{pdb_idx}: Parse sdf file fail!")
        r = 'sdf_fail'
    except DecomposeError:
        print(f"{pdb_idx}: Extract sub-molecule fail!")
        r = 'submol_fail'
    except ExtractPocketError:
        print(f"{pdb_idx}: Extract sub-pocket fail!")
        r = 'subpok_fail'
    except OSError as e:
        print(e)
        r = 'os_fail'
    except Exception:
        print(f"{pdb_idx}: Other fail!")
        r = 'other_fail'

    result_dict = {
        **item,
        'data': r
    }

    if isinstance(r, dict):
        # success
        prefix = os.path.basename(item['src_ligand_filename'])[:-4]
        meta_fn = os.path.join(save_path, prefix + '_meta.pkl')
        with open(meta_fn, 'wb') as f:
            pickle.dump(result_dict, f)

        result_index = {
            **item,
            'data': {
                'protein_file': r['protein_file'],
                'ligand_file': r['ligand_file'],
                'sub_pocket_files': r['sub_pocket_files'],
                'sub_ligand_file': r['sub_ligand_file'],
                'meta_file': meta_fn
            }
        }

    else:
        result_index = result_dict

    return result_index


if __name__ == '__maisn__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--dest', type=str, default='./data/crossdocked_v1.1_rmsd1.0_processed')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.dataset.name == 'pdbbind':
        all_index = parse_pdbbind_index_file(config.dataset.path, config.dataset.split)
    elif config.dataset.name == 'crossdocked':
        all_index = parse_crossdocked_index_file(config.dataset.split)
    else:
        raise NotImplementedError

    results = []
    with mp.Pool(args.num_workers) as pool:
        multiple_fns = [pool.apply_async(partial(process_item, config=config), (index,)) for index in all_index]
        for fn, index in tqdm(zip(multiple_fns, all_index), total=len(all_index)):
            try:
                result = fn.get(timeout=60)
            except mp.TimeoutError:
                results.append({
                    **index,
                    'data': 'timeout_fail'
                })
                continue
            results.append(result)

    valid_index_pocket = []
    al_fail, sdf_parsing_fail, timeout_fail, submol_fail, subpok_fail, os_fail, other_fail = [], [], [], [], [], [], []
    for index in results:
        protein_fn, ligand_fn, r = index['src_protein_filename'], index['src_ligand_filename'], index['data']
        if r == 'al_fail':
            al_fail.append((protein_fn, ligand_fn))
        elif r == 'sdf_fail':
            sdf_parsing_fail.append((protein_fn, ligand_fn))
        elif r == 'timeout_fail':
            timeout_fail.append((protein_fn, ligand_fn))
        elif r == 'submol_fail':
            submol_fail.append((protein_fn, ligand_fn))
        elif r == 'subpok_fail':
            subpok_fail.append((protein_fn, ligand_fn))
        elif r == 'os_fail':
            os_fail.append((protein_fn, ligand_fn))
        elif r == 'other_fail':
            other_fail.append((protein_fn, ligand_fn))
        elif isinstance(r, dict):
            valid_index_pocket.append(index)
        else:
            raise ValueError

    # merge refined set if necessary
    # if args.subset == 'general' and args.refined_index_pkl is not None:
    #     with open(args.refined_index_pkl, 'rb') as f:
    #         refined_index = pickle.load(f)
    #     valid_index_pocket += refined_index

    # save index
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    with open(os.path.join(args.dest, 'index.pkl'), 'wb') as f:
        pickle.dump(valid_index_pocket, f)

    all_fail_index = {
        'al_fail': al_fail,
        'sdf_parsing_fail': sdf_parsing_fail,
        'timeout_fail': timeout_fail,
        'submol_fail': submol_fail,
        'subpok_fail': subpok_fail,
        'os_fail': os_fail,
        'other_fail': other_fail
    }

    with open(os.path.join(args.dest, 'fail_index.pkl'), 'wb') as f:
        pickle.dump(all_fail_index, f)

    print('Done. %d protein-ligand pairs in total.' % len(valid_index_pocket))
    print(
        f'AlphaSpace fail: %d SDF parsing fail: %d SubMol extraction fail: %d SubPocket extraction fail: %d '
        f'Timeout fail: %d OS fail: %d Other fail: %d' % (
            len(al_fail), len(sdf_parsing_fail), len(submol_fail), len(subpok_fail),
            len(timeout_fail), len(os_fail), len(other_fail)
        )
    )

class ExtractSubcomplexDPO(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.noise_featureizer = trans.ComputeLigandAtomNoiseDist(version='ref_prior')
        self.protein_featurizer = trans.FeaturizeProteinAtom()
        self.ligand_featurizer = trans.FeaturizeLigandAtom("basic", False)
        self.decomp_indicator = trans.AddDecompIndicator(max_num_arms=10,
            global_prior_index=self.ligand_featurizer.ligand_feature_dim, add_ord_feat=False)
        self.bond_featurizer = trans.FeaturizeLigandBond('fc', set_bond_type=True)
        
    def __call__(self, data):

        data_w = copy.deepcopy(data)
        data_l = data
        data_w = self.processing(data_w, 'w')
        data_l = self.processing(data_l, 'l')

        if data_w is None or data_l is None:
            ret_data = ProteinLigandData()
            ret_data.error = True
            return ret_data
        def transform(data):
            data = self.noise_featureizer.__call__(data)
            data = self.protein_featurizer.__call__(data)
            data = self.ligand_featurizer.__call__(data)
            data = self.decomp_indicator.__call__(data)
            data = self.bond_featurizer.__call__(data)
            return data
    
        data_w = transform(data_w)
        data_l = transform(data_l)

        ret_data = ProteinLigandData()
        ret_data.error = False

        ret_data.protein_element=data_w.protein_element
        ret_data.protein_molecule_name='pocket'
        ret_data.protein_pos=data_w.protein_pos
        ret_data.protein_is_backbone=data_w.protein_is_backbone
        ret_data.protein_atom_name=data_w.protein_atom_name
        ret_data.protein_atom_to_aa_type=data_w.protein_atom_to_aa_type
        ret_data.protein_atom_feature=data_w.protein_atom_feature
        #ret_data.id=data_w.id

        ret_data.ligand_rdmol_w=data_w.ligand_rdmol
        ret_data.ligand_element_w=data_w.ligand_element
        ret_data.ligand_pos_w=data_w.ligand_pos
        ret_data.ligand_bond_index_w=data_w.ligand_bond_index
        ret_data.ligand_bond_type_w=data_w.ligand_bond_type
        ret_data.ligand_center_of_mass_w=data_w.ligand_center_of_mass
        ret_data.ligand_atom_feature_w=data_w.ligand_atom_feature
        ret_data.ligand_hybridization_w=data_w.ligand_hybridization
        ret_data.ligand_smiles_w=data_w.ligand_smiles
        ret_data.num_arms_w=data_w.num_arms
        ret_data.num_scaffold_w=data_w.num_scaffold
        ret_data.pocket_atom_masks_w=data_w.pocket_atom_masks
        ret_data.ligand_atom_mask_w=data_w.ligand_atom_mask
        ret_data.scaffold_prior_w=data_w.scaffold_prior
        ret_data.arms_prior_w=data_w.arms_prior
        ret_data.pocket_prior_masks_w=data_w.pocket_prior_masks
        ret_data.ligand_decomp_centers_w=data_w.ligand_decomp_centers
        ret_data.ligand_decomp_stds_w=data_w.ligand_decomp_stds
        ret_data.arm_num_atoms_w=data_w.arm_num_atoms
        ret_data.scaffold_num_atoms_w=data_w.scaffold_num_atoms
        ret_data.ligand_decomp_num_atoms_w=data_w.ligand_decomp_num_atoms
        ret_data.ligand_atom_feature_full_w=data_w.ligand_atom_feature_full
        ret_data.prior_group_idx_w=data_w.prior_group_idx
        ret_data.max_decomp_group_w=data_w.max_decomp_group
        ret_data.ligand_decomp_mask_w=data_w.ligand_decomp_mask
        ret_data.ligand_decomp_group_idx_w=data_w.ligand_decomp_group_idx
        ret_data.ligand_atom_aux_feature_w=data_w.ligand_atom_aux_feature
        ret_data.protein_decomp_group_idx_w=data_w.protein_decomp_group_idx
        ret_data.ligand_fc_bond_index_w=data_w.ligand_fc_bond_index
        ret_data.ligand_fc_bond_type_w=data_w.ligand_fc_bond_type

        ret_data.ligand_rdmol_l=data_l.ligand_rdmol
        ret_data.ligand_element_l=data_l.ligand_element
        ret_data.ligand_pos_l=data_l.ligand_pos
        ret_data.ligand_bond_index_l=data_l.ligand_bond_index
        ret_data.ligand_bond_type_l=data_l.ligand_bond_type
        ret_data.ligand_center_of_mass_l=data_l.ligand_center_of_mass
        ret_data.ligand_atom_feature_l=data_l.ligand_atom_feature
        ret_data.ligand_hybridization_l=data_l.ligand_hybridization
        ret_data.ligand_smiles_l=data_l.ligand_smiles
        ret_data.num_arms_l=data_l.num_arms
        ret_data.num_scaffold_l=data_l.num_scaffold
        ret_data.pocket_atom_masks_l=data_l.pocket_atom_masks
        ret_data.ligand_atom_mask_l=data_l.ligand_atom_mask
        ret_data.scaffold_prior_l=data_l.scaffold_prior
        ret_data.arms_prior_l=data_l.arms_prior
        ret_data.pocket_prior_masks_l=data_l.pocket_prior_masks
        ret_data.ligand_decomp_centers_l=data_l.ligand_decomp_centers
        ret_data.ligand_decomp_stds_l=data_l.ligand_decomp_stds
        ret_data.arm_num_atoms_l=data_l.arm_num_atoms
        ret_data.scaffold_num_atoms_l=data_l.scaffold_num_atoms
        ret_data.ligand_decomp_num_atoms_l=data_l.ligand_decomp_num_atoms
        ret_data.ligand_atom_feature_full_l=data_l.ligand_atom_feature_full
        ret_data.prior_group_idx_l=data_l.prior_group_idx
        ret_data.max_decomp_group_l=data_l.max_decomp_group
        ret_data.ligand_decomp_mask_l=data_l.ligand_decomp_mask
        ret_data.ligand_decomp_group_idx_l=data_l.ligand_decomp_group_idx
        ret_data.ligand_atom_aux_feature_l=data_l.ligand_atom_aux_feature
        ret_data.protein_decomp_group_idx_l=data_l.protein_decomp_group_idx
        ret_data.ligand_fc_bond_index_l=data_l.ligand_fc_bond_index
        ret_data.ligand_fc_bond_type_l=data_l.ligand_fc_bond_type
        


        #data = data.to_dict()  # avoid torch_geometric version issue
        #print(data)
        
        return ret_data

    def extract_subcomplex_dpo(self, data, wl='w', save_snapshot_path=None, verbose=False):

        '''config = load_config("configs/preprocessing/crossdocked.yml")
        protein_path = './data/crossdocked_v1.1_rmsd1.0_processed/DYR_STAAU_2_158_0/4xe6_X_rec_3fqc_55v_lig_tt_docked_4_pocket10.pdb'
        ligand_sdf_path = './outputs_pairwise/sdfs/0_w.sdf'
        '''
        protein_path = data.protein_file[:-4] + '10.pdb'
        ligand_sdf_path = os.path.join(os.environ.get("SYNC_DECOMPDIFF_PAIR_SDFS", "outputs_pairwise/sdfs"), f"{data.id}_{wl}.sdf")
        tmp_ligand_pdb_path = ligand_sdf_path.replace('.sdf', '.pdb')
        tmp_ligand_pdb_path = os.path.join('./tmp', os.path.basename(tmp_ligand_pdb_path))
        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')
        try:
            # Convert sdf format ligand file to pdb format
            convert_sdf_to_pdb(ligand_sdf_path, tmp_ligand_pdb_path)
            # Load in receptor and ligand seperately using mdtraj
            receptor = mdtraj.load(protein_path)
            binder = mdtraj.load(tmp_ligand_pdb_path)
            os.remove(tmp_ligand_pdb_path)
            # Initialize a snapshot object, this will contain the receptor and the binder informations
            ss = al.Snapshot()
            for k, v in self.config.alphaspace.items():
                setattr(ss, k, v)
            # Run the snapshot object by feeding it receptor and binder mdtraj objects.
            ss.run(receptor=receptor, binder=binder)
        except:
            raise AlphaSpaceError
        pocket_list = [p for p in sorted(ss.pockets, key=lambda i: i.nonpolar_space, reverse=True) if p.isContact]
        if len(pocket_list) == 0:
            raise ExtractPocketError
        all_pocket_occupancy = []
        for p_idx, p in enumerate(pocket_list):
            all_pocket_occupancy.append(p.occupancy_nonpolar)
            if verbose:
                print(
                    "Pocket {} has alpha-space of {} A3, BetaScore of {:.1f} kcal/mol and is {:.0f}% occupied".format(
                        p_idx, round(p.nonpolar_space), p.score, (p.occupancy_nonpolar * 100)
                    )
                )
            # print(p.occupiedNonpolarSpace, p.nonpolar_space)
        if save_snapshot_path:
            al.write_snapshot(folder_path=save_snapshot_path, snapshot=ss, receptor=receptor, binder=binder)
        # ligand = parse_sdf_file(ligand_sdf_path)
        rdmol = next(iter(Chem.SDMolSupplier(ligand_sdf_path, removeHs=True, sanitize=True)))
        if rdmol is None:
            ligand_mol2_path = ligand_sdf_path.replace('.sdf', '.mol2')
            rdmol = Chem.MolFromMol2File(ligand_mol2_path)
        if rdmol is None:
            raise SDFParsingError
        # Extract sub-molecules
        all_frag_centroid, assignment, all_submol_atom_idx, all_submols, valid_pocket_id = \
            extract_submols(rdmol, pocket_list)
        # Extract pocket protein atoms / residues
        all_pocket_atom_serial, all_pocket_residues = [], []
        all_pocket_occupancy_by_submol = []
        protein = PDBProtein(protein_path)
        for i, pocket_id in enumerate(valid_pocket_id):
            pocket = pocket_list[pocket_id]
            selected_atom_serial, selected_residues = extract_subpockets(
                protein, pocket, method=self.config.pocket.protein_ext_method,
                submol=all_submols[i],
                mdtraj_protein=receptor, protein_radius=self.config.pocket.protein_radius,
            )
            all_pocket_atom_serial.append(selected_atom_serial)
            all_pocket_residues.append(selected_residues)
            centers = [a.centroid for a in pocket.alphas]
            contact_mask = mark_in_range(centers, all_submols[i].GetConformer(0).GetPositions())
            occupied_space = np.sum(np.array([alpha.nonpolar_space for alpha in pocket.alphas]) * contact_mask)
            occupancy = occupied_space / pocket.nonpolar_space
            all_pocket_occupancy_by_submol.append(occupancy)
        sub_pocket_dest_files, sub_ligand_dest = [], None
        ori_protein_fn = os.path.basename(protein_path)
        ori_ligand_fn = os.path.basename(ligand_sdf_path)
        union_pocket_dest, union_ligand_dest = None, None
        return {
            'al_snapshot': ss,
            'all_pockets': [pocket_list[pocket_id] for pocket_id in valid_pocket_id],
            'all_pocket_atom_serial': all_pocket_atom_serial,
            'all_pocket_residues': all_pocket_residues,
            'all_submols': all_submols,
            'all_submol_atom_idx': all_submol_atom_idx,
            'protein_file': union_pocket_dest,
            'ligand_file': union_ligand_dest,
            'sub_pocket_files': sub_pocket_dest_files,
            'sub_ligand_file': sub_ligand_dest,
            'num_pockets': len(pocket_list),
            'num_frags': len(all_frag_centroid),
            'num_arms': len(valid_pocket_id),
            'num_scaffold': 1 if all_submols[-1] is not None else 0,
            'pocket_occupancies_by_mol': all_pocket_occupancy,
            'pocket_occupancies_by_submol': all_pocket_occupancy_by_submol
        }
    
    def processing(self, data, wl):
        protein_path = data.protein_file[:-4] + '10.pdb'
        ligand_path = os.path.join(os.environ.get("SYNC_DECOMPDIFF_PAIR_SDFS", "outputs_pairwise/sdfs"), f"{data.id}_{wl}.sdf")
        #r = self.extract_subcomplex_dpo(data, wl)
        try:
        # print(f'begin process PDB {pdb_idx}')
            r = self.extract_subcomplex_dpo(data, wl)
            #print(f"PDB {pdb_idx}: num_pockets: {r['num_pockets']} num_frags: {r['num_frags']} "
            #      f"num_arms: {r['num_arms']} num_scaffold: {r['num_scaffold']}")
        except AlphaSpaceError:
            print(f"{protein_path}: AlphaSpace run fail!")
            r = 'al_fail'
        except SDFParsingError:
            print(f"{protein_path}: Parse sdf file fail!")
            r = 'sdf_fail'
        except DecomposeError:
            print(f"{protein_path}: Extract sub-molecule fail!")
            r = 'submol_fail'
        except ExtractPocketError:
            print(f"{protein_path}: Extract sub-pocket fail!")
            r = 'subpok_fail'
        except OSError as e:
            print(e)
            r = 'os_fail'
        except Exception:
            print(f"{protein_path}: Other fail!")
            r = 'other_fail'
        if 'fail' in r:
            return None
        num_arms, num_scaffold = r['num_arms'], r['num_scaffold']
        protein = PDBProtein(protein_path)
        protein_dict = protein.to_dict_atom()
        ligand_dict = parse_sdf_file(ligand_path, kekulize=True)
        num_protein_atoms, num_ligand_atoms = len(protein.atoms), ligand_dict['rdmol'].GetNumAtoms()
        assert num_ligand_atoms == sum([len(x) for x in r['all_submol_atom_idx']])
        # extract pocket atom mask
        protein_atom_serial = [atom['atom_id'] for atom in protein.atoms]
        pocket_atom_masks = []
        assert len(r['all_pocket_atom_serial']) == num_arms
        for pocket_atom_serial in r['all_pocket_atom_serial']:
            pocket_atom_idx = [protein_atom_serial.index(i) for i in pocket_atom_serial]
            pocket_atom_mask = torch.zeros(num_protein_atoms, dtype=torch.bool)
            pocket_atom_mask[pocket_atom_idx] = 1
            pocket_atom_masks.append(pocket_atom_mask)
        pocket_atom_masks = torch.stack(pocket_atom_masks)
        # extract ligand atom mask
        ligand_atom_mask = torch.zeros(num_ligand_atoms, dtype=int)
        for arm_idx, atom_idx in enumerate(r['all_submol_atom_idx']):
            if arm_idx == len(r['all_submol_atom_idx']) - 1:
                ligand_atom_mask[atom_idx] = -1
            else:
                ligand_atom_mask[atom_idx] = arm_idx
        assert len(ligand_atom_mask.unique()) == num_arms + num_scaffold
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(protein_dict),
            ligand_dict=torchify_dict(ligand_dict),
        )
        data.src_protein_filename = protein_path
        data.src_ligand_filename = ligand_path
        for k, v in r.items():
            data[k] = v
        data.num_arms, data.num_scaffold = num_arms, num_scaffold
        data.pocket_atom_masks, data.ligand_atom_mask = pocket_atom_masks, ligand_atom_mask
        data = compute_golden_prior_from_data(data)

        return data

if __name__ == '__main__':

    pass
    #print(extract_subcomplex_dpo())