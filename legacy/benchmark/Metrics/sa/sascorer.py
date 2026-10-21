#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#
from __future__ import print_function

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.six.moves import cPickle
from rdkit.six import iteritems

import math
from collections import defaultdict

import os.path as op
import time

_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = cPickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m, verbose=False):
    if _fscores is None:
        readFragmentScores()
    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    if verbose:
        print('score1: ', score1)
        print('-sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty: ',
              sizePenalty, stereoPenalty, spiroPenalty, bridgePenalty, macrocyclePenalty)
        print('score2: ', score2)
        print('score3: ', score3)
        print('raw sascore: ', sascore)
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    if verbose:
        print('smooth sascore: ', sascore)
        print('norm sascore: ', round((10 - sascore) / 9, 2))
    return sascore


def processMols(mols):
    print('smiles\tName\tsa_score')
    for i, m in enumerate(mols):
        if m is None:
            continue

        s = calculateScore(m)

        smiles = Chem.MolToSmiles(m)
        print(smiles + "\t" + m.GetProp('_Name') + "\t%3f" % s)

def compute_sa_score(smiles):
    rdmol = Chem.MolFromSmiles(smiles)
    sa = calculateScore(rdmol)
    sa_norm = round((10 - sa) / 9, 2)
    return sa_norm


if __name__ == '__main__':
    models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind"]
    #models = ["pos", "neg"]
    #models = ["decompdiff"]
    #models = ["gasa_test1_neg", "gasa_test1_pos", "gasa_test2_neg", "gasa_test2_pos", 
    #          "gasa_test3_neg", "gasa_test3_pos", "nonpher_neg", "nonpher_pos"]
    #models = ["enamine", "compas"]
    #models = ["guide_12_23"]
    #models = ['decompdiff_guide']
    #models = ['dpo_300']
    #models = ['diffsbdd_caj']
    models = ['targetdiff_noedge']
    models = ['enamine_1w']
    models = ['targetdiff_dpo_1800']
    models = ['compas_filter', 'coconut_5w']
    models = ['decomp_0510']
    models = ['decomp_0509']
    models = ['decomp_origin']
    models = ['decomp_dpo']
    models = ['flowr', 'moldiff']
    models = ['neg_samples']

    models = ['8dyg_origin', '8dyg_dpo']
    start_time = time.time()
    for model in models:
        out = open(f"./{model}_sa.csv", "w")
        out.write("smiles,smiles_score\n")
        file = open(f"../smiles/{model}.txt")
        smiles = file.readlines()
        for i, smile in enumerate(smiles):
            if i% 1000 == 0:
                print(i)
            smile = smile.strip()
            try:
                sa = compute_sa_score(smile)
                out.write(f'{smile},{sa}\n')
            except:
                print(f"Error: {smile}")
            
        file.close()
        out.close()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
        


        



