import numpy as np
import os
import pandas as pd
import scipy.stats as sp
from tqdm.auto import tqdm
from sklearn.cluster import AgglomerativeClustering

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MCS
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

####### Fragment and Compound Filtering Functions #######

def threshold_on_score(df, thresh, column):
    keep_indices = [x != 'Invalid SMILES' for x in list(df[column])]
    df = df[keep_indices]
    keep_indices = [float(s) > thresh for s in list(df[column])]
    df = df[keep_indices]
    print('length of df >' + str(thresh) + ': ', len(df))
    return(df)

def filter_require_more_than_coh(df, smiles_column):
    coh_characters = ['C', 'c', 'O', 'o', 'H', '(', ')', '[', ']', '=', '#', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    smis = list(df[smiles_column])
    def require_more_than_coh_one_mol(smi):
        for char in smi:
            if char not in coh_characters:
                return(True) # good, more interesting
        return(False) # bad, only COH
    
    keep_indices = [require_more_than_coh_one_mol(smi) for smi in smis]
    df = df[keep_indices]
    print('length of df with more than C,O,H characters: ', len(df))
    return(df)        
                
def keep_valid_molecules(df, smiles_column):
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    keep_indices = [m is not None for m in mols]
    df = df[keep_indices]
    print('length of df with valid mols: ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    for m, smi in zip(mols, smis):
        m.SetProp('SMILES', smi)
    return(df, mols)

def check_pains_brenk(df, mols, method = 'both'):
    # initialize filter
    params = FilterCatalogParams()
    if method == 'both' or method == 'pains':
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    if method == 'both' or method == 'brenk':
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)

    def search_for_pains_or_brenk(mol):
        entry = catalog.GetFirstMatch(mol)  # Get the first matching PAINS or Brenk
        if entry is not None:
            return(False) # contains bad
        else:
            return(True) # clean

    keep_indices = [search_for_pains_or_brenk(m) for m in mols]
    df = df[keep_indices]
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    print('length of all preds with clean (no PAINS or Brenk) mols: ', len(df))
    return(df, mols)

def deduplicate_mols_on_structure(df, mols):
    
    fps = [Chem.RDKFingerprint(m) for m in mols]
    keep_indices = [True] * len(fps)

    for i, fp1 in enumerate(fps):
        tans = DataStructs.BulkTanimotoSimilarity(fp1, fps)
        for j, tan in enumerate(tans):
            if tan < 1.0 or i == j:
                continue
            else:
                if keep_indices[i]:
                    keep_indices[j] = False
                    
    df = df[keep_indices]
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    print('length of all preds deduplicated: ', len(df))   
    return(df, mols)

def is_pattern(mol, pattern_mol):
    if mol is None:
        return(False)
    num_atoms_frag = pattern_mol.GetNumAtoms()
    mcs = MCS.FindMCS([mol, pattern_mol], atomCompare='elements',completeRingsOnly = True)
    if mcs.smarts is None:
        return(False)
    mcs_mol = Chem.MolFromSmarts(mcs.smarts)
    num_atoms_mcs = mcs_mol.GetNumAtoms()
    if num_atoms_frag == num_atoms_mcs:
        return(True)
    else:
        return(False)

def filter_for_pattern(df, mols, pattern_smi):
    pattern_mol = Chem.MolFromSmiles(pattern_smi)
    if pattern_mol is None:
        print('Pattern smiles is invalid.')
        return(df, mols)
    keep_indices = [not is_pattern(mol, pattern_mol) for mol in mols]
    df = df[keep_indices]
    print('length of df with no ' + pattern_smi + ': ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)   

def filter_for_druglikeness(df, mols, smiles_column, criteria):
    smis = list(df[smiles_column])
    if criteria == 'egan':
        druglikeness = [ADME(smi).druglikeness_egan() for smi in smis]
    elif criteria == 'ghose':
        druglikeness = [ADME(smi).druglikeness_ghose() for smi in smis]
    elif criteria == 'lipinski':
        druglikeness = [ADME(smi).druglikeness_lipinski() for smi in smis]
    elif criteria == 'muegge':
        druglikeness = [ADME(smi).druglikeness_muegge() for smi in smis]
    else:
        print('Criteria ' + criteria + ' not supported for drug-likeness filtering.')
        return(df, mols)
    df = df[druglikeness]
    print('length of df satisfying druglikeness ' + criteria + ': ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)

def process_dataset(frag_or_cpd, path, score, smi_col, hit_col, require_more_than_coh, remove_pains_brenk, remove_patterns, druglikeness_filter):
    
    # process fragments first
    df = pd.read_csv(path)
    print('length of df: ', len(df))

    # keep copy of full df for cpds
    fulldf = df.copy()

    # threshold on score
    df = threshold_on_score(df, score, hit_col)

    # keep only frags with more than just C, O, H
    if require_more_than_coh and frag_or_cpd == 'cpd':
            print('Removing compounds with more than C, O, H is not supported because it is redundant. Try removing only these fragments.')
    if require_more_than_coh:
        df = filter_require_more_than_coh(df, smi_col)

    # keep only valid frags
    df, mols = keep_valid_molecules(df, smi_col)

    # keep only frags without pains or brenk
    if remove_pains_brenk != 'none':
        df, mols = check_pains_brenk(df, mols, method = remove_pains_brenk)

    # remove common abx
    if frag_or_cpd == 'cpd' and len(remove_patterns) > 0:
        print('Removing patterns from compounds is not supported because it is redundant. Try removing patterns from fragments only.')
    elif frag_or_cpd == 'frag':
        for patt_smi in remove_patterns:
            df, mols = filter_for_pattern(df, mols, patt_smi)

    # filter on druglikeness
    for criteria in druglikeness_filter:
        df, mols = filter_for_druglikeness(df, mols, smi_col, criteria)
    
    return(df, mols, fulldf)

 ####### Cpd + Fragment Matching Functions #######

def match_frags_and_mols(frags, cpd_mols):
    
    matches=[] # very slow because n frags x n mols
    frag_match_indices=[]
    for frag_index, frag in enumerate(frags):
        matched_with_this_frag=[]
        for full_mol_index, full_mol in enumerate(cpd_mols):
            if full_mol.HasSubstructMatch(frag): # contains entirely the fragment
                matched_with_this_frag.append(full_mol_index)
        if len(matched_with_this_frag) > 0:
            matches.append(matched_with_this_frag)
            frag_match_indices.append(frag_index)

    print('number of matched fragments: ', len(matches))
    return(frag_match_indices, matches)

def check_for_complete_ring_fragments(frags, frag_match_indices, cpd_mols, cpd_match_indices_lists):
    def check_frag_does_not_disrupt_mol(frag, mol):    
        mcs = rdFMCS.FindMCS([frag,mol], completeRingsOnly=True)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        num_atoms_mcs = mcs_mol.GetNumAtoms()
        num_atoms_frag = frag.GetNumAtoms()
        return(num_atoms_frag == num_atoms_mcs)  

    new_frag_match_indices = []
    new_cpd_match_indices_lists = []
    for frag_match_index, full_mol_index_list in zip(frag_match_indices, cpd_match_indices_lists):
        frag = frags[frag_match_index]
        mols = [m for i,m in enumerate(cpd_mols) if i in full_mol_index_list]
        keep_indices = [check_frag_does_not_disrupt_mol(frag, mol) for mol in mols]
        new_index_list = [i for (i, v) in zip(full_mol_index_list, keep_indices) if v]
        if len(new_index_list) > 0:
            new_frag_match_indices.append(frag_match_index)
            new_cpd_match_indices_lists.append(new_index_list)
    return(new_frag_match_indices, new_cpd_match_indices_lists)

def compile_results_into_df(df, cpd_df, mols, frag_match_indices, cpd_match_indices_lists, result_path, frag_hit_column, cpd_hit_column):
    rank_df = pd.DataFrame()
    rank_df['matched_fragments'] = frag_match_indices
    rank_df['fragment_SMILES'] = [mols[i].GetProp('SMILES') for i in list(rank_df['matched_fragments'])]
    rank_df['length_of_fragment'] = [mols[i].GetNumAtoms() for i in frag_match_indices]
    rank_df['matched_molecules'] = cpd_match_indices_lists
    rank_df['number_of_matched_molecules'] = [len(m) for m in cpd_match_indices_lists]
    rank_df['fragment_scores'] = [df.iloc[i,list(df.columns).index(frag_hit_column)] for i in frag_match_indices]
    rank_df['full_molecule_scores'] = [[cpd_df.iloc[i,list(cpd_df.columns).index(cpd_hit_column)] for i in sublist] for sublist in cpd_match_indices_lists]
    rank_df['average_molecule_score'] = [np.mean([float(x) for x in sublist]) for sublist in list(rank_df['full_molecule_scores'])]

    # save rank_df
    rank_df = rank_df.sort_values('number_of_matched_molecules', ascending = False)
    rank_df.to_csv(result_path + 'candidates_after_matching.csv', index = False)

    print('Previewing dataframe so far...')
    display(rank_df.head(5))
    return(rank_df)

####### Filtering (abx, train set, toxicity) Functions #######

def check_for_toxicity(df, fragment_column, frag_mols):
    # make dictionary mapping SMILES strings to list of hep_means and primary_means
    # hardcoded file paths
    hepg2 = pd.read_excel('../data/static_datasets/HEPG2_Primary_Toxicity_Data.xlsx', sheet_name = 'HepG2')
    hepg2['hepg2_mean'] = hepg2['Mean_10uM']
    primary = pd.read_excel('../data/static_datasets/HEPG2_Primary_Toxicity_Data.xlsx', sheet_name = 'Primary')
    primary['primary_mean'] = primary['Mean_10uM']
    tox = hepg2.merge(primary, on = 'Compound_ID', how = 'left')
    tox = tox.drop_duplicates('Compound_ID')
    tox['both_means'] = [[i,j] for i,j in zip(tox['hepg2_mean'], tox['primary_mean'])]
    tox_dict = dict(zip(list(tox['SMILES_x']), list(tox['both_means'])))
    
    tox_smis = tox_dict.keys()
    tox_mols = [Chem.MolFromSmiles(smi) for smi in tox_smis]
    hepg2_tox_matches=[]
    prim_tox_matches=[]
    
    for frag_index in list(df[fragment_column]): # only do it for the short-listed frags
        frag = frag_mols[frag_index]
        hepg2_matched_with_this_frag=[]
        prim_matched_with_this_frag=[]
        for tox_smi, tox_mol in zip(tox_smis, tox_mols):
            if tox_mol is None:
                continue
            # this match can disrupt a ring - not picky about that
            if tox_mol.HasSubstructMatch(frag): # contains entirely the fragment
                hepg2_matched_with_this_frag.append(tox_dict[tox_smi][0])
                prim_matched_with_this_frag.append(tox_dict[tox_smi][1])
        hepg2_tox_matches.append(hepg2_matched_with_this_frag)
        prim_tox_matches.append(prim_matched_with_this_frag)

    df['hepg2_tox_matched_molecule_values'] = hepg2_tox_matches
    df['primary_tox_matched_molecule_values'] = prim_tox_matches
    df['average_hepg2_tox_score_of_matched_molecules'] = [np.mean(sublist) for sublist in list(df['hepg2_tox_matched_molecule_values'])]
    df['average_primary_tox_score_of_matched_molecules'] = [np.mean(sublist) for sublist in list(df['primary_tox_matched_molecule_values'])]
    return(df)

def filter_out_toxicity(df, tox_thresh, fragment_index_column, frag_mols):
    df = check_for_toxicity(df, fragment_index_column, frag_mols)
    # do for those molecules with good tox or no tox
    keep_indices = []

    for i, row in df.iterrows():
        hep = row['average_hepg2_tox_score_of_matched_molecules']
        prim = row['average_primary_tox_score_of_matched_molecules']

        if np.isnan(hep) or np.isnan(prim):
            keep_indices.append(i)
        elif hep > tox_thresh and prim > tox_thresh:
            keep_indices.append(i)
    df = df.iloc[keep_indices,:]
    print('number of fragments passing both toxicity filters under ' + str(tox_thresh) + ': ' + str(len(df)))
    return(df)

# for any set of frags and any set of mols
def check_for_fuzzy_frags_in_mols(frags, mols):
    matches=[]
    for frag in frags: # only do it for the short-listed frags
        matched_with_this_frag=[]
        for index, mol in enumerate(mols):
            if mol.HasSubstructMatch(frag): # contains entirely the fragment
                matched_with_this_frag.append(index)
        matches.append(matched_with_this_frag)
    return(matches)

def check_for_frags_in_known_abx(df, fragment_index_column, frag_mols, abx_mols):
    frags = [frag_mols[frag_index] for frag_index in list(df[fragment_index_column])]
    abx_matches = check_for_fuzzy_frags_in_mols(frags, abx_mols)
    df['matched_antibiotics'] = abx_matches
    return(df)

# for two sets of mols - this is a typo btw, it should say get HIGHEST tanimoto
def for_mol_list_get_lowest_tanimoto_to_closest_mol(cpd_mols, molecules_to_check):
    mols_fps = [Chem.RDKFingerprint(mo) for mo in cpd_mols]
    check_fps = [Chem.RDKFingerprint(mo) for mo in molecules_to_check]
    best_sims = []
    for m in mols_fps:
        if m is None:
            best_sims.append(-1)
        curr_highest_sim = 0
        for check in check_fps:
            sim = DataStructs.FingerprintSimilarity(m,check)
            if sim > curr_highest_sim: # optionally, could implement a thresholding here
                curr_highest_sim = sim
        best_sims.append(curr_highest_sim)
    return(best_sims)

def check_full_cpd_similarity_to_closest_abx(df, full_cpd_index_column, cpd_mols, abx_mols):
    tan_scores = [for_mol_list_get_lowest_tanimoto_to_closest_mol([cpd_mols[i] for i in sublist], abx_mols) for sublist in list(df[full_cpd_index_column])]
    df['tanimoto_scores_of_full_mols_to_nearest_abx'] = tan_scores
    return(df)

def check_abx(abx_path, abx_smiles_col, abx_name_col, df, fragment_index_column, frag_mols, full_cpd_index_column, cpd_mols):
    if abx_path == '':
        print('No antibiotics have been provided for a comparison.')
        return(df)
    # hardcoded file path
    abx = pd.read_csv(abx_path)
    print('number of abx: ', len(abx))

    abx_mols = [Chem.MolFromSmiles(smi) for smi in list(abx[abx_smiles_col])]
    keep_indices = [m is not None for m in abx_mols]
    abx = abx[keep_indices]
    abx_mols = [m for i,m in enumerate(abx_mols) if keep_indices[i]]
    abx_names = abx[abx_name_col]
    
    df = check_for_frags_in_known_abx(df, fragment_index_column, frag_mols, abx_mols)
    df = check_full_cpd_similarity_to_closest_abx(df, full_cpd_index_column, cpd_mols, abx_mols)
    return(df, abx_mols, abx_names)

def check_for_frags_in_train_set(df, fragment_index_column, frag_mols, ts_mols):
    frags = [frag_mols[frag_index] for frag_index in list(df[fragment_index_column])]
    ts_matches = check_for_fuzzy_frags_in_mols(frags, ts_mols)
    df['matched train set molecules'] = ts_matches
    return(df)

def check_full_cpd_similarity_to_closest_train_set(df, full_cpd_index_column, cpd_mols, ts_mols):
    tan_scores = [for_mol_list_get_lowest_tanimoto_to_closest_mol([cpd_mols[i] for i in sublist], ts_mols) for sublist in list(df[full_cpd_index_column])]
    df['tanimoto_scores_of_full_mols_to_nearest_train_set'] = tan_scores
    return(df)

def check_training_set(train_set_path, train_set_smiles_col, train_set_name_col, df, fragment_index_column, frag_mols, full_cpd_index_column, cpd_mols):
    if train_set_path == '':
        print('No training set data have been provided for a comparison.')
        return(df)
    ts = pd.read_csv(train_set_path)
    ts_mols = [Chem.MolFromSmiles(smi) for smi in list(ts[train_set_smiles_col])]
    keep_indices = [m is not None for m in ts_mols]
    ts = ts[keep_indices]
    print('number of train set molecules: ', len(ts))

    ts_mols = [m for i,m in enumerate(ts_mols) if keep_indices[i]]
    ts_names = list(ts[train_set_name_col])
    
    df = check_for_frags_in_train_set(df, 'matched_fragments', frag_mols, ts_mols)
    df = check_full_cpd_similarity_to_closest_train_set(df, 'matched_molecules', cpd_mols, ts_mols)
    return(df, ts_mols, ts_names)

####### Statistical Testing for Analogues Helper Functions #######

# look for difference between mols w/ and w/o frag in the same neighborhood
def preprocess_cpds(full_cpd_df, compound_smi_col, compound_hit_col):

    all_smis = []
    all_fps = []
    all_scos = []

    # this takes a long time but is necessary for making sure the analogue code below goes reasonably quickly
    for smi, sco in zip(list(full_cpd_df[compound_smi_col]), list(full_cpd_df[compound_hit_col])):
        try:
            mo = Chem.MolFromSmiles(smi)
            fp = Chem.RDKFingerprint(mo)
            if fp is not None:
                all_smis.append(smi)
                all_fps.append(fp)
                all_scos.append(sco)
        except:
            continue
    return(all_smis, all_fps, all_scos)

# for every molecule, get similarity to closest N broad800k
def find_analogous_cpds_with_and_without_frags(mol_list, frag, all_smis, all_fps, all_scos, N=100, starting_thresh = 0.9, lower_limit_thresh = 0.7):
        
    all_scos_with_frag = []
    all_scos_without_frag = []  
    all_cpds_with_frag = []
    all_cpds_without_frag = []
    currently_used_smis = []

    while starting_thresh > lower_limit_thresh and (len(all_cpds_with_frag) < N or len(all_cpds_without_frag) < N):

        for mol in mol_list:
            fp = Chem.RDKFingerprint(mol)
            if fp is None:
                continue
            for i in range(0, len(all_fps)):
                f = all_fps[i]
                smi = all_smis[i]
                sco = all_scos[i]
                if f is None:
                    continue
                if smi in currently_used_smis:
                    continue

                sim = DataStructs.FingerprintSimilarity(f,fp)
                if sim > starting_thresh:
                    # now determine appropriate list to put mol in
                    curr_mol = Chem.MolFromSmiles(smi)
                    if curr_mol.HasSubstructMatch(frag):
                        if len(all_cpds_with_frag) < N:
                            all_scos_with_frag.append(sco)
                            all_cpds_with_frag.append(curr_mol)
                            currently_used_smis.append(smi)

                    else:
                        if len(all_cpds_without_frag) < (N*2):
                            all_scos_without_frag.append(sco)
                            all_cpds_without_frag.append(curr_mol)
                            currently_used_smis.append(smi)
                # have to break out of both while loops
                if len(all_cpds_with_frag) == N and len(all_cpds_without_frag) == N:
                    break
            if len(all_cpds_with_frag) == N and len(all_cpds_without_frag) == N:
                break        
        starting_thresh = starting_thresh - 0.05
    return(all_cpds_with_frag, all_scos_with_frag, all_cpds_without_frag, all_scos_without_frag)


def find_cpds_with_and_without_frags_for_whole_list(df, full_cpd_df, compound_smi_col, compound_hit_col, fragment_index_column, frag_mols, full_cpd_index_column, cpd_mols):
    all_smis, all_fps, all_scos = preprocess_cpds(full_cpd_df, compound_smi_col, compound_hit_col)
    list_of_cpds_w_frag = []
    list_of_scos_w_frag = []
    list_of_cpds_wo_frag = []
    list_of_scos_wo_frag = []
    for i,cpd_list in tqdm(zip(list(df[fragment_index_column]), list(df[full_cpd_index_column]))):
        frag = frag_mols[i]
        mol_list = [cpd_mols[i] for i in cpd_list]
        l1, l2, l3, l4 = find_analogous_cpds_with_and_without_frags(mol_list, frag, all_smis, all_fps, all_scos, N=100)
        list_of_cpds_w_frag.append(l1)
        list_of_scos_w_frag.append(l2)
        list_of_cpds_wo_frag.append(l3)
        list_of_scos_wo_frag.append(l4)

    list_of_scos_w_frag = [[float(x1) for x1 in sublist] for sublist in list_of_scos_w_frag]
    list_of_scos_wo_frag = [[float(x1) for x1 in sublist] for sublist in list_of_scos_wo_frag]
    df['random_analogue_cpds_w_frag'] = list_of_cpds_w_frag
    df['random_analogue_scos_w_frag'] = list_of_scos_w_frag
    df['random_analogue_cpds_without_frag'] = list_of_cpds_wo_frag
    df['random_analogue_scos_without_frag'] = list_of_scos_wo_frag
        
    df['average_difference_with_and_without_frag'] = [np.mean(x) - np.mean(y) for x,y in zip(list_of_scos_w_frag, list_of_scos_wo_frag)]
    df['ttest_scos_with_and_without_frag'] = [sp.ttest_ind(x,y) for x,y in zip(list_of_scos_w_frag, list_of_scos_wo_frag)]
    return(df)
    
def threshold_on_stat_sign_analogues_with_and_without_frags(df, absolute_diff_thresh = 0, pval_diff_thresh = 0):
    def threshold_on_column(df, col, diff_thresh):
        keep_indices = []
        for x in list(df[col]):
            if np.isnan(x[1]): # 1 element is the pval
                keep_indices.append(True)
            else:
                keep_indices.append(float(x[1]) < diff_thresh)
        df = df[keep_indices]
        return(df)

    if pval_diff_thresh > 0:
        df = threshold_on_column(df, 'ttest_scos_with_and_without_frag', pval_diff_thresh)
        print('number of fragments with <' + str(pval_diff_thresh) + ' (or n/a) stat significance between analogues w/ and w/o frag: ', len(df))
    if absolute_diff_thresh > 0:
        df = threshold_on_column(df, 'average_difference_with_and_without_frag', absolute_diff_thresh)
        print('number of fragments with >' + str(pval_diff_thresh) + ' (or n/a) absolute value difference between analogues w/ and w/o frag: ', len(df))
    return(df)

####### Visualization Helper Functions #######

# code adapted from https://www.macinchem.org/reviews/clustering/clustering.php
def clusterFps(fps,num_clusters):

    tan_array = [DataStructs.BulkTanimotoSimilarity(i, fps) for i in fps]
    tan_array = np.array(tan_array)
    clusterer= AgglomerativeClustering(n_clusters = num_clusters, compute_full_tree = True).fit(tan_array)
    final_clusters = {}
    for ix, m in enumerate(clusterer.labels_):
        if m in final_clusters:
            curr_list = final_clusters[m]
            curr_list.append(ix)
            final_clusters[m] = curr_list
        else:
            final_clusters[m] = [ix]
    
    return clusterer.labels_, final_clusters

def draw_mols(mols, legends, file_path, cut_down_size = False, black_and_white = False):
    # code with help from the OG greg landrum: https://gist.github.com/greglandrum/d5f12058682f6b336905450e278d3399
    if cut_down_size:
        mols = mols[0:10]
        legends = legends[0:10]
    molsPerRow = 5
    subImgSize= (500,500)
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow:
        nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0],fullSize[1], subImgSize[0], subImgSize[1])
    d2d.drawOptions().legendFontSize=100
    if black_and_white: 
        d2d.drawOptions().useBWAtomPalette()
    d2d.DrawMolecules(mols,legends=legends)
    d2d.FinishDrawing()
    open(file_path,'wb+').write(d2d.GetDrawingText())

def plot_final_fragments_with_all_info(df, output_folder, frags, cpd_mols, cpd_names, abx_mols, abx_names, ts_mols, ts_names, plot_neighborhoods = True):

    index = 0
    # just display one as an example
    for i, row in df.iterrows():
        print('-------------------------------------------------------------------')
        frag_match_index = row['matched_fragments']
        print('candidate: ', index)
        print('SMILES: ', row['fragment_SMILES'])
        print('fragment index: ' + str(frag_match_index))
        display(frags[frag_match_index])
        print('fragment score: ', np.round(row['fragment_scores'],3))
        print('average matched molecule score: ', np.round(row['average_molecule_score'],3))
        print('number of matched broad800k molecules: ', len(row['matched_molecules']))
        print('average matched hepg2 growth: ', np.round(row['average_hepg2_tox_score_of_matched_molecules'],3))
        print('average matched primary growth: ', np.round(row['average_primary_tox_score_of_matched_molecules'],3))
        print('length of fragment: ', row['length_of_fragment'])
        abx_index_list = row['matched_antibiotics']
        if len(abx_index_list) > 0:
            print('matching abx')
            img=Draw.MolsToGridImage([m for i,m in enumerate(abx_mols) if i in abx_index_list], molsPerRow=10,maxMols=100, legends = [str(abx_names[i]) for i in abx_index_list])
            display(img)
        else:
            print('no matching abx found')
            
        ts_index_list = row['matched train set molecules']
        if len(ts_index_list) > 0:
            print('matching train set compounds')
            img=Draw.MolsToGridImage([m for i,m in enumerate(ts_mols) if i in ts_index_list], molsPerRow=10,maxMols=100, legends = [str(ts_names[i]) for i in ts_index_list])
            display(img)
        else:
            print('no matching train set molecules found')
        
        print('matching broad800k molecules')
        full_mol_index_list = row['matched_molecules']
        if len(full_mol_index_list) > 0:
            curr_match_names = [cpd_names[i] for i in full_mol_index_list]
            curr_match_names = [x if type(x) is str else 'nan' for x in curr_match_names]
            curr_match_scores = [float(x) for x in list(row['full_molecule_scores'])]
            curr_tan_abxs = [float(x) for x in list(row['tanimoto_scores_of_full_mols_to_nearest_abx'])]
            curr_tan_tss = [float(x) for x in list(row['tanimoto_scores_of_full_mols_to_nearest_train_set'])]
            curr_match_scores, curr_tan_abxs, curr_tan_tss, curr_match_mol_index_list, curr_match_names = zip(*sorted(zip(curr_match_scores, curr_tan_abxs, curr_tan_tss, full_mol_index_list, curr_match_names), reverse = True))
            curr_match_mols = [m for i,m in enumerate(cpd_mols) if i in curr_match_mol_index_list]
            legends = [n + ', ' + str(np.round(sc, 3)) + '\n tan score to closest abx: ' + str(np.round(ta, 3)) + '\n tan score to closest TS: ' + str(np.round(tt, 3)) for n,sc,ta,tt in zip(curr_match_names, curr_match_scores, curr_tan_abxs, curr_tan_tss)]
            img=Draw.MolsToGridImage(curr_match_mols, molsPerRow=5,maxMols=500, legends = legends)
            display(img)
            draw_mols(curr_match_mols, legends, output_folder + str(index) + '_' + row['fragment_SMILES'] + '.png', cut_down_size = True)
        else:
            print('no matching broad800k molecules found')

        if plot_neighborhoods:
            print('random sampling of cpds with and without frag:')
            cpds_w_frag = list(row['random_analogue_cpds_w_frag'])
            scos_w_frag = list(row['random_analogue_scos_w_frag'])
            cpds_wo_frag = list(row['random_analogue_cpds_without_frag'])
            scos_wo_frag = list(row['random_analogue_scos_without_frag'])

            if len(cpds_w_frag) > 1 and len(cpds_wo_frag) > 1:
                print('avg difference of analogues with and without frag: ', np.round(row['average_difference_with_and_without_frag'],3))
                print('t test of analogues with and without frag: ', row['ttest_scos_with_and_without_frag'])
                print('random analogues with frag:')
                ana_w_frag_legends = [str(np.round(x,3)) for x in scos_w_frag]
                display(Draw.MolsToGridImage(cpds_w_frag, legends = ana_w_frag_legends))
                draw_mols(cpds_w_frag, ana_w_frag_legends, output_folder + str(index) + '_random_mols_with_frag.png', cut_down_size = True)

                print('random analogues without frag:')
                ana_wo_frag_legends = [str(np.round(x,3)) for x in scos_wo_frag]
                display(Draw.MolsToGridImage(cpds_wo_frag, legends = ana_wo_frag_legends))
                draw_mols(cpds_wo_frag, ana_wo_frag_legends, output_folder + str(index) + '_random_mols_without_frag.png', cut_down_size = True)

            else:
                print('not enough matching analogues found')

        index = index + 1

def add_legends_to_compounds(df, smiles_column = 'SMILES', name_column = 'Name'):
    df['row_num'] = list(range(len(df)))
    df = df.drop_duplicates(subset=smiles_column)
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(mol) for mol in smis]
    
    for row_num, smi in enumerate(smis):
        try:
            mol = Chem.MolFromSmiles(smi)
            row = df.iloc[row_num,:]
            actual_row_num = str(row.loc['row_num'])
            actualrowname = str(row.loc[name_column])
            legend = str(actual_row_num) + ', ' + actualrowname + '\n' + 'SMILES: ' + smi
        except Exception as e:
            print(e)
            actual_row_num = str(row.loc['row_num'])
            legend = 'row: ' + actual_row_num
        mols[row_num].SetProp('legend', legend)
    return(mols)
    
def add_legends_to_fragments(df, smiles_column = 'fragment_SMILES'):
    df['row_num'] = list(range(len(df)))
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(mol) for mol in smis]
    
    for row_num, smi in enumerate(smis):
        legend = 'Candidate: ' + str(row_num)
        mols[row_num].SetProp('legend', legend)
    return(mols)

def extract_legends_and_plot(df, mols, plot_suffix, path, num_clusters = 30, murcko_scaffold = False):
    if num_clusters == 0:
        num_clusters = 1
    if num_clusters > len(df):
        num_clusters = len(df)
    # code with help from the OG greg landrum: https://gist.github.com/greglandrum/d5f12058682f6b336905450e278d3399
    molsPerRow = 4
    subImgSize= (500,500)
    
    if murcko_scaffold:
        mols = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in mols]
    raw_cluster_labels, final_clusters=clusterFps(fps,num_clusters=num_clusters)
    
    #show clusters
    img_list = []
    name_index = 0
    
    for cluster_key in final_clusters:
        cluster_mols = final_clusters[cluster_key]
        cluster_mols = [mols[i] for i in cluster_mols]

        nRows = len(cluster_mols) // molsPerRow
        if len(cluster_mols) % molsPerRow:
            nRows += 1
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
        d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0],fullSize[1], subImgSize[0], subImgSize[1])
        d2d.drawOptions().legendFontSize=100
        #d2d.drawOptions().useBWAtomPalette()
        d2d.DrawMolecules(cluster_mols,legends=[mol.GetProp('legend') for mol in cluster_mols]) # assume mols have legend property
        d2d.FinishDrawing()

        new_name = path + str(name_index) + '_' + plot_suffix
        open(new_name,'wb+').write(d2d.GetDrawingText())
        img_list.append(new_name)
        name_index = name_index + 1
    df['cluster'] = [str(i) for i in raw_cluster_labels]
    return(df)

def cluster_mols_based_on_fragments(df, compound_smi_col, cpd_name_col, frags, result_path):
    mols = add_legends_to_compounds(df, smiles_column = compound_smi_col, name_column = cpd_name_col)

    matching_frag_indices = []
    # find the fragments they match to
    for i, mol in enumerate(mols):
        frag_indices = []
        for j, frag in enumerate(frags): # only works bc all smiles make valid mols
            if mol.HasSubstructMatch(frag):
                frag_indices.append(j) 
        matching_frag_indices.append(frag_indices)
    df['matching_frags'] = matching_frag_indices
        
    folder = result_path + 'FINAL_mols_clustered_by_fragment/'
    os.mkdir(folder)
    df = extract_legends_and_plot(df, mols, plot_suffix='cluster.png', path=folder, murcko_scaffold=True, num_clusters = int(len(df)/5))
    return(df)

#### Additional Filtering #####

def filter_for_existing_mols(df, df_name_col, looking_for_presence, test_path, test_name_col, test_name_needs_split):
    if test_path == '':
        return(df)
    if '.xlsx' in test_path:
        testdf = pd.read_excel(test_path)
    if '.csv' in test_path:
        testdf = pd.read_csv(test_path)
    if test_name_needs_split:
        test_names = [x.split('-')[0] + '-' + x.split('-')[1] for x in list(testdf[test_name_col])]
    else:
        test_names = list(testdf[test_name_col])
    if looking_for_presence:
        keep_indices = [n in test_names for n in list(df[df_name_col])]
    else: # looking for absence
        keep_indices = [n not in test_names for n in list(df[df_name_col])]
    df = df[keep_indices]
    return(df)
        
#### Condensed version for controls ####

def mini_algo(fragment_path, compound_path, result_path, fragment_smi_col = 'smiles', compound_smi_col = 'smiles', fragment_hit_col = 'hit', compound_hit_col = 'hit', fragment_score = 0.2, compound_score = 0.2, fragment_require_more_than_coh = True, fragment_remove_pains_brenk = 'both', compound_remove_pains_brenk = 'both', fragment_druglikeness_filter=[], compound_druglikeness_filter =[], fragment_remove_patterns=[], frags_cannot_disrupt_rings=False):
    ##### part 1: process frags and compounds #####
    print('\nProcessing fragments...')
    df, mols, _ = process_dataset(frag_or_cpd='frag', path=fragment_path, score=fragment_score, smi_col=fragment_smi_col, hit_col=fragment_hit_col, require_more_than_coh=fragment_require_more_than_coh, remove_pains_brenk=fragment_remove_pains_brenk, druglikeness_filter=fragment_druglikeness_filter, remove_patterns=fragment_remove_patterns)
    print('\nProcessing compounds...')
    cpd_df, cpd_mols, _ = process_dataset(frag_or_cpd='cpd', path=compound_path, score=compound_score, smi_col=compound_smi_col, hit_col=compound_hit_col, require_more_than_coh=False, remove_pains_brenk=compound_remove_pains_brenk, druglikeness_filter=compound_druglikeness_filter, remove_patterns=[])
    print('\nMatching fragments in compounds...')
        
    ##### part 2: get all matching frag / molecule pairs #####
    frag_match_indices, cpd_match_indices_lists = match_frags_and_mols(mols, cpd_mols)
    if frags_cannot_disrupt_rings:
        # for all matching fragments, keep only matches that do not disrupt rings
        frag_match_indices, cpd_match_indices_lists = check_for_complete_ring_fragments(mols, frag_match_indices, cpd_mols, cpd_match_indices_lists)
    rank_df = compile_results_into_df(df, cpd_df, mols, frag_match_indices, cpd_match_indices_lists, result_path, frag_hit_column=fragment_hit_col, cpd_hit_column=compound_hit_col)
    return(rank_df, cpd_df)