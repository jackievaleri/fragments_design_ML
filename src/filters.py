"""Utilities for filtering pipeline."""

import numpy as np
import pandas as pd
from adme_pred import ADME
from IPython.display import display

from rdkit import Chem
from rdkit.Chem import MCS, rdFMCS
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from downselect import for_mol_list_get_highest_tanimoto_to_closest_mol

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def threshold_on_score(df, thresh, column):
    """
    Filter DataFrame based on a threshold value on a specified column.

    :param df: DataFrame to filter.
    :param thresh: Threshold value.
    :param column: Name of the column to apply the threshold.
    :return: Filtered DataFrame.
    """
    keep_indices = [x != "Invalid SMILES" for x in list(df[column])]
    df = df[keep_indices]
    keep_indices = [float(s) > thresh for s in list(df[column])]
    df = df[keep_indices]
    print("length of df >" + str(thresh) + ": ", len(df))
    return df


def filter_require_more_than_coh(df, smiles_column):
    """
    Filter DataFrame to keep only rows with SMILES containing characters
        other than C, O, and H.

    :param df: DataFrame to filter.
    :param smiles_column: Name of the column containing SMILES.
    :return: Filtered DataFrame.
    """
    coh_characters = [
        "C",
        "c",
        "O",
        "o",
        "H",
        "(",
        ")",
        "[",
        "]",
        "=",
        "#",
        "+",
        "-",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    smis = list(df[smiles_column])

    def require_more_than_coh_one_mol(smi):
        """
        Check if a SMILES string contains characters other than C, O, and H.

        :param smi: SMILES string to check.
        :return: True if the SMILES contains characters other than C, O, and H,
            False otherwise.
        """
        for char in smi:
            if char not in coh_characters:
                return True  # good, more interesting
        return False  # bad, only COH

    keep_indices = [require_more_than_coh_one_mol(smi) for smi in smis]
    df = df[keep_indices]
    print("length of df with more than C,O,H characters: ", len(df))
    return df


def keep_valid_molecules(df, smiles_column):
    """
    Filter out invalid molecules from a DataFrame based on the SMILES column.

    :param df: DataFrame containing the molecules.
    :param smiles_column: Name of the column containing SMILES strings.
    :return: Tuple containing the filtered DataFrame
        and a list of valid RDKit molecule objects.
    """
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    keep_indices = [m is not None for m in mols]
    df = df[keep_indices]
    print("length of df with valid mols: ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    for m, smi in zip(mols, smis):
        m.SetProp("SMILES", smi)
    return (df, mols)


def check_pains_brenk(df, mols, method="both"):
    """
    Filter out molecules containing PAINS or Brenk alerts from a DataFrame.

    :param df: DataFrame containing the molecules.
    :param mols: List of RDKit molecules corresponding to the mols in the df.
    :param method: Method to filter, can be "both", "pains", or "brenk".
    :return: Tuple containing the filtered DataFrame
        and a list of corresponding RDKit molecule objects.
    """
    params = FilterCatalogParams()
    if method == "both" or method == "pains":
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    if method == "both" or method == "brenk":
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)

    def search_for_pains_or_brenk(mol):
        """
        Search for PAINS or Brenk alerts in an RDKit molecule.

        Get the first matching PAINS or Brenk.

        :param mol: RDKit molecule object to search for alerts.
        :return: True if the molecule does not contain any PAINS or Brenk,
            False otherwise.
        """
        entry = catalog.GetFirstMatch(mol)
        if entry is not None:
            return False
        else:
            return True

    keep_indices = [search_for_pains_or_brenk(m) for m in mols]
    df = df[keep_indices]
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    print("length of all preds with clean (no PAINS or Brenk) mols: ", len(df))
    return (df, mols)


def is_pattern(mol, pattern_mol):
    """
    Check if a molecule contains a specific substructure pattern.

    :param mol: RDKit molecule object.
    :param pattern_mol: RDKit molecule object representing the substructure.
    :return: True if the molecule contains the pattern, False otherwise.
    """
    if mol is None:
        return False
    num_atoms_frag = pattern_mol.GetNumAtoms()
    mcs = MCS.FindMCS(
        [mol, pattern_mol], atomCompare="elements", completeRingsOnly=True
    )
    if mcs.smarts is None:
        return False
    mcs_mol = Chem.MolFromSmarts(mcs.smarts)
    num_atoms_mcs = mcs_mol.GetNumAtoms()
    if num_atoms_frag == num_atoms_mcs:
        return True
    else:
        return False


def filter_for_pattern(df, mols, pattern_smi):
    """
    Filter molecules based on the absence of a specific substructure pattern.

    :param df: DataFrame containing molecular data.
    :param mols: List of RDKit molecule objects.
    :param pattern_smi: SMILES string representing the substructure to filter.
    :return: Tuple of filtered DataFrame and filtered list of RDKit molecules.
    """
    pattern_mol = Chem.MolFromSmiles(pattern_smi)
    if pattern_mol is None:
        print("Pattern smiles is invalid.")
        return (df, mols)
    keep_indices = [not is_pattern(mol, pattern_mol) for mol in mols]
    df = df[keep_indices]
    print("length of df with no " + pattern_smi + ": ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def filter_for_druglikeness(df, mols, smiles_column, criteria):
    """
    Filter molecules based on drug-likeness criteria.

    :param df: DataFrame containing molecular data.
    :param mols: List of RDKit molecule objects.
    :param smiles_column: Name of the column containing SMILES strings.
    :param criteria: Criteria for drug-likeness filtering.
        Supported criteria are "egan", "ghose", "lipinski", and "muegge".
    :return: Tuple of filtered DataFrame and filtered list of RDKit molecules.
    """
    smis = list(df[smiles_column])
    if criteria == "egan":
        druglikeness = [ADME(smi).druglikeness_egan() for smi in smis]
    elif criteria == "ghose":
        druglikeness = [ADME(smi).druglikeness_ghose() for smi in smis]
    elif criteria == "lipinski":
        druglikeness = [ADME(smi).druglikeness_lipinski() for smi in smis]
    elif criteria == "muegge":
        druglikeness = [ADME(smi).druglikeness_muegge() for smi in smis]
    else:
        print(
            "Criteria " +
            criteria +
            " not supported for drug-likeness filtering.")
        return (df, mols)
    df = df[druglikeness]
    print("length of df satisfying druglikeness " + criteria + ": ", len(df))
    mols = [m for i, m in enumerate(mols) if druglikeness[i]]
    return (df, mols)


def process_dataset(
    frag_or_cpd,
    path,
    score,
    smi_col,
    hit_col,
    require_more_than_coh,
    remove_pains_brenk,
    remove_patterns,
    druglikeness_filter,
):
    """
    Process a dataset for fragments or compounds, applying various filters.

    :param frag_or_cpd: Type of molecules in the dataset
        either "frag" for fragments or "cpd" for compounds.
    :param path: Path to the dataset file.
    :param score: Threshold score value for filtering.
    :param smi_col: Name of the column containing SMILES strings.
    :param hit_col: Name of the column containing hit scores.
    :param require_more_than_coh: Boolean indicating whether to filter
        molecules requiring more than C, O, H characters.
    :param remove_pains_brenk: Method for removing PAINS or Brenk alerts.
        Options are "pains", "brenk", or "both".
    :param remove_patterns: List of SMILES patterns to remove from the dataset.
    :param druglikeness_filter: List of criteria for drug-likeness filtering.
    :return: Tuple containing filtered DataFrame,
        filtered list of RDKit molecule objects, and a copy of the original df.
    """
    # process fragments first
    df = pd.read_csv(path)
    print("length of df: ", len(df))

    # keep copy of full df for cpds
    fulldf = df.copy()

    # threshold on score
    df = threshold_on_score(df, score, hit_col)

    # keep only frags with more than just C, O, H
    if require_more_than_coh and frag_or_cpd == "cpd":
        print(
            "Removing compounds with more than C, O, H is not supported "
            + "because it is redundant. "
            + "Try removing only these fragments."
        )
    if require_more_than_coh:
        df = filter_require_more_than_coh(df, smi_col)

    # keep only valid frags
    df, mols = keep_valid_molecules(df, smi_col)

    # keep only frags without pains or brenk
    if remove_pains_brenk != "none":
        df, mols = check_pains_brenk(df, mols, method=remove_pains_brenk)

    # remove common abx
    if frag_or_cpd == "cpd" and len(remove_patterns) > 0:
        print(
            "Removing patterns from compounds is not supported "
            + "because it is redundant. "
            + "Try removing patterns from fragments only."
        )
    elif frag_or_cpd == "frag":
        for patt_smi in remove_patterns:
            df, mols = filter_for_pattern(df, mols, patt_smi)

    # filter on druglikeness
    for criteria in druglikeness_filter:
        df, mols = filter_for_druglikeness(df, mols, smi_col, criteria)

    return (df, mols, fulldf)


def match_frags_and_mols(frags, cpd_mols):
    """
    Match fragments with compounds based on substructure search.

    Very slow because n frags x n mols

    :param frags: List of RDKit molecule objects representing fragments.
    :param cpd_mols: List of RDKit molecule objects representing compounds.
    :return: Tuple containing indices of matched fragments
        and a list of lists containing indices of compounds
        matched with each fragment.
    """
    matches = []
    frag_match_indices = []
    for frag_index, frag in enumerate(frags):
        matched_with_this_frag = []
        for full_mol_index, full_mol in enumerate(cpd_mols):
            if full_mol.HasSubstructMatch(
                    frag):  # contains entirely the fragment
                matched_with_this_frag.append(full_mol_index)
        if len(matched_with_this_frag) > 0:
            matches.append(matched_with_this_frag)
            frag_match_indices.append(frag_index)

    print("number of matched fragments: ", len(matches))
    return (frag_match_indices, matches)


def check_for_complete_ring_fragments(
    frags, frag_match_indices, cpd_mols, cpd_match_indices_lists
):
    """
    Check for fragments that form complete rings within compounds.

    :param frags: List of RDKit molecule objects representing fragments.
    :param frag_match_indices: List of indices of matched fragments.
    :param cpd_mols: List of RDKit molecule objects representing compounds.
    :param cpd_match_indices_lists: List of lists containing indices
        of compounds matched with each fragment.
    :return: Tuple containing updated indices of matched fragments
        and lists of indices of compounds matched
        with each fragment after filtering out fragments
        that form complete rings within compounds.
    """
    def check_frag_does_not_disrupt_mol(frag, mol):
        """
        Check if a fragment disrupts compound's structure via breaking rings.

        :param frag: RDKit molecule object representing the fragment.
        :param mol: RDKit molecule object representing the molecule.
        :return: True if the fragment does not disrupt the molecule
            by forming a complete ring, False otherwise.
        """
        mcs = rdFMCS.FindMCS([frag, mol], completeRingsOnly=True)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        num_atoms_mcs = mcs_mol.GetNumAtoms()
        num_atoms_frag = frag.GetNumAtoms()
        return num_atoms_frag == num_atoms_mcs

    new_frag_match_indices = []
    new_cpd_match_indices_lists = []
    for frag_match_index, full_mol_index_list in zip(
        frag_match_indices, cpd_match_indices_lists
    ):
        frag = frags[frag_match_index]
        mols = [m for i, m in enumerate(cpd_mols) if i in full_mol_index_list]
        keep_indices = [
            check_frag_does_not_disrupt_mol(
                frag, mol) for mol in mols]
        new_index_list = [
            i for (
                i,
                v) in zip(
                full_mol_index_list,
                keep_indices) if v]
        if len(new_index_list) > 0:
            new_frag_match_indices.append(frag_match_index)
            new_cpd_match_indices_lists.append(new_index_list)
    return (new_frag_match_indices, new_cpd_match_indices_lists)


def compile_results_into_df(
    df,
    cpd_df,
    mols,
    frag_match_indices,
    cpd_match_indices_lists,
    result_path,
    frag_hit_column,
    cpd_hit_column,
):
    """
    Compile the matching results into a DataFrame.

    :param df: DataFrame containing fragment data.
    :param cpd_df: DataFrame containing compound data.
    :param mols: List of RDKit molecule objects representing fragments.
    :param frag_match_indices: List of indices of matched fragments.
    :param cpd_match_indices_lists: List of lists where each inner list
        contains the indices of matched compounds for each fragment.
    :param result_path: Path to save the result DataFrame.
    :param frag_hit_column: Column name containing hit scores for fragments.
    :param cpd_hit_column: Column name containing hit scores for compounds.
    :return: DataFrame containing compiled matching results.
    """
    rank_df = pd.DataFrame()
    rank_df["matched_fragments"] = frag_match_indices
    rank_df["fragment_SMILES"] = [
        mols[i].GetProp("SMILES") for i in list(rank_df["matched_fragments"])
    ]
    rank_df["length_of_fragment"] = [mols[i].GetNumAtoms()
                                     for i in frag_match_indices]
    rank_df["matched_molecules"] = cpd_match_indices_lists
    rank_df["number_of_matched_molecules"] = [
        len(m) for m in cpd_match_indices_lists]
    rank_df["fragment_scores"] = [
        df.iloc[i, list(df.columns).index(frag_hit_column)]
        for i in frag_match_indices
    ]
    rank_df["full_molecule_scores"] = [
        [cpd_df.iloc[i, list(cpd_df.columns).index(cpd_hit_column)]
         for i in sublist]
        for sublist in cpd_match_indices_lists
    ]
    rank_df["average_molecule_score"] = [
        np.mean([float(x) for x in sublist])
        for sublist in list(rank_df["full_molecule_scores"])
    ]
    rank_df = rank_df.sort_values(
        "number_of_matched_molecules",
        ascending=False)
    rank_df.to_csv(result_path + "candidates_after_matching.csv", index=False)

    print("Previewing dataframe so far...")
    display(rank_df.head(5))
    return rank_df


def check_for_toxicity(df, fragment_column, frag_mols):
    """
    Check for toxicity of fragments against HEPG2 and primary cell lines.

    Make dictionary mapping SMILES to list of hep_means and primary_means.
    Use hardcoded file paths.

    :param df: DataFrame containing fragment data.
    :param fragment_column: Column containing fragment indices.
    :param frag_mols: List of RDKit molecule objects representing fragments.
    :return: DataFrame with toxicity information.
    """
    hepg2 = pd.read_excel(
        "../data/static_datasets/HEPG2_Primary_Toxicity_Data.xlsx",
        sheet_name="HepG2"
    )
    hepg2["hepg2_mean"] = hepg2["Mean_10uM"]
    primary = pd.read_excel(
        "../data/static_datasets/HEPG2_Primary_Toxicity_Data.xlsx",
        sheet_name="Primary"
    )
    primary["primary_mean"] = primary["Mean_10uM"]
    tox = hepg2.merge(primary, on="Compound_ID", how="left")
    tox = tox.drop_duplicates("Compound_ID")
    tox["both_means"] = [[i, j]
                         for i, j in zip(tox["hepg2_mean"],
                                         tox["primary_mean"])]
    tox_dict = dict(zip(list(tox["SMILES_x"]), list(tox["both_means"])))

    tox_smis = tox_dict.keys()
    tox_mols = [Chem.MolFromSmiles(smi) for smi in tox_smis]
    hepg2_tox_matches = []
    prim_tox_matches = []

    for frag_index in list(
        df[fragment_column]
    ):  # only do it for the short-listed frags
        frag = frag_mols[frag_index]
        hepg2_matched_with_this_frag = []
        prim_matched_with_this_frag = []
        for tox_smi, tox_mol in zip(tox_smis, tox_mols):
            if tox_mol is None:
                continue
            # this match can disrupt a ring - not picky about that
            if tox_mol.HasSubstructMatch(
                    frag):  # contains entirely the fragment
                hepg2_matched_with_this_frag.append(tox_dict[tox_smi][0])
                prim_matched_with_this_frag.append(tox_dict[tox_smi][1])
        hepg2_tox_matches.append(hepg2_matched_with_this_frag)
        prim_tox_matches.append(prim_matched_with_this_frag)

    df["hepg2_tox_matched_molecule_values"] = hepg2_tox_matches
    df["primary_tox_matched_molecule_values"] = prim_tox_matches
    df["average_hepg2_tox_score_of_matched_molecules"] = [
        np.mean(sublist)
        for sublist in list(df["hepg2_tox_matched_molecule_values"])
    ]
    df["average_primary_tox_score_of_matched_molecules"] = [
        np.mean(sublist)
        for sublist in list(df["primary_tox_matched_molecule_values"])
    ]
    return df


def filter_out_toxicity(
    df,
    tox_thresh,
    fragment_index_column,
    frag_mols,
    toxicity_threshold_require_presence,
):
    """
    Filter out fragments based on toxicity thresholds.

    :param df: DataFrame containing fragment data.
    :param tox_thresh: Threshold for toxicity.
    :param fragment_index_column: Column containing fragment indices.
    :param frag_mols: List of RDKit molecule objects representing fragments.
    :param toxicity_threshold_require_presence: Whether to require toxicity
        data. In some cases, can be okay without it.
    :return: Filtered DataFrame.
    """
    df = check_for_toxicity(df, fragment_index_column, frag_mols)
    keep_indices = []
    for i, row in df.iterrows():
        hep = row["average_hepg2_tox_score_of_matched_molecules"]
        prim = row["average_primary_tox_score_of_matched_molecules"]
        if not toxicity_threshold_require_presence:
            if np.isnan(hep) or np.isnan(prim):
                keep_indices.append(i)
            elif hep > tox_thresh and prim > tox_thresh:
                keep_indices.append(i)
        else:
            if hep > tox_thresh and prim > tox_thresh:
                keep_indices.append(i)
    df = df.iloc[keep_indices, :]
    print(
        "number of fragments passing both toxicity filters under "
        + str(tox_thresh)
        + ": "
        + str(len(df))
    )
    return df


def check_for_fuzzy_frags_in_mols(frags, mols):
    """
    Check for fuzzy matches of fragments in molecules.

    Do for any set of frags and any set of mols.

    :param frags: List of RDKit molecule objects representing fragments.
    :param mols: List of RDKit molecule objects representing molecules.
    :return: List of lists containing idxs of molecules that match fragments.
    """
    matches = []
    for frag in frags:
        matched_with_this_frag = []
        for index, mol in enumerate(mols):
            if mol.HasSubstructMatch(frag):  # contains entirely the fragment
                matched_with_this_frag.append(index)
        matches.append(matched_with_this_frag)
    return matches


def check_for_frags_in_known_abx(
        df, fragment_index_column, frag_mols, abx_mols):
    """
    Check for fragments in known antibiotics.

    :param df: DataFrame containing fragment information.
    :param fragment_index_column: Name of the column containing fragment idxs.
    :param frag_mols: List of RDKit molecule objects representing fragments.
    :param abx_mols: List of RDKit molecule objects representing known abx.
    :return: DataFrame with information about matched antibiotics.
    """
    frags = [frag_mols[frag_index]
             for frag_index in list(df[fragment_index_column])]
    abx_matches = check_for_fuzzy_frags_in_mols(frags, abx_mols)
    df["matched_antibiotics"] = abx_matches
    return df


def check_full_cpd_similarity_to_closest_abx(
    df, full_cpd_index_column, cpd_mols, abx_mols
):
    """
    Calculate the Tanimoto similarity scores of molecules to the nearest abx.

    :param df: DataFrame containing information about the compounds.
    :param full_cpd_index_column: Column containing the indices
        of the full compounds in cpd_mols.
    :param cpd_mols: List of RDKit molecule objects representing the full cpds.
    :param abx_mols: List of RDKit molecule objects representing the abx.
    :return: DataFrame with an additional column for Tanimoto similarities.
    """
    tan_scores = [
        for_mol_list_get_highest_tanimoto_to_closest_mol(
            [cpd_mols[i] for i in sublist], abx_mols
        )
        for sublist in list(df[full_cpd_index_column])
    ]
    df["tanimoto_scores_of_full_mols_to_nearest_abx"] = tan_scores
    return df


def process_molset(
    path, smi_col,
    hit_col="",
    just_actives=False,
    hit_thresh=0,
    greater_than=False
):
    """
    Process a set of molecules from a CSV file.

    :param path: Path to the CSV file containing molecule data.
    :param smi_col: Name of the column containing SMILES strings.
    :param hit_col: Name of the column containing activity/hit data.
    :param just_actives: Boolean indicating whether to consider only actives.
    :param hit_thresh: Threshold value for activity/hit data.
    :param greater_than: Boolean indicating whether to consider values
        greater than or less than the threshold.
    :return: DataFrame containing processed molecule data and a list of mols.
    """
    if path == "":
        print("No data have been provided for a comparison.")
    df = pd.read_csv(path)
    if just_actives:
        if greater_than:
            df = df[df[hit_col] > hit_thresh]
        else:
            df = df[df[hit_col] < hit_thresh]
        df = df.reset_index(drop=True)
    mols = [Chem.MolFromSmiles(smi) for smi in list(df[smi_col])]
    keep_indices = [m is not None for m in mols]
    df = df[keep_indices]
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def check_abx(
    abx_path,
    abx_smiles_col,
    abx_name_col,
    df,
    fragment_index_column,
    frag_mols,
    full_cpd_index_column,
    cpd_mols,
):
    """
    Check for antibiotic fragments and similarity to closest antibiotics.

    :param abx_path: Path to the CSV file containing antibiotic data.
    :param abx_smiles_col: Name of the column containing SMILES in the abx df.
    :param abx_name_col: Name of the column containing names of antibiotics.
    :param df: DataFrame containing fragment data.
    :param fragment_index_column: Name of the column containing fragment idxs.
    :param frag_mols: List of RDKit molecule objects representing fragments.
    :param full_cpd_index_column: Name of the column containing compound idxs.
    :param cpd_mols: List of RDKit molecule objects representing compounds.
    :return: DataFrame with antibiotic matching information,
        list of RDKit molecule objects for antibiotics,
        and list of antibiotic names.
    """
    abx, abx_mols = process_molset(abx_path, abx_smiles_col)
    print("number of abx: ", len(abx))
    abx_names = abx[abx_name_col]
    df = check_for_frags_in_known_abx(
        df, fragment_index_column, frag_mols, abx_mols)
    df = check_full_cpd_similarity_to_closest_abx(
        df, full_cpd_index_column, cpd_mols, abx_mols
    )
    return (df, abx_mols, abx_names)


def check_for_frags_in_train_set(
        df, fragment_index_column, frag_mols, ts_mols):
    """
    Check for fragments in the training set molecules.

    :param df: DataFrame containing fragment data.
    :param fragment_index_column: Name of the column containing fragment idxs.
    :param frag_mols: List of RDKit molecule objects representing fragments.
    :param ts_mols: List of RDKit molecules representing cpds in the train set.
    :return: DataFrame with info about matching fragments in the train set.
    """
    frags = [frag_mols[frag_index]
             for frag_index in list(df[fragment_index_column])]
    ts_matches = check_for_fuzzy_frags_in_mols(frags, ts_mols)
    df["matched train set molecules"] = ts_matches
    return df


def check_full_cpd_similarity_to_closest_train_set(
    df, full_cpd_index_column, cpd_mols, ts_mols
):
    """
    Check the compound similarity to the closest molecule in the train set.

    :param df: DataFrame containing compound data.
    :param full_cpd_index_column: Name of the column containing compound idxs.
    :param cpd_mols: List of RDKit molecule objects representing compounds.
    :param ts_mols: List of RDKit molecules representing cpds in the train set.
    :return: DataFrame with additional column for Tanimoto scores
        of full compounds to nearest training set molecule.
    """
    tan_scores = [
        for_mol_list_get_highest_tanimoto_to_closest_mol(
            [cpd_mols[i] for i in sublist], ts_mols
        )
        for sublist in list(df[full_cpd_index_column])
    ]
    df["tanimoto_scores_of_full_mols_to_nearest_train_set"] = tan_scores
    return df


def check_training_set(
    train_set_path,
    train_set_smiles_col,
    train_set_name_col,
    df,
    fragment_index_column,
    frag_mols,
    full_cpd_index_column,
    cpd_mols,
    just_actives=False,
    hit_col="",
    hit_thresh=0,
    greater_than=False,
):
    """
    Check the compounds against a training set.

    :param train_set_path: Path to the training set file.
    :param train_set_smiles_col: Name of the column containing SMILES
        in the training set.
    :param train_set_name_col: Name of the column containing names
        in the training set.
    :param df: DataFrame containing compound data.
    :param fragment_index_column: Name of the column containing fragment idxs.
    :param frag_mols: List of RDKit molecule objects representing fragments.
    :param full_cpd_index_column: Name of the column containing compound idxs.
    :param cpd_mols: List of RDKit molecule objects representing compounds.
    :param just_actives: If True, consider only active compounds
        in the training set.
    :param hit_col: Name of the column containing hit data.
    :param hit_thresh: Threshold for hit data.
    :param greater_than: If True, consider compounds with hit values
        greater than the threshold.
    :return: DataFrame with additional columns for matched fragments
        in the training set and Tanimoto scores of full compounds
        to nearest training set molecule,
        list of RDKit molecule objects representing molecules in the train set,
        and list of names of molecules in the training set.
    """
    ts, ts_mols = process_molset(
        train_set_path,
        train_set_smiles_col,
        hit_col,
        just_actives,
        hit_thresh,
        greater_than,
    )
    print("number of train set molecules: ", len(ts))
    ts_names = list(ts[train_set_name_col])
    df = check_for_frags_in_train_set(
        df, "matched_fragments", frag_mols, ts_mols)
    df = check_full_cpd_similarity_to_closest_train_set(
        df, "matched_molecules", cpd_mols, ts_mols
    )
    return (df, ts_mols, ts_names)
