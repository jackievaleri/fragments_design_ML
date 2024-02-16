"""Utilities for filtering pipeline."""

import numpy as np
import os
import pandas as pd
import scipy.stats as sp
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit import DataStructs

from downselect import for_mol_list_get_highest_tanimoto_to_closest_mol
from filters import (
    process_dataset,
    match_frags_and_mols,
    compile_results_into_df,
    filter_out_toxicity,
    check_for_complete_ring_fragments,
    check_abx,
    check_training_set
)
from vis import (
    add_legends_to_fragments,
    extract_legends_and_plot,
    draw_mols,
    plot_final_fragments_with_all_info,
    cluster_mols_based_on_fragments,
    add_legends_to_compounds
)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def preprocess_cpds(full_cpd_df, compound_smi_col, compound_hit_col):
    """
    Preprocess compounds from a DataFrame.

    This is a helper function for statistical testing for analogues.
    Look for difference between mols w/ and w/o frag in the same neighborhood
    This takes a long time but is necessary for making sure the analogue
    code in find_analogous_cpds_with_and_without_frags goes reasonably quickly.

    :param full_cpd_df: DataFrame containing full compounds data.
    :param compound_smi_col: Name of the column containing SMILES in the df.
    :param compound_hit_col: Name of the column containing hit data in the df.
    :return: Tuple containing lists of SMILES, RDKit fingerprints,
        and hit scores of compounds.
    """
    all_smis = []
    all_fps = []
    all_scos = []
    for smi, sco in zip(
        list(
            full_cpd_df[compound_smi_col]), list(
            full_cpd_df[compound_hit_col])
    ):
        try:
            mo = Chem.MolFromSmiles(smi)
            fp = Chem.RDKFingerprint(mo)
            if fp is not None:
                all_smis.append(smi)
                all_fps.append(fp)
                all_scos.append(sco)
        except BaseException:
            continue
    return (all_smis, all_fps, all_scos)


def find_analogous_cpds_with_and_without_frags(
    mol_list,
    frag,
    all_smis,
    all_fps,
    all_scos,
    N=100,
    starting_thresh=0.9,
    lower_limit_thresh=0.7,
):
    """
    Find analogous compounds with and without a given fragment.

    :param mol_list: List of query molecules.
    :param frag: RDKit molecule object representing the fragment.
    :param all_smis: List of SMILES strings of all compounds.
    :param all_fps: List of RDKit fingerprints of all compounds.
    :param all_scos: List of hit scores of all compounds.
    :param N: Number of compounds to find for each category
        (with and without fragment).
    :param starting_thresh: Starting Tanimoto similarity threshold.
    :param lower_limit_thresh: Lower limit threshold for Tanimoto similarity.
    :return: Tuple containing lists of compounds and hit scores
        with and without the fragment.
    """
    all_scos_with_frag = []
    all_scos_without_frag = []
    all_cpds_with_frag = []
    all_cpds_without_frag = []
    currently_used_smis = []

    while starting_thresh > lower_limit_thresh and (
        len(all_cpds_with_frag) < N or len(all_cpds_without_frag) < N
    ):
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

                sim = DataStructs.FingerprintSimilarity(f, fp)
                if sim > starting_thresh:
                    # now determine appropriate list to put mol in
                    curr_mol = Chem.MolFromSmiles(smi)
                    if curr_mol.HasSubstructMatch(frag):
                        if len(all_cpds_with_frag) < N:
                            all_scos_with_frag.append(sco)
                            all_cpds_with_frag.append(curr_mol)
                            currently_used_smis.append(smi)

                    else:
                        if len(all_cpds_without_frag) < (N * 2):
                            all_scos_without_frag.append(sco)
                            all_cpds_without_frag.append(curr_mol)
                            currently_used_smis.append(smi)
                # have to break out of both while loops
                if len(all_cpds_with_frag) == N and len(
                        all_cpds_without_frag) == N:
                    break
            if len(all_cpds_with_frag) == N and len(
                    all_cpds_without_frag) == N:
                break
        starting_thresh = starting_thresh - 0.05
    return (
        all_cpds_with_frag,
        all_scos_with_frag,
        all_cpds_without_frag,
        all_scos_without_frag,
    )


def find_cpds_with_and_without_frags_for_whole_list(
    df,
    full_cpd_df,
    compound_smi_col,
    compound_hit_col,
    fragment_index_column,
    frag_mols,
    full_cpd_index_column,
    cpd_mols,
):
    """
    Find compounds with and without fragments for the entire list.

    :param df: DataFrame containing the list.
    :param full_cpd_df: DataFrame containing full compounds.
    :param compound_smi_col: Column name for SMILES in full compound df.
    :param compound_hit_col: Column name for hit score in full compound df.
    :param fragment_index_column: Column name for fragment indices in list df.
    :param frag_mols: List of RDKit molecule objects representing fragments.
    :param full_cpd_index_column: Column name for full compound indices.
    :param cpd_mols: List of RDKit molecule objects representing full cpds.
    :return: DataFrame with additional columns for compounds
        with and without fragments.
    """
    all_smis, all_fps, all_scos = preprocess_cpds(
        full_cpd_df, compound_smi_col, compound_hit_col
    )
    list_of_cpds_w_frag = []
    list_of_scos_w_frag = []
    list_of_cpds_wo_frag = []
    list_of_scos_wo_frag = []
    for i, cpd_list in tqdm(
        zip(list(df[fragment_index_column]), list(df[full_cpd_index_column]))
    ):
        frag = frag_mols[i]
        mol_list = [cpd_mols[i] for i in cpd_list]
        l1, l2, l3, l4 = find_analogous_cpds_with_and_without_frags(
            mol_list, frag, all_smis, all_fps, all_scos, N=100
        )
        list_of_cpds_w_frag.append(l1)
        list_of_scos_w_frag.append(l2)
        list_of_cpds_wo_frag.append(l3)
        list_of_scos_wo_frag.append(l4)

    list_of_scos_w_frag = [
        [float(x1) for x1 in sublist] for sublist in list_of_scos_w_frag
    ]
    list_of_scos_wo_frag = [
        [float(x1) for x1 in sublist] for sublist in list_of_scos_wo_frag
    ]
    df["random_analogue_cpds_w_frag"] = list_of_cpds_w_frag
    df["random_analogue_scos_w_frag"] = list_of_scos_w_frag
    df["random_analogue_cpds_without_frag"] = list_of_cpds_wo_frag
    df["random_analogue_scos_without_frag"] = list_of_scos_wo_frag
    df["average_difference_with_and_without_frag"] = [
        np.mean(x) - np.mean(y)
        for x, y in zip(list_of_scos_w_frag, list_of_scos_wo_frag)
    ]
    df["ttest_scos_with_and_without_frag"] = [
        sp.ttest_ind(x, y) for x, y in zip(list_of_scos_w_frag,
                                           list_of_scos_wo_frag)
    ]
    return df


def threshold_on_stat_sign_analogues_with_and_without_frags(
    df,
    absolute_diff_thresh=0,
    pval_diff_thresh=0
):
    """
    Apply thresholds on statistical significance of analogues
        with and without fragments.

    :param df: DataFrame containing the results.
    :param absolute_diff_thresh: Absolute difference threshold.
    :param pval_diff_thresh: P-value difference threshold.
    :return: DataFrame filtered based on thresholds.
    """
    def threshold_on_pval_column(df, col, diff_thresh):
        """
        Apply threshold on p-value column in the DataFrame.

        :param df: DataFrame containing the data.
        :param col: Name of the column containing the p-values.
        :param diff_thresh: Threshold value for the p-values.
        :return: DataFrame filtered based on the threshold.
        """
        keep_indices = []
        for x in list(df[col]):
            if np.isnan(x[1]):  # 1 element is the pval
                keep_indices.append(True)
            else:
                keep_indices.append(float(x[1]) < diff_thresh)
        df = df[keep_indices]
        return df

    def threshold_on_absval_column(df, col, diff_thresh):
        """
        Apply threshold on absolute value column in the DataFrame.

        :param df: DataFrame containing the data.
        :param col: Name of the column containing the absolute values.
        :param diff_thresh: Threshold value for the absolute values.
        :return: DataFrame filtered based on the threshold.
        """
        keep_indices = []
        for x in list(df[col]):
            if np.isnan(x):
                keep_indices.append(True)
            else:
                keep_indices.append(float(x) > diff_thresh)
        df = df[keep_indices]
        return df

    if pval_diff_thresh > 0:
        df = threshold_on_pval_column(
            df, "ttest_scos_with_and_without_frag", pval_diff_thresh
        )
        print(
            "number of fragments with <"
            + str(pval_diff_thresh)
            + " (or n/a) stat significance "
            + "between analogues w/ and w/o frag: ",
            len(df),
        )
    if absolute_diff_thresh > 0:
        df = threshold_on_absval_column(
            df,
            "average_difference_with_and_without_frag",
            absolute_diff_thresh
        )
        print(
            "number of fragments with >"
            + str(absolute_diff_thresh)
            + " (or n/a) absolute value difference "
            + "between analogues w/ and w/o frag: ",
            len(df),
        )
    return df


def filter_for_existing_mols(
    df,
    df_name_col,
    looking_for_presence,
    test_path,
    test_name_col,
    test_name_needs_split,
):
    if test_path == "":
        return df
    if ".xlsx" in test_path:
        testdf = pd.read_excel(test_path)
    if ".csv" in test_path:
        testdf = pd.read_csv(test_path)
    if test_name_needs_split:
        test_names = [
            x.split("-")[0] + "-" + x.split("-")[1]
            for x in list(testdf[test_name_col])
        ]
    else:
        test_names = list(testdf[test_name_col])
    if looking_for_presence:
        keep_indices = [n in test_names for n in list(df[df_name_col])]
    else:  # looking for absence
        keep_indices = [n not in test_names for n in list(df[df_name_col])]
    df = df[keep_indices]
    return df


def run_pipeline(
    fragment_path,
    compound_path,
    result_path,
    fragment_smi_col="smiles",
    compound_smi_col="smiles",
    fragment_hit_col="hit",
    compound_hit_col="hit",
    fragment_score=0.2,
    compound_score=0.2,
    fragment_require_more_than_coh=True,
    fragment_remove_pains_brenk="both",
    compound_remove_pains_brenk="both",
    fragment_druglikeness_filter=[],
    compound_druglikeness_filter=[],
    fragment_remove_patterns=[],
    frags_cannot_disrupt_rings=True,
    fragment_length_threshold=0,
    toxicity_threshold_if_present=0,
    toxicity_threshold_require_presence=False,
    abx_path="",
    abx_smiles_col="smiles",
    abx_name_col="Name",
    train_set_path="",
    train_set_smiles_col="smiles",
    train_set_name_col="Name",
    train_set_just_actives=False,
    train_set_hit_col="",
    train_set_thresh=0,
    train_set_greater_than=False,
    analogues_pval_diff_thresh=0,
    analogues_absolute_diff_thresh=0,
    cpd_name_col="Name",
    display_inline_candidates=False,
    purch_path="",
    purch_name_col="Name",
    purch_name_needs_split=False,
    tested_before_path="",
    tested_before_name_col="Name",
    tested_before_name_needs_split=False,
    cpd_sim_to_abx=0,
    cpd_sim_to_train_set=0,
):
    """
    Execute the pipeline to process fragments and compounds,
    match them, perform statistical analysis, visualize results,
    and save relevant information.

    :param fragment_path: Path to the file containing fragment data.
    :param compound_path: Path to the file containing compound data.
    :param result_path: Path to store the results.
    :param fragment_smi_col: Column name for fragment SMILES.
    :param compound_smi_col: Column name for compound SMILES.
    :param fragment_hit_col: Column name indicating fragment hit.
    :param compound_hit_col: Column name indicating compound hit.
    :param fragment_score: Score threshold for fragments.
    :param compound_score: Score threshold for compounds.
    :param fragment_require_more_than_coh: Flag indicating if fragments
        require more than cohesion.
    :param fragment_remove_pains_brenk: Method to remove PAINS/BRENK fragments.
    :param compound_remove_pains_brenk: Method to remove PAINS/BRENK compounds.
    :param fragment_druglikeness_filter: Drug-likeness filter for fragments.
    :param compound_druglikeness_filter: Drug-likeness filter for compounds.
    :param fragment_remove_patterns: Patterns to remove from fragments.
    :param frags_cannot_disrupt_rings: Flag indicating if fragments
        cannot disrupt rings.
    :param fragment_length_threshold: Threshold for fragment length.
    :param toxicity_threshold_if_present: Threshold for toxicity if present.
    :param toxicity_threshold_require_presence: Flag indicating if toxicity
        threshold requires presence.
    :param abx_path: Path to the file containing antibiotics data.
    :param abx_smiles_col: Column name for antibiotic SMILES.
    :param abx_name_col: Column name for antibiotic names.
    :param train_set_path: Path to the file containing training set data.
    :param train_set_smiles_col: Column name for training set SMILES.
    :param train_set_name_col: Column name for training set names.
    :param train_set_just_actives: Flag indicating if only actives
        should be considered in the training set.
    :param train_set_hit_col: Column name indicating hits in the training set.
    :param train_set_thresh: Threshold for the training set.
    :param train_set_greater_than: Flag indicating if training set
        threshold is greater than.
    :param analogues_pval_diff_thresh: P-value difference threshold
        for statistical significance on analogues test.
    :param analogues_absolute_diff_thresh: Absolute difference threshold
        for statistical significance on analogues test.
    :param cpd_name_col: Column name for compound names.
    :param display_inline_candidates: Flag indicating if candidates
        should be displayed inline.
    :param purch_path: Path to the file containing purchasable data.
    :param purch_name_col: Column name for purchasable names.
    :param purch_name_needs_split: Flag indicating if purchasable names
        need splitting.
    :param tested_before_path: Path to the file containing
        previously tested data.
    :param tested_before_name_col: Column name for previously tested names.
    :param tested_before_name_needs_split: Flag indicating if
        previously tested names need splitting.
    :param cpd_sim_to_abx: Similarity threshold to antibiotics for filtering.
    :param cpd_sim_to_train_set: Similarity threshold to the training set
        for filtering.
    :return: DataFrame containing the final matching molecules.
    """
    # part 1: process frags and compounds
    print("\nProcessing fragments...")
    df, mols, _ = process_dataset(
        frag_or_cpd="frag",
        path=fragment_path,
        score=fragment_score,
        smi_col=fragment_smi_col,
        hit_col=fragment_hit_col,
        require_more_than_coh=fragment_require_more_than_coh,
        remove_pains_brenk=fragment_remove_pains_brenk,
        remove_patterns=fragment_remove_patterns,
        druglikeness_filter=fragment_druglikeness_filter,
    )
    print("\nProcessing compounds...")
    cpd_df, cpd_mols, full_cpd_df = process_dataset(
        frag_or_cpd="cpd",
        path=compound_path,
        score=compound_score,
        smi_col=compound_smi_col,
        hit_col=compound_hit_col,
        require_more_than_coh=False,
        remove_pains_brenk=compound_remove_pains_brenk,
        remove_patterns=[],
        druglikeness_filter=compound_druglikeness_filter,
    )
    print("\nMatching fragments in compounds...")

    # part 2: get all matching frag / molecule pairs
    frag_match_indices, cpd_match_indices_lists = match_frags_and_mols(
        mols, cpd_mols)
    if frags_cannot_disrupt_rings:
        # for all matching fragments, keep only matches that do not disrupt
        # rings
        frag_match_indices, cpd_match_indices_lists = check_for_complete_ring_fragments(  # noqa
            mols, frag_match_indices, cpd_mols, cpd_match_indices_lists
        )
    rank_df = compile_results_into_df(
        df,
        cpd_df,
        mols,
        frag_match_indices,
        cpd_match_indices_lists,
        result_path,
        fragment_hit_col,
        compound_hit_col,
    )

    # part 3: add additional data and filters
    # check for frags associated with toxicity in the in-house tox database
    rank_df = filter_out_toxicity(
        rank_df,
        toxicity_threshold_if_present,
        "matched_fragments",
        mols,
        toxicity_threshold_require_presence,
    )
    # check for frags within abx or cpds close to known abx
    # does not remove any cpds
    if abx_path != "":
        rank_df, abx_mols, abx_names = check_abx(
            abx_path,
            abx_smiles_col,
            abx_name_col,
            rank_df,
            "matched_fragments",
            mols,
            "matched_molecules",
            cpd_mols,
        )
    else:
        abx_mols = []
        abx_names = []
    # check for frags within train set or molecules close to train set
    # again does not remove cpds
    if train_set_path != "":
        rank_df, ts_mols, ts_names = check_training_set(
            train_set_path,
            train_set_smiles_col,
            train_set_name_col,
            rank_df,
            "matched_fragments",
            mols,
            "matched_molecules",
            cpd_mols,
            just_actives=train_set_just_actives,
            hit_col=train_set_hit_col,
            hit_thresh=train_set_thresh,
            greater_than=train_set_greater_than,
        )
    else:
        ts_mols = []
        ts_names = []
    # check for fragments at least bigger than fragment_length_threshold
    rank_df = rank_df[rank_df["length_of_fragment"]
                      > fragment_length_threshold]

    # part 4: statistical significance on analogues test
    if analogues_pval_diff_thresh > 0 or analogues_absolute_diff_thresh > 0:
        print("Checking analogues of compounds with and without fragments...")
        rank_df = find_cpds_with_and_without_frags_for_whole_list(
            rank_df,
            full_cpd_df,
            compound_smi_col,
            compound_hit_col,
            "matched_fragments",
            mols,
            "matched_molecules",
            cpd_mols,
        )
        rank_df = threshold_on_stat_sign_analogues_with_and_without_frags(
            rank_df,
            pval_diff_thresh=analogues_pval_diff_thresh,
            absolute_diff_thresh=analogues_absolute_diff_thresh,
        )

    # part 5: visualization
    rank_df = rank_df.sort_values(
        "number_of_matched_molecules",
        ascending=False)
    rank_df.to_csv(
        result_path + "candidates_after_matching_and_filtering.csv",
        index=False
    )

    # group fragments and see clusters
    frag_folder = result_path + "fragment_clusters/"
    os.mkdir(frag_folder)
    fragment_mols_for_plotting = add_legends_to_fragments(
        rank_df, smiles_column="fragment_SMILES"
    )
    rank_df = extract_legends_and_plot(
        rank_df,
        fragment_mols_for_plotting,
        plot_suffix="cluster.png",
        path=frag_folder,
        murcko_scaffold=False,
        num_clusters=int(len(rank_df) / 5),
    )
    rank_df.to_csv(frag_folder + "finalmols.csv", index=False)

    # draw all fragments
    draw_mols(
        [mols[i] for i in list(rank_df["matched_fragments"])],
        [str(i) for i in range(0, len(rank_df))],
        result_path + "ALL_FRAGMENT_MOLS.png",
        cut_down_size=False,
    )

    # look at molecules corresponding to fragments
    cpd_names = list(cpd_df[cpd_name_col])
    candidate_folder = result_path + "candidate_info/"
    os.mkdir(candidate_folder)
    plot_final_fragments_with_all_info(
        rank_df,
        candidate_folder,
        mols,
        cpd_mols,
        cpd_names,
        abx_mols,
        abx_names,
        ts_mols,
        ts_names,
        display_inline_candidates,
    )

    # part 6: save relevant information
    # get the final matching molecules for saving
    all_matching_mol_indices = [
        x for xlist in list(rank_df["matched_molecules"]) for x in xlist
    ]
    all_matching_mol_indices = list(
        set(all_matching_mol_indices))  # deduplicate
    print("final number of molecules to test: ", len(all_matching_mol_indices))

    # save the names
    all_matching_mols = [cpd_names[i]
                         for i in list(set(all_matching_mol_indices))]
    cpd_smiles = list(cpd_df[compound_smi_col])
    all_matching_smis = [cpd_smiles[i]
                         for i in list(set(all_matching_mol_indices))]

    # and save the final molecules to df
    all_matching_mols_df = pd.DataFrame()
    all_matching_mols_df[cpd_name_col] = all_matching_mols
    all_matching_mols_df[compound_smi_col] = all_matching_smis

    # add metadata to mols
    cpd_df_meta = cpd_df[[cpd_name_col, compound_hit_col]]
    all_matching_mols_df = all_matching_mols_df.merge(
        cpd_df_meta, on=cpd_name_col)
    all_matching_mols_df.to_csv(
        result_path
        + "candidate_compounds_after_matching_and_filtering_with_metadata.csv"
    )

    # part 7: additional filtering
    # keep only molecules IN the PURCHASABLE Broad800K library
    if purch_path != "":
        all_matching_mols_df = filter_for_existing_mols(
            all_matching_mols_df,
            cpd_name_col,
            looking_for_presence=True,
            test_path=purch_path,
            test_name_col=purch_name_col,
            test_name_needs_split=purch_name_needs_split,
        )
        print(
            "length of df with purchasable mols: ",
            len(all_matching_mols_df))
    if tested_before_path != "":
        # make only molecules NOT IN previously-tested dataframes (make sure
        # none of the molecules are exact matches to those tested before)
        all_matching_mols_df = filter_for_existing_mols(
            all_matching_mols_df,
            cpd_name_col,
            looking_for_presence=False,
            test_path=tested_before_path,
            test_name_col=tested_before_name_col,
            test_name_needs_split=tested_before_name_needs_split,
        )
        print(
            "length of df that has not been tested before: ", len(
                all_matching_mols_df)
        )

    # now do tanimoto filtering on antibiotics and training set
    current_all_matching_mols = [
        Chem.MolFromSmiles(smi)
        for smi in list(all_matching_mols_df[compound_smi_col])
    ]
    if len(abx_mols) > 0:
        all_matching_mols_df[
            "tan to nearest abx"
        ] = for_mol_list_get_highest_tanimoto_to_closest_mol(
            current_all_matching_mols, abx_mols
        )
    if len(ts_mols) > 0:
        all_matching_mols_df[
            "tan to nearest ts"
        ] = for_mol_list_get_highest_tanimoto_to_closest_mol(
            current_all_matching_mols, ts_mols
        )

    if cpd_sim_to_abx > 0:
        all_matching_mols_df = all_matching_mols_df[
            all_matching_mols_df["tan to nearest abx"] < cpd_sim_to_abx
        ]
        print(
            "length of all preds with tan abx < " + str(cpd_sim_to_abx) + ": ",
            len(all_matching_mols_df),
        )
    if cpd_sim_to_train_set > 0:
        all_matching_mols_df = all_matching_mols_df[
            all_matching_mols_df["tan to nearest ts"] < cpd_sim_to_train_set
        ]
        print(
            "length of all preds with tan ts < " +
            str(cpd_sim_to_train_set) + ": ",
            len(all_matching_mols_df),
        )

    # part 8: final round of visualization
    # cluster molecules based on fragment they have
    frags = [Chem.MolFromSmiles(smi)
             for smi in list(rank_df["fragment_SMILES"])]
    all_matching_mols_df = cluster_mols_based_on_fragments(
        all_matching_mols_df,
        compound_smi_col,
        cpd_name_col,
        frags,
        result_path
    )
    rank_df.to_csv(
        result_path +
        "FINAL_fragments_with_metadata.csv",
        index=False)

    # group fragments and see clusters
    final_folder = result_path + "FINAL_mols_after_all_thresholding/"
    os.mkdir(final_folder)
    mols = add_legends_to_compounds(
        all_matching_mols_df,
        smiles_column=compound_smi_col,
        name_column=cpd_name_col
    )
    df = extract_legends_and_plot(
        all_matching_mols_df,
        mols,
        "cluster.png",
        path=final_folder,
        murcko_scaffold=True,
        num_clusters=int(len(all_matching_mols_df) / 5),
    )
    all_matching_mols_df.to_csv(final_folder +
                                "final_proposed_molecules_to_order.csv")

    return all_matching_mols_df


def mini_algo(
    fragment_path,
    compound_path,
    result_path,
    fragment_smi_col="smiles",
    compound_smi_col="smiles",
    fragment_hit_col="hit",
    compound_hit_col="hit",
    fragment_score=0.2,
    compound_score=0.2,
    fragment_require_more_than_coh=True,
    fragment_remove_pains_brenk="both",
    compound_remove_pains_brenk="both",
    fragment_druglikeness_filter=[],
    compound_druglikeness_filter=[],
    fragment_remove_patterns=[],
    frags_cannot_disrupt_rings=False,
):
    """Condensed version of pipeline for controls.

    :param fragment_path: Path to the file containing fragment data.
    :param compound_path: Path to the file containing compound data.
    :param result_path: Path to store the results.
    :param fragment_smi_col: Column name for fragment SMILES.
    :param compound_smi_col: Column name for compound SMILES.
    :param fragment_hit_col: Column name indicating fragment hit.
    :param compound_hit_col: Column name indicating compound hit.
    :param fragment_score: Score threshold for fragments.
    :param compound_score: Score threshold for compounds.
    :param fragment_require_more_than_coh: Flag indicating if fragments
        require more than C, O, H atoms.
    :param fragment_remove_pains_brenk: Method to remove PAINS/BRENK fragments.
    :param compound_remove_pains_brenk: Method to remove PAINS/BRENK compounds.
    :param fragment_druglikeness_filter: Drug-likeness filter for fragments.
    :param compound_druglikeness_filter: Drug-likeness filter for compounds.
    :param fragment_remove_patterns: Patterns to remove from fragments.
    :param frags_cannot_disrupt_rings: Flag indicating if fragments
        cannot disrupt rings.
    """
    # part 1: process frags and compounds #####
    print("\nProcessing fragments...")
    df, mols, _ = process_dataset(
        frag_or_cpd="frag",
        path=fragment_path,
        score=fragment_score,
        smi_col=fragment_smi_col,
        hit_col=fragment_hit_col,
        require_more_than_coh=fragment_require_more_than_coh,
        remove_pains_brenk=fragment_remove_pains_brenk,
        druglikeness_filter=fragment_druglikeness_filter,
        remove_patterns=fragment_remove_patterns,
    )
    print("\nProcessing compounds...")
    cpd_df, cpd_mols, _ = process_dataset(
        frag_or_cpd="cpd",
        path=compound_path,
        score=compound_score,
        smi_col=compound_smi_col,
        hit_col=compound_hit_col,
        require_more_than_coh=False,
        remove_pains_brenk=compound_remove_pains_brenk,
        druglikeness_filter=compound_druglikeness_filter,
        remove_patterns=[],
    )
    print("\nMatching fragments in compounds...")

    # part 2: get all matching frag / molecule pairs #####
    frag_match_indices, cpd_match_indices_lists = match_frags_and_mols(
        mols, cpd_mols)
    if frags_cannot_disrupt_rings:
        # for all matching fragments, keep only matches that do not disrupt
        # rings
        frag_match_indices, cpd_match_indices_lists = check_for_complete_ring_fragments(  # noqa
            mols, frag_match_indices, cpd_mols, cpd_match_indices_lists
        )
    rank_df = compile_results_into_df(
        df,
        cpd_df,
        mols,
        frag_match_indices,
        cpd_match_indices_lists,
        result_path,
        frag_hit_column=fragment_hit_col,
        cpd_hit_column=compound_hit_col,
    )
    return (rank_df, cpd_df)
