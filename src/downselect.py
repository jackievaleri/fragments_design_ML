"""Helper functions for downselecting molecules."""

import pandas as pd
from IPython.display import display
from rdkit import Chem, DataStructs


def process_df(df,
               method,
               cpd,
               good_cols=['SMILES', 'ACTIVITY'],
               smiles_col='SMILES',
               hit_col='hit'):
    """
    Process a DataFrame for a specific method and compound.

    This function takes a DataFrame containing compound information,
    selects relevant columns, renames them,
    adds columns for the method and compound name,
    and prints the number of molecules
    generated by the method for the compound.

    :param df: DataFrame containing compound information
    :param method: Name of the method used
    :param cpd: Name of the compound
    :param good_cols: List of column names to keep
        (default: ['SMILES', 'ACTIVITY'])
    :param smiles_col: Name of the column containing SMILES
        (default: 'SMILES')
    :param hit_col: Name of the column containing hit information
        (default: 'hit')
    :return: Processed DataFrame with added 'Method' and 'Cpd' columns
    """
    df = df[[smiles_col, hit_col]]
    df.columns = good_cols
    df['Method'] = [method] * len(df)
    df['Cpd'] = [cpd] * len(df)
    print('# of ' + cpd + ' molecules generated by ' +
          method + ': ' + str(len(df)))
    return df


def for_mol_list_get_highest_tanimoto_to_closest_mol(
        cpd_mols, molecules_to_check):
    """
    Get the highest Tanimoto similarity to the closest molecule for each mol.

    This function calculates the Tanimoto similarity between each molecule.
    (`cpd_mols`) and a set of molecules to check (`molecules_to_check`),
    and returns the highest similarity score for each molecule in `cpd_mols`.
    Returns -1 for None mols.

    :param cpd_mols: List of RDKit Mol objects to compare
    :param molecules_to_check: List of RDKit Mol objects to check against
    :return: List of highest Tanimoto similarity for each molecule in cpd_mols
    """
    mols_fps = [Chem.RDKFingerprint(mo) for mo in cpd_mols]
    check_fps = [Chem.RDKFingerprint(mo) for mo in molecules_to_check]
    best_sims = []
    for m in mols_fps:
        if m is None:
            best_sims.append(-1)
        curr_highest_sim = 0
        for check in check_fps:
            sim = DataStructs.FingerprintSimilarity(m, check)
            # optionally, could implement a thresholding here
            if sim > curr_highest_sim:
                curr_highest_sim = sim
        best_sims.append(curr_highest_sim)
    return best_sims


def check_full_cpd_sim_to_closest_mols(
        smi_of_interest, selected_mols_df, df):
    """
    Check the full compound similarity to the closest selected molecules.

    This function calculates the Tanimoto similarity between a compound
    and the closest selected molecules in a DataFrame (`selected_mols_df`),
    and returns similarity scores for each molecule in another DataFrame.

    :param smi_of_interest: SMILES string of the compound of interest
    :param selected_mols_df: DataFrame containing selected molecules
        contains 'SMILES' column
    :param df: DataFrame containing molecules to compare with 'SMILES' column
    :return: List of Tanimoto similarity scores for each molecule in `df`
    """
    if len(selected_mols_df) == 0:
        return [0] * len(df)
    mol = Chem.MolFromSmiles(smi_of_interest)
    selected_mols = [
        Chem.MolFromSmiles(smi) for smi in list(
            selected_mols_df['SMILES'])]
    tan_scores = for_mol_list_get_highest_tanimoto_to_closest_mol(
        [mol], selected_mols)
    return tan_scores


def select_cpds_from_df(df, maximum, top_to_get,
                        max_tan_allowed_after_top=0.75):
    """
    Select compounds from a DataFrame based on specified criteria.

    This function selects compounds from a DataFrame (`df`)
    based on their activity levels,
    considering a maximum number of compounds (`maximum`)
    and a top number to consider (`top_to_get`).
    It also ensures that the Tanimoto similarity to
    previously selected compounds does not exceed a threshold
    (`max_tan_allowed_after_top`).

    :param df: DataFrame containing compound information
        with columns 'SMILES', 'Method', and 'ACTIVITY'
    :param maximum: Maximum number of compounds to select
    :param top_to_get: Number of top compounds to consider
        before applying the similarity constraint
    :param max_tan_allowed_after_top: Maximum Tanimoto similarity allowed
        after the top compounds are selected (default: 0.75)
    :return: DataFrame containing selected compounds
    """
    selected_mols_df = pd.DataFrame()
    df = df.sort_values('ACTIVITY', ascending=False).reset_index(drop=True)
    for i, row in df.iterrows():
        if i < top_to_get:
            display(Chem.MolFromSmiles(row['SMILES']))
            print(i, row['Method'], row['ACTIVITY'])
            selected_mols_df = pd.concat(
                [selected_mols_df, pd.DataFrame(row).T])
        else:
            if len(selected_mols_df) >= maximum:
                print("*********REACHED END: SELECTED " +
                      str(len(selected_mols_df)) + " MOLECULES*********")
                break
            else:
                smi_of_interest = row['SMILES']
                tan_to_selected_mols = check_full_cpd_sim_to_closest_mols(
                    smi_of_interest, selected_mols_df, df)
                if tan_to_selected_mols[0] >= max_tan_allowed_after_top:
                    continue
                else:
                    # SELECTION
                    display(Chem.MolFromSmiles(row['SMILES']))
                    print(i, row['Method'], row['ACTIVITY'])
                    selected_mols_df = pd.concat(
                        [selected_mols_df, pd.DataFrame(row).T])
    return selected_mols_df
