"""Visualization functions."""

import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from IPython.display import display


from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold


def convert_df_smis_to_fps(df, smi_column="SMILES"):
    """
    Convert SMILES in DataFrame to RDKit fingerprints.

    This function takes a DataFrame and a column name containing SMILES.
    It converts the SMILES strings to RDKit mols and then to fingerprints.

    :param df: DataFrame containing SMILES strings
    :param smi_column: Name of the column containing SMILES (default: "SMILES")
    :return: Tuple containing the original SMILES and RDKit fingerprints
    """
    smiles = list(df[smi_column])
    mols = [Chem.MolFromSmiles(x) for x in smiles if not isinstance(x, float)]
    smis = [x for x in smiles if not isinstance(x, float)]
    fps = [Chem.RDKFingerprint(x) for x in mols if x is not None]
    smis = [x for x, y in zip(smis, mols) if y is not None]
    return (smis, fps)


def make_joined_list_of_fps_and_labels(list_of_lists, labels_in_order):
    """
    Make a joined list of fingerprints and their corresponding labels.

    This function takes a list of lists containing fps and labels,
    and joins them into a single list of fingerprints and a list of labels.
    Note, this could be really slow for long lists or many long lists

    :param list_of_lists: List of lists containing fingerprints
    :param labels_in_order: List of labels in the same order as the fps
    :return: Tuple containing the joined list of fingerprints and labels
    """
    fp_list = []
    lab_list = []
    for lab, currlist in zip(labels_in_order, list_of_lists):
        fp_list.extend(currlist)
        lab_list.extend([lab] * len(currlist))
    return (fp_list, lab_list)


def tsne_from_pca_components(fp_list, fp_labels):
    """
    Perform t-Distributed Stochastic Neighbor Embedding using PCA components.

    This function takes a list of fingerprints and their corresponding labels,
    performs Principal Component Analysis (PCA) to reduce dimensionality, and
    applies t-SNE to further reduce the dimensionality to 2 dimensions.

    :param fp_list: List of fingerprints
    :param fp_labels: List of labels corresponding to the fingerprints
    :return: DataFrame containing the 2D tSNE embeddings with labels
    """
    pca = PCA(n_components=np.min([len(fp_list), len(fp_list[0])]))
    crds = pca.fit_transform(fp_list)
    crds_embedded = TSNE(n_components=2).fit_transform(crds)
    tsne_df = pd.DataFrame(crds_embedded, columns=["X", "Y"])
    tsne_df["label"] = fp_labels
    return tsne_df


def make_tsne_figure(tsne_df, fp_labels, fig_path, colors=None):
    """
    Create a t-SNE scatter plot figure.

    This function takes a DataFrame containing t-SNE embeddings, labels,
    and a path to save the figure. It makes a scatter plot of t-SNE embeddings
    with points colored by their labels and saves the figure in PNG and SVG.

    :param tsne_df: DataFrame containing t-SNE embeddings
        requires columns "X", "Y", and "label"
    :param fp_labels: List of labels corresponding to the t-SNE embeddings
    :param fig_path: Path to save the figure (without extension)
    :param colors: Optional list of colors for each label (default: None)
    """
    plt.figure(figsize=(8, 5), dpi=300)
    if colors is None:
        palette = sns.color_palette("hls", len(set(fp_labels)))
    else:
        ordered_labs = tsne_df.drop_duplicates("label")
        palette = dict(zip(list(ordered_labs["label"]), colors))
    sns.scatterplot(
        data=tsne_df,
        x="X",
        y="Y",
        hue="label",
        palette=palette,
        alpha=0.7,
        s=4
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path + ".png")
    plt.savefig(fig_path + ".svg")
    plt.show()


def clusterFps(fps, num_clusters):
    """
    Cluster fingerprints using Agglomerative Clustering.

    Code adapted from
    https://www.macinchem.org/reviews/clustering/clustering.php

    :param fps: List of fingerprints.
    :param num_clusters: Number of clusters to create.
    :return: Tuple containing cluster labels and dictionary of final clusters.
    """
    tan_array = [DataStructs.BulkTanimotoSimilarity(i, fps) for i in fps]
    tan_array = np.array(tan_array)
    clusterer = AgglomerativeClustering(
        n_clusters=num_clusters, compute_full_tree=True
    ).fit(tan_array)
    final_clusters = {}
    for ix, m in enumerate(clusterer.labels_):
        if m in final_clusters:
            curr_list = final_clusters[m]
            curr_list.append(ix)
            final_clusters[m] = curr_list
        else:
            final_clusters[m] = [ix]
    return clusterer.labels_, final_clusters


def draw_mols(mols, legends, file_path, cut_down_size=False,
              black_and_white=False):
    """
    Draw molecular structures and save the image to a file.

    Code with help from the GOAT greg landrum:
    https://gist.github.com/greglandrum/d5f12058682f6b336905450e278d3399

    :param mols: List of RDKit Mol objects representing the molecules to draw.
    :param legends: List of legend strings corresponding to each molecule.
    :param file_path: File path to save the image.
    :param cut_down_size: Whether to limit the number of mols to draw to 10.
        Defaults to False.
    :param black_and_white: Whether to draw the molecules in black and white.
        Defaults to False.
    """
    if cut_down_size:
        mols = mols[0:10]
        legends = legends[0:10]
    molsPerRow = 5
    subImgSize = (500, 500)
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow:
        nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(
        fullSize[0], fullSize[1], subImgSize[0], subImgSize[1]
    )
    d2d.drawOptions().legendFontSize = 100
    if black_and_white:
        d2d.drawOptions().useBWAtomPalette()
    d2d.DrawMolecules(mols, legends=legends)
    d2d.FinishDrawing()
    open(file_path, "wb+").write(d2d.GetDrawingText())


def plot_final_fragments_with_all_info(
    df,
    output_folder,
    frags,
    cpd_mols,
    cpd_names,
    abx_mols,
    abx_names,
    ts_mols,
    ts_names,
    display_inline_candidates,
    plot_neighborhoods=True,
):
    """
    Plot final fragments with all information including matched molecules,
        antibiotics, and training set compounds.

    :param df: DataFrame containing information about frags and their matches.
    :param output_folder: Path to the folder where images will be saved.
    :param frags: List of RDKit Mol objects representing the fragments.
    :param cpd_mols: List of RDKit Mol objects representing the compounds.
    :param cpd_names: List of names corresponding to the compounds.
    :param abx_mols: List of RDKit Mol objects representing the antibiotics.
    :param abx_names: List of names corresponding to the antibiotics.
    :param ts_mols: List of RDKit Mol objects representing the train set cpds.
    :param ts_names: List of names corresponding to the training set compounds.
    :param display_inline_candidates: Whether to display candidates inline.
    :param plot_neighborhoods: Whether to plot neighborhood information.
        Defaults to True.
    """
    index = 0
    for _, row in df.iterrows():
        frag_match_index = row["matched_fragments"]
        if display_inline_candidates:
            print("-------------------------------------------------------------------")  # noqa
            print("candidate: ", index)
            print("SMILES: ", row["fragment_SMILES"])
            print("fragment index: " + str(frag_match_index))
            display(frags[frag_match_index])
            print("fragment score: ", np.round(row["fragment_scores"], 3))
            print(
                "average matched molecule score: ",
                np.round(row["average_molecule_score"], 3),
            )
            print(
                "number of matched broad800k molecules: ", len(
                    row["matched_molecules"])
            )
            try:
                print(
                    "average matched hepg2 growth: ",
                    np.round(
                        row["average_hepg2_tox_score_of_matched_molecules"],
                        3),
                )
                print(
                    "average matched primary growth: ",
                    np.round(
                        row["average_primary_tox_score_of_matched_molecules"],
                        3),
                )
            except BaseException:
                continue
            print("length of fragment: ", row["length_of_fragment"])

        if len(abx_mols) > 0 and display_inline_candidates:
            abx_index_list = row["matched_antibiotics"]
            print("matching abx")
            img = Draw.MolsToGridImage(
                [m for i, m in enumerate(abx_mols) if i in abx_index_list],
                molsPerRow=10,
                maxMols=100,
                legends=[str(abx_names[i]) for i in abx_index_list],
            )
            display(img)
        elif display_inline_candidates:
            print("no matching abx found")

        if len(ts_mols) > 0 and display_inline_candidates:
            ts_index_list = row["matched train set molecules"]
            print("matching train set compounds")
            img = Draw.MolsToGridImage(
                [m for i, m in enumerate(ts_mols) if i in ts_index_list],
                molsPerRow=10,
                maxMols=100,
                legends=[str(ts_names[i]) for i in ts_index_list],
            )
            display(img)
        elif display_inline_candidates:
            print("no matching train set molecules found")

        full_mol_index_list = row["matched_molecules"]
        if len(full_mol_index_list) > 0:
            curr_match_names = [cpd_names[i] for i in full_mol_index_list]
            curr_match_names = [
                x if isinstance(x, str) else "nan" for x in curr_match_names
            ]
            curr_match_scores = [
                float(x) for x in list(
                    row["full_molecule_scores"])]
            if len(abx_mols) > 0:
                curr_tan_abxs = [
                    float(x)
                    for x
                    in list(row["tanimoto_scores_of_full_mols_to_nearest_abx"])
                ]
            else:
                curr_tan_abxs = [-1] * len(curr_match_scores)
            if len(ts_mols) > 0:
                curr_tan_tss = [
                    float(x)
                    for x in list(
                        row["tanimoto_scores_of_full_mols_to_nearest_train_set"]  # noqa
                    )
                ]
            else:
                curr_tan_tss = [-1] * len(curr_match_scores)
            (
                curr_match_scores,
                curr_tan_abxs,
                curr_tan_tss,
                curr_match_mol_index_list,
                curr_match_names,
            ) = zip(
                *sorted(
                    zip(
                        curr_match_scores,
                        curr_tan_abxs,
                        curr_tan_tss,
                        full_mol_index_list,
                        curr_match_names,
                    ),
                    reverse=True,
                )
            )
            curr_match_mols = [
                m for i, m in enumerate(cpd_mols)
                if i in curr_match_mol_index_list
            ]
            legends = [
                n
                + ", "
                + str(np.round(sc, 3))
                + "\n tan score to closest abx: "
                + str(np.round(ta, 3))
                + "\n tan score to closest TS: "
                + str(np.round(tt, 3))
                for n, sc, ta, tt in zip(
                    curr_match_names,
                    curr_match_scores,
                    curr_tan_abxs,
                    curr_tan_tss
                )
            ]
            if display_inline_candidates:
                print("matching molecules")
                img = Draw.MolsToGridImage(
                    curr_match_mols, molsPerRow=5, maxMols=500, legends=legends
                )
                display(img)
            draw_mols(
                curr_match_mols,
                legends,
                output_folder + str(index) + "_" +
                row["fragment_SMILES"] + ".png",
                cut_down_size=True,
            )
        else:
            if display_inline_candidates:
                print("no matching molecules found")

        if plot_neighborhoods:
            if display_inline_candidates:
                print("random sampling of cpds with and without frag:")
            cpds_w_frag = list(row["random_analogue_cpds_w_frag"])
            scos_w_frag = list(row["random_analogue_scos_w_frag"])
            cpds_wo_frag = list(row["random_analogue_cpds_without_frag"])
            scos_wo_frag = list(row["random_analogue_scos_without_frag"])

            if len(cpds_w_frag) > 1 and len(cpds_wo_frag) > 1:
                ana_w_frag_legends = [str(np.round(x, 3)) for x in scos_w_frag]
                if display_inline_candidates:
                    print(
                        "avg difference of analogues with and without frag: ",
                        np.round(
                            row["average_difference_with_and_without_frag"],
                            3),
                    )
                    print(
                        "t test of analogues with and without frag: ",
                        row["ttest_scos_with_and_without_frag"],
                    )
                    print("random analogues with frag:")
                    display(
                        Draw.MolsToGridImage(
                            cpds_w_frag, legends=ana_w_frag_legends)
                    )
                draw_mols(
                    cpds_w_frag,
                    ana_w_frag_legends,
                    output_folder + str(index) + "_random_mols_with_frag.png",
                    cut_down_size=True,
                )

                ana_wo_frag_legends = [str(np.round(x, 3))
                                       for x in scos_wo_frag]
                if display_inline_candidates:
                    print("random analogues without frag:")
                    display(
                        Draw.MolsToGridImage(
                            cpds_wo_frag, legends=ana_wo_frag_legends)
                    )
                draw_mols(
                    cpds_wo_frag,
                    ana_wo_frag_legends,
                    output_folder + str(index) +
                    "_random_mols_without_frag.png",
                    cut_down_size=True,
                )

            else:
                if display_inline_candidates:
                    print("not enough matching analogues found")

        index = index + 1


def add_legends_to_compounds(df, smiles_column="SMILES", name_column="Name"):
    """
    Add legends to compounds based on DataFrame columns.

    :param df: DataFrame containing compound information.
    :param smiles_column: Name of the column containing SMILES representations.
        Defaults to "SMILES".
    :param name_column: Name of the column containing compound names.
        Defaults to "Name".
    :return: List of RDKit Mol objects with legends added.
    """
    df["row_num"] = list(range(len(df)))
    df = df.drop_duplicates(subset=smiles_column)
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(mol) for mol in smis]

    for row_num, smi in enumerate(smis):
        try:
            _ = Chem.MolFromSmiles(smi)
            row = df.iloc[row_num, :]
            actual_row_num = str(row.loc["row_num"])
            actualrowname = str(row.loc[name_column])
            legend = (
                str(actual_row_num) + ", " +
                actualrowname + "\n" + "SMILES: " + smi
            )
        except Exception as e:
            print(e)
            actual_row_num = str(row.loc["row_num"])
            legend = "row: " + actual_row_num
        mols[row_num].SetProp("legend", legend)
    return mols


def add_legends_to_fragments(df, smiles_column="fragment_SMILES"):
    """
    Add legends to fragments based on DataFrame columns.

    :param df: DataFrame containing fragment information.
    :param smiles_column: Name of the column containing SMILES representations.
        Defaults to "fragment_SMILES".
    :return: List of RDKit Mol objects with legends added.
    """
    df["row_num"] = list(range(len(df)))
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(mol) for mol in smis]
    for row_num, _ in enumerate(smis):
        legend = "Candidate: " + str(row_num)
        mols[row_num].SetProp("legend", legend)
    return mols


def extract_legends_and_plot(
    df, mols, plot_suffix, path, num_clusters=30, murcko_scaffold=False
):
    """
    Extract legends and plot clusters.

    :param df: DataFrame containing cluster information.
    :param mols: List of RDKit Mol objects.
    :param plot_suffix: Suffix to add to the plot file name.
    :param path: Path to save the plot files.
    :param num_clusters: Number of clusters. Defaults to 30.
    :param murcko_scaffold: Whether to use Murcko scaffolds. Defaults to False.
    :return: DataFrame with added cluster information.
    """
    if num_clusters == 0:
        num_clusters = 1
    if num_clusters > len(df):
        num_clusters = len(df)
    molsPerRow = 4
    subImgSize = (500, 500)

    if murcko_scaffold:
        mols = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
    raw_cluster_labels, final_clusters = clusterFps(
        fps, num_clusters=num_clusters)

    img_list = []
    name_index = 0
    for cluster_key in final_clusters:
        cluster_mols = final_clusters[cluster_key]
        cluster_mols = [mols[i] for i in cluster_mols]

        nRows = len(cluster_mols) // molsPerRow
        if len(cluster_mols) % molsPerRow:
            nRows += 1
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
        d2d = rdMolDraw2D.MolDraw2DCairo(
            fullSize[0], fullSize[1], subImgSize[0], subImgSize[1]
        )
        d2d.drawOptions().legendFontSize = 100
        d2d.DrawMolecules(
            cluster_mols, legends=[
                mol.GetProp("legend") for mol in cluster_mols]
        )  # assume mols have legend property
        d2d.FinishDrawing()

        new_name = path + str(name_index) + "_" + plot_suffix
        open(new_name, "wb+").write(d2d.GetDrawingText())
        img_list.append(new_name)
        name_index = name_index + 1
    df["cluster"] = [str(i) for i in raw_cluster_labels]
    return df


def cluster_mols_based_on_fragments(
    df, compound_smi_col, cpd_name_col, frags, result_path
):
    """
    Cluster molecules based on fragments.

    :param df: DataFrame containing compound information.
    :param compound_smi_col: Column name for compound SMILES.
    :param cpd_name_col: Column name for compound names.
    :param frags: List of RDKit Mol objects representing fragments.
    :param result_path: Path to save the clustering results.
    :return: DataFrame with added fragment matching information.
    """
    mols = add_legends_to_compounds(
        df, smiles_column=compound_smi_col, name_column=cpd_name_col
    )
    matching_frag_indices = []
    # find the fragments they match to
    for i, mol in enumerate(mols):
        frag_indices = []
        for j, frag in enumerate(
                frags):  # only works bc all smiles make valid mols
            if mol.HasSubstructMatch(frag):
                frag_indices.append(j)
        matching_frag_indices.append(frag_indices)
    df["matching_frags"] = matching_frag_indices

    folder = result_path + "FINAL_mols_clustered_by_fragment/"
    os.mkdir(folder)
    df = extract_legends_and_plot(
        df,
        mols,
        plot_suffix="cluster.png",
        path=folder,
        murcko_scaffold=True,
        num_clusters=int(len(df) / 5),
    )
    return df
