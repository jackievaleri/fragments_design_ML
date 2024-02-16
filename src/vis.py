"""Visualization functions."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rdkit import Chem


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
