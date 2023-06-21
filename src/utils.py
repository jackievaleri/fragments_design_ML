import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn import metrics

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Draw import rdMolDraw2D

#### Part 1: t-SNE Helper Functions

def convert_df_smis_to_fps(df, smi_column = 'SMILES'):
    smiles = list(df[smi_column])
    mols = [Chem.MolFromSmiles(x) for x in smiles if type(x) is not float]
    smis = [x for x in smiles if type(x) is not float]
    fps = [Chem.RDKFingerprint(x) for x in mols if x is not None]
    smis = [x for x,y in zip(smis, mols) if y is not None]
    return(smis,fps)

def make_joined_list_of_fps_and_labels(list_of_lists, labels_in_order):
    fp_list = []
    lab_list = []
    for lab, currlist in zip(labels_in_order, list_of_lists):
        fp_list.extend(currlist) # could be slow for long lists or many long lists
        lab_list.extend([lab] * len(currlist))
    return(fp_list, lab_list)

def tsne_from_pca_components(fp_list, fp_labels):
    # use all PCs
    pca = PCA(n_components=np.min([len(fp_list), len(fp_list[0])]))
    crds = pca.fit_transform(fp_list)

    # use PCs as input to tSNE
    crds_embedded = TSNE(n_components=2).fit_transform(crds)

    tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])
    tsne_df['label'] = fp_labels
    return(tsne_df)

def make_tsne_figure(tsne_df, fig_path, colors = None):
    plt.figure(figsize=(8,5), dpi = 300)
    if colors == None:
        palette = sns.color_palette("hls", len(set(fp_labels)))
    else:
        ordered_labs = tsne_df.drop_duplicates('label')
        palette = dict(zip(list(ordered_labs['label']), colors))
    ax = sns.scatterplot(data=tsne_df,x="X",y="Y",hue="label", palette = palette, alpha = 0.7, s = 4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(fig_path + '.png')
    plt.savefig(fig_path + '.svg')  
    plt.show()

#### Part 2: Model evaluation

def aupr(y_true, y_pred):
    # Compute precision-recall and plot curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr = float(auc(recall,precision))
    print('precision recall: ' + str(pr))

    fig, ax = plt.subplots(figsize = (2,2), dpi = 300)
    plt.clf()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()  

def evaluate_model(validation, cutoff_for_positive, actual_col = 'class', predicted_col = 'ACTIVITY'):
    actual = list(validation[actual_col])
    predicted = list(validation[predicted_col])

    aupr(actual, predicted)

    predicted_bin = [1.0 if x > cutoff_for_positive else 0.0 for x in list(predicted)]
    print('recall: ')
    print(sklearn.metrics.recall_score(actual, predicted_bin))
    print('precision: ')
    print(sklearn.metrics.precision_score(actual, predicted_bin))

#### Part 3: Helper functions

def draw_mols_with_highlight(mols, frag, legends, file_path, cut_down_size = False):
    
    patt = Chem.MolFromSmiles(frag)
    hit_ats = [mol.GetSubstructMatch(patt) for mol in mols]
    hit_bonds = []
    for mol, hit_at in zip(mols, hit_ats):
        hit_bond = []
        for bond in patt.GetBonds():
            aid1 = hit_at[bond.GetBeginAtomIdx()]
            aid2 = hit_at[bond.GetEndAtomIdx()]
            hit_bond.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
        hit_bonds.append(hit_bond)
    
    # code with help from the OG greg landrum: https://gist.github.com/greglandrum/d5f12058682f6b336905450e278d3399
    if cut_down_size:
        mols = mols[0:10]
        legends = legends[0:10]
        hit_ats = hit_ats[0:10]
        hit_bonds = hit_bonds[0:10]
    molsPerRow = 5
    subImgSize= (500,500)
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow:
        nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0],fullSize[1], subImgSize[0], subImgSize[1])
    
    # Set the drawing options
    d2d.drawOptions().legendFontSize=100
    d2d.drawOptions().useBWAtomPalette()
    color = matplotlib.colors.ColorConverter().to_rgb('lightskyblue')

    hit_atom_cols = [{b: color for b in hit_at} for hit_at in hit_ats]
    hit_bond_cols = [{b: color for b in hb} for hb in hit_bonds]
    d2d.DrawMolecules(mols,legends=legends, highlightAtoms=hit_ats,
                                  highlightBonds=hit_bonds, highlightAtomColors=hit_atom_cols, highlightBondColors=hit_bond_cols)
    d2d.FinishDrawing()
    open(file_path,'wb+').write(d2d.GetDrawingText())