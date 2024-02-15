#### Imports ####
from crem.crem import mutate_mol, grow_mol
from multiprocessing import Pool
import pandas as pd
import subprocess
import numpy as np
import random
import os
import tqdm
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, Image
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # shut off warnings
from rdkit.Chem import RDConfig

sys.path.append(
    os.path.join(RDConfig.RDContribDir, "SA_Score")
)  # now you can import sascore!
import sascorer
import cairosvg

#### Global Variables ####

IPythonConsole.molSize = (400, 300)
IPythonConsole.ipython_useSVG = True

# specify the path to the fragment database (replace with your actual path)
db_fname = "../data/static_datasets/replacements02_sc2.5.db"

# gate on tanimoto similarity to abx
abx = pd.read_csv(
    "../data/static_datasets/04052022_CLEANED_v5_antibiotics_across_many_classes.csv"
)
abx_smiles = list(abx["Smiles"])
abx_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in abx_smiles]

#### Helper Functions ####

# modified for NG model where have to save features ahead of time
def calculateScoreThruChemprop(
    patterns, results_folder_name, results_file_name, model_path, hit_column
):
    # write to clean csv
    clean_name = results_folder_name + results_file_name
    new_df = pd.DataFrame(patterns, columns=["SMILES"])
    new_df.to_csv(clean_name, index=False)

    # use subprocess to run command line thru jupyter notebook - could easily just run command line but this is automated
    activate_command = "source ~/opt/anaconda3/bin/activate; conda activate chemprop; "
    genML_folder = "../"  # from chemprop folder

    if "gonorrhea" in model_path:
        ft_name = clean_name.split(".csv")[0] + "_features.npz"
        ft_command = (
            "python scripts/save_features.py --data_path "
            + genML_folder
            + clean_name
            + " --features_generator rdkit_2d_normalized --save_path "
            + genML_folder
            + ft_name
            + "; "
        )
        run_command = (
            "python predict.py --test_path "
            + genML_folder
            + clean_name
            + " --checkpoint_dir "
            + model_path
            + " --preds_path "
            + genML_folder
            + clean_name
            + " --features_path "
            + genML_folder
            + ft_name
            + " --no_features_scaling --smiles_columns SMILES"
        )
        full_command = activate_command + ft_command + run_command
    else:  # no features during training
        run_command = (
            "python predict.py --test_path "
            + genML_folder
            + clean_name
            + " --checkpoint_dir "
            + model_path
            + " --preds_path "
            + genML_folder
            + clean_name
            + " --features_generator rdkit_2d_normalized --no_features_scaling --smiles_columns SMILES"
        )
        full_command = activate_command + run_command

    test = subprocess.run(
        full_command, cwd="../models/chemprop-master/", shell=True, capture_output=True
    )
    # print(test)
    preds = pd.read_csv(clean_name)

    new_smis = list(preds["SMILES"])
    new_scores = list(preds[hit_column])
    if len(new_smis) > 0:
        return [new_smis, new_scores]
    else:
        return []


def get_molecule_scores(ms, intermediate_folder, model_path, hit_column):
    # get the score of the model from the relevant path
    smis = [Chem.MolToSmiles(m) for m in ms]
    smis, scores = calculateScoreThruChemprop(
        smis, intermediate_folder, "_scores.csv", model_path, hit_column
    )
    return (smis, scores)


def calculateScoreThruToxModel(
    patterns, results_folder_name, results_file_name, tox_model
):
    # write to clean csv
    clean_name = results_folder_name + tox_model + "_" + results_file_name
    new_df = pd.DataFrame(patterns, columns=["SMILES"])
    new_df.to_csv(clean_name, index=False)

    # use subprocess to run command line thru jupyter notebook - could easily just run command line but this is automated
    activate_command = "source ~/opt/anaconda3/bin/activate; conda activate chemprop; "
    genML_folder = "../"  # from chemprop folder

    if tox_model == "primary":
        model_folder = "final_tox_primary/"
    elif tox_model == "hepg2":
        model_folder = "final_tox_hepg2/"
    run_command = (
        "python predict.py --test_path "
        + genML_folder
        + clean_name
        + " --checkpoint_dir ../"
        + model_folder
        + " --preds_path "
        + genML_folder
        + clean_name
        + " --features_generator rdkit_2d_normalized --no_features_scaling --smiles_columns SMILES"
    )
    full_command = activate_command + run_command
    test = subprocess.run(
        full_command, cwd="../models/chemprop-master/", shell=True, capture_output=True
    )
    print(test)
    preds = pd.read_csv(clean_name)

    new_smis = list(preds["SMILES"])
    new_scores = list(preds["TOXICITY"])
    if len(new_smis) > 0:
        return [new_smis, new_scores]
    else:
        return []


def get_sim(ms, ref_fps):
    # ms - list of molecules
    # ref_fps - list of fingerprints of reference molecules
    output = []
    fps1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in ms]
    for fp in fps1:
        v = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        i = np.argmax(v)
        output.append([v[i], i])
    return output


def select_top_based_on_criteria(
    smis, scores, regular_score=False, num_top_to_get=5, num_random_to_get=5
):
    # score will be chemprop score
    sorteddf = pd.DataFrame()
    sorteddf["SMILES"] = smis
    sorteddf["scores"] = scores
    sorteddf = sorteddf.drop_duplicates(subset="SMILES").reset_index()
    sorteddf = sorteddf.sort_values("scores", ascending=False)

    if not regular_score:
        # add synthesizability score
        mols = [Chem.MolFromSmiles(smi) for smi in list(sorteddf["SMILES"])]
        sascores = [sascorer.calculateScore(mol) for mol in mols]
        sorteddf["SAScore"] = sascores

        # add tanimoto similarity to known abx
        query_fps = [Chem.RDKFingerprint(mol) for mol in mols]
        max_tans = [
            max(DataStructs.BulkTanimotoSimilarity(query_fp, abx_fps))
            for query_fp in query_fps
        ]
        sorteddf["max_tan_sim_to_abx"] = max_tans

        # add tox score - hepg2
        _, hepg2_toxs = calculateScoreThruToxModel(
            list(sorteddf["SMILES"]),
            "../out/crem/v6_adjusted_directed_score_02102023/",
            "_temp_predictions.csv",
            "hepg2",
        )
        sorteddf["hepg2_tox"] = hepg2_toxs
        _, prim_toxs = calculateScoreThruToxModel(
            list(sorteddf["SMILES"]),
            "../out/crem/v6_adjusted_directed_score_02102023/",
            "_temp_predictions.csv",
            "primary",
        )
        sorteddf["prim_tox"] = prim_toxs

        # calculate adjusted scores
        adj_scores = []
        for i, row in sorteddf.iterrows():
            chempropsco = row["scores"]
            sascore = row["SAScore"]
            tansim = row["max_tan_sim_to_abx"]
            hepg2 = row["hepg2_tox"]
            prim = row["prim_tox"]
            adj_score = (2.0 * chempropsco) - ((sascore / 10.0) + tansim + hepg2 + prim)
            adj_scores.append(adj_score)
        sorteddf["adjusted_score"] = adj_scores
    else:  # regular score
        sorteddf["adjusted_score"] = list(sorteddf["scores"])
    sorteddf = sorteddf.sort_values("adjusted_score", ascending=False).reset_index()

    if len(sorteddf) < num_top_to_get + num_random_to_get:
        good_smis = list(sorteddf["SMILES"])
        good_scos = list(sorteddf["adjusted_score"])
        return (
            [Chem.MolFromSmiles(smi) for smi in good_smis],
            good_scos,
            sorteddf,
            selecteddf,
        )
    good_smis = list(sorteddf.iloc[0:num_top_to_get]["SMILES"])
    good_scos = list(sorteddf.iloc[0:num_top_to_get]["adjusted_score"])
    for x in random.sample(
        list(range(num_top_to_get, len(sorteddf))), num_random_to_get
    ):
        new = sorteddf.iloc[x, :]["SMILES"]
        good_smis.append(new)
        new_sco = sorteddf.iloc[x, :]["adjusted_score"]
        good_scos.append(new_sco)

    selecteddf = sorteddf[[smi in good_smis for smi in list(sorteddf["SMILES"])]]
    return (
        [Chem.MolFromSmiles(smi) for smi in good_smis],
        good_scos,
        sorteddf,
        selecteddf,
    )


# borrowed code from example here: https://github.com/DrrDom/crem/blob/master/example/crem_example.ipynb and https://crem.readthedocs.io/en/latest/readme.html#
def generate_molecules(
    mols,
    clean_dir,
    orig_frag_to_protect,
    round_num,
    grow_or_mut="grow",
    params=None,
    catalog=None,
    model_path="",
    hit_column="hit",
):
    print("-----------------------------")
    print("iteration number: ", round_num)

    # grow_mol documentation: https://crem.readthedocs.io/en/latest/operations.html
    # use grow_mol with small max_atoms and small radius for very conservative changes
    # use mututate_mol with large change parameters for drastic changes
    if grow_or_mut == "grow":
        if params == None:
            new_mols = [
                list(grow_mol(mol, db_fname, return_mol=True, ncores=16))
                for mol in mols
            ]

        else:
            max_atoms = params[0]
            min_atoms = params[1]
            radius = params[2]
            new_mols = [
                list(
                    grow_mol(
                        mol,
                        db_fname,
                        max_atoms=max_atoms,
                        min_atoms=min_atoms,
                        radius=radius,
                        return_mol=True,
                        ncores=16,
                    )
                )
                for mol in mols
            ]
    elif grow_or_mut == "mut":
        if params == None:
            new_mols = [
                list(
                    mutate_mol(
                        Chem.AddHs(mol), db_name=db_fname, return_mol=True, ncores=16
                    )
                )
                for mol in mols
            ]
        else:
            min_size = params[0]
            max_size = params[1]
            min_inc = params[2]
            max_inc = params[3]
            radius = params[4]
            new_mols = [
                list(
                    mutate_mol(
                        Chem.AddHs(mol),
                        db_name=db_fname,
                        return_mol=True,
                        min_size=min_size,
                        max_size=max_size,
                        min_inc=min_inc,
                        max_inc=max_inc,
                        radius=radius,
                        ncores=16,
                    )
                )
                for mol in mols
            ]

    new_mols = [x for l in new_mols for x in l]
    new_mols = [Chem.RemoveHs(j[1]) for j in new_mols]
    print("molecules generated:", len(new_mols))

    # first remove any without fragment, or with PAINS and Brenk alerts
    # stringent but keep this step here for now
    good_new_mols = []
    frag_mol = Chem.MolFromSmiles(orig_frag_to_protect)
    for full_mol in new_mols:
        if full_mol.HasSubstructMatch(frag_mol):  # contains entirely the fragment
            if catalog != None:
                entry = catalog.GetFirstMatch(
                    full_mol
                )  # Get the first matching PAINS or Brenk
                if entry is None:  # no matching pains or brenk
                    good_new_mols.append(full_mol)
            else:  # no compound filtering
                good_new_mols.append(full_mol)
    new_mols = good_new_mols

    # write new mols to image
    img = Draw.MolsToGridImage(
        new_mols, molsPerRow=10, maxMols=50, useSVG=True
    )  # throws weird error if empty list is provided
    clean_name = clean_dir + "all_mols_round_" + str(round_num)
    cairosvg.svg2png(img.data, write_to=clean_name + ".png")

    # get new scores
    smis, scores = get_molecule_scores(
        new_mols, clean_dir + "all_mols_round_" + str(round_num), model_path, hit_column
    )  # doesn't include the previous round
    best_score = max(scores)
    print(
        "molecules generated containing fragment + passing PAINS, Brenk filters:",
        len(new_mols),
    )
    print("best score:", np.round(best_score, 3))

    return (smis, scores)


def run_crem(
    out_dir,
    orig_frag_smi,
    orig_mol_smi,
    max_atom_range,
    min_atom_range,
    radius_range,
    min_inc_range=[2],
    max_inc_range=[-2],
    num_iters=5,
    method="grow",
    regular_score=True,
    num_top_to_get=5,
    num_random_to_get=5,
    cpd_filter="both",
    model_path="",
    hit_column="hit",
):

    if method == "grow":
        new_big_dir = out_dir + "grow/"
    else:
        new_big_dir = out_dir + "mutate/"
    os.mkdir(new_big_dir)

    # initialize PAINS + Brenk filter
    if cpd_filter != "none":
        params = FilterCatalogParams()
        if cpd_filter == "pains" or cpd_filter == "both":
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        if cpd_filter == "brenk" or cpd_filter == "both":
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalog = FilterCatalog(params)
    else:
        catalog = None

    param_list = []
    if method == "grow":
        # grow params
        for (
            ma
        ) in (
            max_atom_range
        ):  # max_atoms – maximum number of atoms in the fragment which will replace H
            for (
                mi
            ) in (
                min_atom_range
            ):  # min_atoms – minimum number of atoms in the fragment which will replace H
                for (
                    ra
                ) in (
                    radius_range
                ):  # radius – radius of context which will be considered for replacement
                    param_list.append([ma, mi, ra])
    else:
        # mutate params
        for (
            mi_s
        ) in (
            min_atom_range
        ):  # min_size – minimum number of heavy atoms in a fragment to replace. If 0 - hydrogens will be replaced (if they are explicit)
            for (
                ma_s
            ) in (
                max_atom_range
            ):  # max_size – maximum number of heavy atoms in a fragment to replace
                for (
                    mi
                ) in (
                    min_inc_range
                ):  # min_inc – minimum change of a number of heavy atoms in replacing fragments to a number of heavy atoms in replaced one. Negative value means that the replacing fragments would be smaller than the replaced one on a specified number of heavy atoms
                    for (
                        ma
                    ) in (
                        max_inc_range
                    ):  # max_inc – maximum change of a number of heavy atoms in replacing fragments to a number of heavy atoms in replaced one
                        for (
                            ra
                        ) in (
                            radius_range
                        ):  # radius – radius of context which will be considered for replacement
                            param_list.append([mi_s, ma_s, mi, ma, ra])

    for param_set in tqdm.tqdm(param_list):

        if method == "grow":
            name = (
                str(param_set[0])
                + "_maxatom_"
                + str(param_set[1])
                + "_minatom_"
                + str(param_set[2])
                + "_radius"
            )
            clean_dir = out_dir + "grow/" + name + "/"
        else:
            name = (
                str(param_set[0])
                + "_minsize_"
                + str(param_set[1])
                + "_maxsize_"
                + str(param_set[2])
                + "_mininc_"
                + str(param_set[3])
                + "_maxinc_"
                + str(param_set[4])
                + "_radius"
            )
            clean_dir = out_dir + "mutate/" + name + "/"

        os.mkdir(clean_dir)
        print("***************************************************************")
        print(name)
        print("***************************************************************")

        # actually run the algorithm
        orig_mol = Chem.MolFromSmiles(orig_mol_smi)
        _, orig_mol_scos = calculateScoreThruChemprop(
            [orig_mol_smi],
            out_dir,
            "original_starting_mols_round_0.csv",
            model_path,
            hit_column,
        )
        selected_top_mols_to_mutate = [orig_mol]  # make another copy
        selected_top_mol_scos = list(orig_mol_scos)  # make another copy

        # first make a record
        savedf = pd.DataFrame()
        savedf["SMILES"] = [
            Chem.MolToSmiles(mol) for mol in selected_top_mols_to_mutate
        ]
        savedf["scores"] = selected_top_mol_scos
        savedf.to_csv(clean_dir + "original_starting_mols_round_0.csv", index=False)

        for i in range(num_iters):

            # run the algorithm to generate molecules
            smis, scores = generate_molecules(
                selected_top_mols_to_mutate,
                clean_dir,
                orig_frag_smi,
                i,
                grow_or_mut=method,
                params=param_set,
                catalog=catalog,
                model_path=model_path,
                hit_column=hit_column,
            )

            # select top compounds to continue with
            if np.max(param_set) >= 8:  # any parameters too big, must reduce the size
                num_top_to_get = 2
                num_random_to_get = 1
            (
                selected_top_mols_to_mutate,
                selected_top_mol_scos,
                alldf,
                selecteddf,
            ) = select_top_based_on_criteria(
                smis,
                scores,
                regular_score=regular_score,
                num_top_to_get=num_top_to_get,
                num_random_to_get=num_random_to_get,
            )

            # save the last mols generated
            alldf.to_csv(
                clean_dir + "all_mols_from_round_" + str(i) + ".csv", index=False
            )
            selecteddf.to_csv(
                clean_dir + "original_starting_mols_round_" + str(i + 1) + ".csv",
                index=False,
            )
