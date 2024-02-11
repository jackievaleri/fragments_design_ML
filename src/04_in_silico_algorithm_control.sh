
mkdir ../models/SA/in_silico_algorithm_control/

chemprop_train --data_path ../data/training_data/SA/2300_sa_screen.csv --features_generator rdkit_2d_normalized --save_dir ../models/SA/in_silico_algorithm_control/ --dataset_type classification --no_features_scaling  --num_folds 30 --ensemble_size 1 --split_sizes 0.8 0.1 0.1 --smiles_column smiles --split_type scaffold_balanced --target_column class --metric prc-auc --extra_metrics auc --dropout 0.05 --hidden_size 900 --ffn_num_layers 1 --depth 5 

chemprop_predict --test_path ../data/training_data/SA/37K_sa_screen.csv --features_generator rdkit_2d_normalized --checkpoint_dir ../models/SA/in_silico_algorithm_control/ --preds_path ../out/controls/in_silico_algorithm_control_SA_37k_predictions_with_2300_model.csv --no_features_scaling --smiles_column SMILES
