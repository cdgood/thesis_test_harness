python main.py --output_path .\training_results\3d-densenet_hs\checkpoint_results\c1 --experiments_json ./experiments/3d-densenet_hs_checkpoints1.json
python main.py --output_path .\training_results\3d-densenet_hs\checkpoint_results\c2 --experiments_json ./experiments/3d-densenet_hs_checkpoints2.json
python main.py --output_path .\training_results\3d-densenet_hs\checkpoint_results\c3 --experiments_json ./experiments/3d-densenet_hs_checkpoints3.json
python main.py --output_path .\training_results\3d-densenet_hs\checkpoint_results\c4 --experiments_json ./experiments/3d-densenet_hs_checkpoints4.json
python main.py --output_path .\training_results\3d-densenet_hs\checkpoint_results\c5 --experiments_json ./experiments/3d-densenet_hs_checkpoints5.json

cd training_results\3d-densenet_hs\checkpoint_results

copy /b c1\3d-densenet_hs_checkpoints1_results.csv + c2\3d-densenet_hs_checkpoints2_results.csv + c3\3d-densenet_hs_checkpoints3_results.csv + c4\3d-densenet_hs_checkpoints4_results.csv + c5\3d-densenet_hs_checkpoints5_results.csv 3d-densenet_hs_checkpoints_results.csv
copy /b c1\3d-densenet_hs_checkpoints1__grss_dfc_2018__class_results.csv + c2\3d-densenet_hs_checkpoints2__grss_dfc_2018__class_results.csv + c3\3d-densenet_hs_checkpoints3__grss_dfc_2018__class_results.csv + c4\3d-densenet_hs_checkpoints4__grss_dfc_2018__class_results.csv + c5\3d-densenet_hs_checkpoints5__grss_dfc_2018__class_results.csv 3d-densenet_hs_checkpoints__grss_dfc_2018__class_results.csv

cd ..\..\..