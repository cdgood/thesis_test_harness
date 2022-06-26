python main.py --output_path .\training_results\3d-densenet_all\pt1 --experiments_json ./experiments/issc2/3d-densenet_all_issc_pt1.json
python main.py --output_path .\training_results\3d-densenet_all\pt2 --experiments_json ./experiments/issc2/3d-densenet_all_issc_pt2.json
python main.py --output_path .\training_results\3d-densenet_all\pt3 --experiments_json ./experiments/issc2/3d-densenet_all_issc_pt3.json
python main.py --output_path .\training_results\3d-densenet_all\pt4 --experiments_json ./experiments/issc2/3d-densenet_all_issc_pt4.json
python main.py --output_path .\training_results\3d-densenet_all\pt5 --experiments_json ./experiments/issc2/3d-densenet_all_issc_pt5.json
python main.py --output_path .\training_results\3d-densenet_all\pt6 --experiments_json ./experiments/issc2/3d-densenet_all_issc_pt6.json

cd training_results\3d-densenet_all

copy /b pt1\3d-densenet_all_issc_pt1_results.csv + pt2\3d-densenet_all_issc_pt2_results.csv + pt3\3d-densenet_all_issc_pt3_results.csv + pt4\3d-densenet_all_issc_pt4_results.csv + pt5\3d-densenet_all_issc_pt5_results.csv + pt6\3d-densenet_all_issc_pt6_results.csv 3d-densenet_all_issc_results.csv
copy /b pt1\3d-densenet_all_issc_pt1__grss_dfc_2018__class_results.csv + pt2\3d-densenet_all_issc_pt2__grss_dfc_2018__class_results.csv + pt3\3d-densenet_all_issc_pt3__grss_dfc_2018__class_results.csv + pt4\3d-densenet_all_issc_pt4__grss_dfc_2018__class_results.csv + pt5\3d-densenet_all_issc_pt5__grss_dfc_2018__class_results.csv + pt6\3d-densenet_all_issc_pt6__grss_dfc_2018__class_results.csv 3d-densenet_all_issc__grss_dfc_2018__class_results.csv

cd ..\..\..