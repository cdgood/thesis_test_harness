@echo off
@break off
@title Run hyperspectral ISSC band selection tests
@color 0a
@cls

setlocal EnableDelayedExpansion

:: Create directories if they do not exist
set test_name=test_each_modality_and_checkpoints
set tr_dir=.\training_results\
set working_dir=%tr_dir%%test_name%\


:: Set parameter values
set random_seed=13
set epochs=200
set batch_size=16
set patch_size=13
set lr=0.00005
set save_period=10
set title_params=_lr5e-5_e%epochs%_b%batch_size%_p%patch_size%_rs%random_seed%_no-ts
set training_params=--random-seed %random_seed% --epochs %epochs% --batch-size %batch_size% --patch-size %patch_size% --lr %lr% --model-save-period %save_period%
set checkpoint_params=--random-seed %random_seed% --batch-size %batch_size% --patch-size %patch_size% --lr %lr%

:: Set flag list variables
set save_experiment_path=%working_dir%experiments\
set flags=--cuda 0 --dataset grss_dfc_2018 --skip-band-selection --skip-data-postprocessing --model-id 3d-densenet-modified --center-pixel --split-mode fixed --optimizer nadam --flip-augmentation --radiation-augmentation --mixture-augmentation
set hs_flags=--use-hs-data --hs-resampling average
set lidar_ms_flags=--use-lidar-ms-data
set lidar_ndsm_flags=--use-lidar-ndsm-data
set vhr_rgb_flags=--use-vhr-data --vhr-resampling cubic_spline


:: Set experiments
set titles[0]=
set experiment_flags[0]=

set titles[1]=test_hs%title_params%
set experiment_flags[1]=%flags% %hs_flags%

set titles[2]=test_lidar-ms%title_params%
set experiment_flags[2]=%flags% %lidar_ms_flags%

set titles[3]=test_lidar-ndsm%title_params%
set experiment_flags[3]=%flags% %lidar_ndsm_flags%

set titles[4]=test_vhr-rgb%title_params%
set experiment_flags[4]=%flags% %vhr_rgb_flags%

set titles[5]=test_hs_and_lidar-ms%title_params%
set experiment_flags[5]=%flags% %hs_flags% %lidar_ms_flags%

set titles[6]=test_hs_and_lidar-ndsm%title_params%
set experiment_flags[6]=%flags% %hs_flags% %lidar_ndsm_flags%

set titles[7]=test_hs_and_vhr-rgb%title_params%
set experiment_flags[7]=%flags% %hs_flags% %vhr_rgb_flags%

set titles[8]=test_lidar-ms_and_lidar-ndsm%title_params%
set experiment_flags[8]=%flags% %lidar_ms_flags% %lidar_ndsm_flags%

set titles[9]=test_lidar-ms_and_vhr-rgb%title_params%
set experiment_flags[9]=%flags% %lidar_ms_flags% %vhr_rgb_flags%

set titles[10]=test_lidar-ndsm_and_vhr-rgb%title_params%
set experiment_flags[10]=%flags% %lidar_ndsm_flags% %vhr_rgb_flags%

set titles[11]=test_hs_and_lidar-ms_and_lidar-ndsm%title_params%
set experiment_flags[11]=%flags% %hs_flags% %lidar_ms_flags% %lidar_ndsm_flags%

set titles[12]=test_hs_and_lidar-ms_and_vhr-rgb%title_params%
set experiment_flags[12]=%flags% %hs_flags% %lidar_ms_flags% %vhr_rgb_flags%

set titles[13]=test_hs_and_lidar-ndsm_and_vhr-rgb%title_params%
set experiment_flags[13]=%flags% %hs_flags% %lidar_ndsm_flags% %vhr_rgb_flags%

set titles[14]=test_lidar-ms_and_lidar-ndsm_and_vhr-rgb%title_params%
set experiment_flags[14]=%flags% %lidar_ms_flags% %lidar_ndsm_flags% %vhr_rgb_flags%

set titles[15]=test_all_modalities%title_params%
set experiment_flags[15]=%flags% %hs_flags% %lidar_ms_flags% %lidar_ndsm_flags% %vhr_rgb_flags%

set num_experiments=15


:: Move and combine results files
copy /b %working_dir%results\*__class_results.csv %working_dir%results\%test_name%%title_params%__class_results_tmp.csv
move %working_dir%results\*__class_results.csv %working_dir%results\individual\

copy /b %working_dir%results\*_results.csv %working_dir%results\%test_name%%title_params%__results_tmp.csv
move %working_dir%results\*_results.csv %working_dir%results\individual\

sort /c /unique %working_dir%results\%test_name%%title_params%__class_results_tmp.csv /o %working_dir%results\%test_name%%title_params%__class_results.csv
del %working_dir%results\%test_name%%title_params%__class_results_tmp.csv

sort /c /unique %working_dir%results\%test_name%%title_params%__results_tmp.csv /o %working_dir%results\%test_name%%title_params%__results.csv
del %working_dir%results\%test_name%%title_params%__results_tmp.csv



echo "test_hs_issc.bat script completed!"