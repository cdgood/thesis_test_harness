@echo off
@break off
@title Run hyperspectral manual band reduction tests with data augmentation
@color 0a
@cls

setlocal EnableDelayedExpansion

:: Create directories if they do not exist
set test_name=test_hs_manual
set tr_dir=.\training_results\
set working_dir=%tr_dir%%test_name%\

set dir[0]=%tr_dir%
set dir[1]=%working_dir%
set dir[2]=%working_dir%experiments\
set dir[3]=%working_dir%images\
set dir[4]=%working_dir%training_summaries\
set dir[5]=%working_dir%checkpoints\
set dir[6]=%working_dir%results\
set dir[7]=%working_dir%results\individual\

set num_dirs=7

for /l %%d in (0, 1, %num_dirs%) do (
    if not exist "!dir[%%d]!" (
        mkdir "!dir[%%d]!"
        if "!errorlevel!" == "0" (
            echo "'!dir[%%d]!' created successfully"
        ) else (
            echo "Error while creating '!dir[%%d]!'..."
        )
    ) else (
        echo "'!dir[%%d]!' already exists!"
    )
)


:: Set parameter values
set max_num_channels=48
set random_seed=13
set epochs=100
set batch_size=16
set patch_size=13
set lr=0.00005
set save_period=%epochs%
set title_params=_lr5e-5_e%epochs%_b%batch_size%_p%patch_size%_rs%random_seed%_no-ts
set training_params=--random-seed %random_seed% --epochs %epochs% --batch-size %batch_size% --patch-size %patch_size% --lr %lr%
set checkpoint_params=--random-seed %random_seed% --batch-size %batch_size% --patch-size %patch_size% --lr %lr%

:: Set flag list variables
set save_experiment_path=%working_dir%experiments\
set hs_flags=--use-hs-data --hs-resampling average
set lidar_ms_flags=--use-lidar-ms-data
set lidar_ndsm_flags=--use-lidar-ndsm-data
set vhr_rgb_flags=--use-vhr-data --vhr-resampling cubic_spline
set flags=--cuda 0 --dataset grss_dfc_2018 --skip-data-postprocessing --model-id 3d-densenet-modified --center-pixel --split-mode fixed --optimizer nadam --flip-augmentation --radiation-augmentation --mixture-augmentation %training_params% %hs_flags%

:: Set experiments
set titles[0]=test_remove_
set experiment_flags[0]=%flags% %hs_flags% --band-reduction-method manual --n-components 48 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47

set titles[1]=test_top_10_selected
set experiment_flags[1]=%flags% %hs_flags% --band-reduction-method manual --n-components 10 --selected-bands 6 14 16 19 27 32 37 43

set titles[2]=test_remove_21-23
set experiment_flags[2]=%flags% %hs_flags% --band-reduction-method manual --n-components 45 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47

set titles[3]=test_remove_26-28
set experiment_flags[3]=%flags% %hs_flags% --band-reduction-method manual --n-components 45 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47

set titles[4]=test_remove_21-23_26-28
set experiment_flags[4]=%flags% %hs_flags% --band-reduction-method manual --n-components 42 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 24 25 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47

set titles[5]=test_remove_40
set experiment_flags[5]=%flags% %hs_flags% --band-reduction-method manual --n-components 47 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 41 42 43 44 45 46 47

set titles[6]=test_remove_36-44
set experiment_flags[6]=%flags% %hs_flags% --band-reduction-method manual --n-components 39 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 45 46 47

set titles[7]=test_remove_21-23_26-28_40
set experiment_flags[7]=%flags% %hs_flags% --band-reduction-method manual --n-components 41 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 24 25 29 30 31 32 33 34 35 36 37 38 39 41 42 43 44 45 46 47

set titles[8]=test_remove_21-23_26-28_36-44
set experiment_flags[8]=%flags% %hs_flags% --band-reduction-method manual --n-components 33 --selected-bands 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 24 25 29 30 31 32 33 34 35 45 46 47


set num_experiments=8

:: Run Experiments
for /l %%x in (1, 1, %num_experiments%) do (
    python main.py --experiment-name !titles[%%x]! --experiment-number %%x --output-path %working_dir% --save-experiment-path "%save_experiment_path%!titles[%%x]!.json" !experiment_flags[%%x]!
    
    move %working_dir%*.csv %working_dir%results\
    move %working_dir%*.hdf5 %working_dir%checkpoints\
    move %working_dir%*.png %working_dir%images\
    move %working_dir%*.txt %working_dir%training_summaries\
)


:: Move and combine results files
copy /b %working_dir%results\*__class_results.csv %working_dir%results\%test_name%%title_params%__class_results_tmp.csv
move %working_dir%results\*__class_results.csv %working_dir%results\individual\

copy /b %working_dir%results\*__selected_band_results.csv %working_dir%results\%test_name%%title_params%__selected_band_results_tmp.csv
move %working_dir%results\*__selected_band_results.csv %working_dir%results\individual\

copy /b %working_dir%results\*_results.csv %working_dir%results\%test_name%%title_params%__results_tmp.csv
move %working_dir%results\*_results.csv %working_dir%results\individual\


sort /c /unique %working_dir%results\%test_name%%title_params%__class_results_tmp.csv /o %working_dir%results\%test_name%%title_params%__class_results.csv
del %working_dir%results\%test_name%%title_params%__class_results_tmp.csv

sort /c /unique %working_dir%results\%test_name%%title_params%__selected_band_results_tmp.csv /o %working_dir%results\%test_name%%title_params%__selected_band_results.csv
del %working_dir%results\%test_name%%title_params%__selected_band_results_tmp.csv

sort /c /unique %working_dir%results\%test_name%%title_params%__results_tmp.csv /o %working_dir%results\%test_name%%title_params%__results.csv
del %working_dir%results\%test_name%%title_params%__results_tmp.csv


echo "test_hs_manual_selection.bat script completed!"