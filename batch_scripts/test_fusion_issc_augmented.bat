@echo off
@break off
@title Run ISSC band selection tests on late-stage data fusion of all modalities with data augmentation
@color 0a
@cls

setlocal EnableDelayedExpansion

:: Create directories if they do not exist
set test_name=test_fusion_issc_augmented
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
set max_num_channels=55
set random_seed=13
set epochs=100
set batch_size=16
set patch_size=13
set lr=0.00005
set save_period=%epochs%
set title_params=_lr5e-5_e%epochs%_b%batch_size%_p%patch_size%_rs%random_seed%_no-ts
set training_params=--random-seed %random_seed% --epochs %epochs% --batch-size %batch_size% --patch-size %patch_size% --lr %lr% --model-save-period %save_period%
set checkpoint_params=--random-seed %random_seed% --batch-size %batch_size% --patch-size %patch_size% --lr %lr%

:: Set flag list variables
set save_experiment_path=%working_dir%experiments\
set hs_flags=--use-hs-data --hs-resampling average
set lidar_ms_flags=--use-lidar-ms-data
set lidar_ndsm_flags=--use-lidar-ndsm-data
set vhr_rgb_flags=--use-vhr-data --vhr-resampling cubic_spline
set flags=--cuda 0 --dataset grss_dfc_2018 --skip-data-postprocessing --model-id 3d-densenet-fusion --center-pixel --split-mode fixed --optimizer nadam --flip-augmentation --radiation-augmentation --mixture-augmentation %training_params% %hs_flags% %lidar_ms_flags% %lidar_ndsm_flags% %vhr_rgb_flags%

:: Run Experiments
set /A end=%max_num_channels%
for /l %%x in (1, 1, %end%) do (
    set experiment=%test_name%%title_params%_%%x_bands
    python main.py --experiment-name !experiment! --experiment-number %%x --output-path %working_dir% --save-experiment-path "%save_experiment_path%!experiment!.json" --band-reduction-method issc --select-only-hs-bands --n-components %%x %flags% 
    move %working_dir%*.csv %working_dir%results\
    move %working_dir%*.hdf5 %working_dir%checkpoints\
    move %working_dir%*.png %working_dir%images\
    move %working_dir%*.txt %working_dir%training_summaries\
)

set experiment=%test_name%%title_params%_all_bands
python main.py --experiment-name !experiment! --experiment-number %max_num_channels% --output-path %working_dir% --save-experiment-path "%save_experiment_path%!experiment!.json" --skip-band-selection %flags% 
move %working_dir%*.csv %working_dir%results\
move %working_dir%*.hdf5 %working_dir%checkpoints\
move %working_dir%*.png %working_dir%images\
move %working_dir%*.txt %working_dir%training_summaries\


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


echo "test_fusion_issc_augmented.bat script completed!"