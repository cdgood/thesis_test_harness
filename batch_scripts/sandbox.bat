@echo off
@break off

setlocal EnableDelayedExpansion

@REM set epochs=200
@REM set save_period=10

@REM for /l %%y in (%save_period%, %save_period%, %epochs%) do (
@REM     if %%y GEQ 10000000 ( set "leading_zeros="
@REM     ) else ( if %%y GEQ 1000000 ( set "leading_zeros=0"
@REM     ) else ( if %%y GEQ 100000 ( set "leading_zeros=00"
@REM     ) else ( if %%y GEQ 10000 ( set "leading_zeros=000"
@REM     ) else ( if %%y GEQ 1000 ( set "leading_zeros=0000"
@REM     ) else ( if %%y GEQ 100 ( set "leading_zeros=00000"
@REM     ) else ( if %%y GEQ 10 ( set "leading_zeros=000000"
@REM     ) else ( set "leading_zeros=0000000")))))))

@REM     echo checkpoint_!leading_zeros!%%y.hdf5
@REM )

@REM for /l %%y in (%save_period%, %save_period%, %epochs%) do (
    
@REM     set "leading_zeros="

@REM     if %%y LSS 10000000 set "leading_zeros=!leading_zeros!0"
@REM     if %%y LSS 1000000 set "leading_zeros=!leading_zeros!0"
@REM     if %%y LSS 100000 set "leading_zeros=!leading_zeros!0"
@REM     if %%y LSS 10000 set "leading_zeros=!leading_zeros!0"
@REM     if %%y LSS 1000 set "leading_zeros=!leading_zeros!0"
@REM     if %%y LSS 100 set "leading_zeros=!leading_zeros!0"
@REM     if %%y LSS 10 set "leading_zeros=!leading_zeros!0"

@REM     echo checkpoint_!leading_zeros!%%y.hdf5
@REM )

@REM set /A end=%epochs% - 1
@REM for /l %%i in (1, 1, %end%) do (
@REM     echo %%i
@REM )

@REM set "title=test_hs_lr5e-5_e200_b16_p13_rs13_no-ts_checkpoint"
@REM set "working_dir=.\training_results\test_each_modality_and_checkpoints\checkpoints\test_hs_lr5e-5_e200_b16_p13_rs13_no-ts\"

@REM copy /b %working_dir%*__class_results.csv %working_dir%%title%__class_results_tmp.csv
@REM move %working_dir%*__class_results.csv %working_dir%individual\

@REM copy /b %working_dir%*_results.csv %working_dir%%title%__results_tmp.csv
@REM move %working_dir%*_results.csv %working_dir%individual\


@REM sort /c /unique %working_dir%%title%__class_results_tmp.csv /o %working_dir%%title%__class_results.csv
@REM del %working_dir%%title%__class_results_tmp.csv

@REM sort /c /unique %working_dir%%title%__results_tmp.csv /o %working_dir%%title%__results.csv
@REM del %working_dir%%title%__results_tmp.csv


:: Create directories if they do not exist
set test_name=sandbox
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
set max_num_channels=2
set random_seed=13
set epochs=5
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
set flags=--cuda 0 --dataset grss_dfc_2018 --skip-data-postprocessing --model-id 3d-densenet-modified --center-pixel --split-mode fixed --optimizer nadam %training_params% %hs_flags%

:: Run Experiments
set /A end=%max_num_channels%
for /l %%x in (1, 1, %end%) do (
    if %%x LSS %max_num_channels% (
        set experiment=%test_name%%title_params%_%%x_bands
        python main.py --experiment-name !experiment! --experiment-number %%x --output-path %working_dir% --save-experiment-path "%save_experiment_path%!experiment!.json" --band-reduction-method issc --n-components %%x %flags% 
    ) else ( if %%x EQU %max_num_channels% (
        set experiment=%test_name%%title_params%_all_bands
        python main.py --experiment-name !experiment! --experiment-number %%x --output-path %working_dir% --save-experiment-path "%save_experiment_path%!experiment!.json" --skip-band-selection %flags% 
    ))
    
    
    if %%x LEQ %max_num_channels% (
        move %working_dir%*.csv %working_dir%results\
        move %working_dir%*.hdf5 %working_dir%checkpoints\
        move %working_dir%*.png %working_dir%images\
        move %working_dir%*.txt %working_dir%training_summaries\
    )
    
)
