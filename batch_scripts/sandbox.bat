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

set "title=test_fusion_issc_augmented"
set "working_dir=.\training_results\test_fusion_issc_augmented\results\"

copy /b %working_dir%*__class_results.csv %working_dir%%title%__class_results_tmp.csv
move %working_dir%*__class_results.csv %working_dir%individual\

copy /b %working_dir%*__selected_band_results.csv %working_dir%%title%__selected_band_results_tmp.csv
move %working_dir%*__selected_band_results.csv %working_dir%\individual\

copy /b %working_dir%*_results.csv %working_dir%%title%__results_tmp.csv
move %working_dir%*_results.csv %working_dir%individual\


sort /c /unique %working_dir%%title%__selected_band_results_tmp.csv /o %working_dir%%title%__selected_band_results.csv
del %working_dir%%title%__selected_band_results_tmp.csv

sort /c /unique %working_dir%%title%__class_results_tmp.csv /o %working_dir%%title%__class_results.csv
del %working_dir%%title%__class_results_tmp.csv

sort /c /unique %working_dir%%title%__results_tmp.csv /o %working_dir%%title%__results.csv
del %working_dir%%title%__results_tmp.csv


:: Create directories if they do not exist
@REM set test_name=sandbox
@REM set tr_dir=.\training_results\
@REM set working_dir=%tr_dir%%test_name%\

@REM set dir[0]=%tr_dir%
@REM set dir[1]=%working_dir%
@REM set dir[2]=%working_dir%experiments\
@REM set dir[3]=%working_dir%images\
@REM set dir[4]=%working_dir%training_summaries\
@REM set dir[5]=%working_dir%checkpoints\
@REM set dir[6]=%working_dir%results\
@REM set dir[7]=%working_dir%results\individual\

@REM set num_dirs=7

@REM for /l %%d in (0, 1, %num_dirs%) do (
@REM     if not exist "!dir[%%d]!" (
@REM         mkdir "!dir[%%d]!"
@REM         if "!errorlevel!" == "0" (
@REM             echo "'!dir[%%d]!' created successfully"
@REM         ) else (
@REM             echo "Error while creating '!dir[%%d]!'..."
@REM         )
@REM     ) else (
@REM         echo "'!dir[%%d]!' already exists!"
@REM     )
@REM )


@REM :: Set parameter values
@REM set max_num_channels=2
@REM set random_seed=13
@REM set epochs=5
@REM set batch_size=16
@REM set patch_size=13
@REM set lr=0.00005
@REM set save_period=%epochs%
@REM set title_params=_lr5e-5_e%epochs%_b%batch_size%_p%patch_size%_rs%random_seed%_no-ts
@REM set training_params=--random-seed %random_seed% --epochs %epochs% --batch-size %batch_size% --patch-size %patch_size% --lr %lr% --model-save-period %save_period%
@REM set checkpoint_params=--random-seed %random_seed% --batch-size %batch_size% --patch-size %patch_size% --lr %lr%

@REM :: Set flag list variables
@REM set save_experiment_path=%working_dir%experiments\
@REM set hs_flags=--use-hs-data --hs-resampling average
@REM set lidar_ms_flags=--use-lidar-ms-data
@REM set lidar_ndsm_flags=--use-lidar-ndsm-data
@REM set vhr_rgb_flags=--use-vhr-data --vhr-resampling cubic_spline
@REM set flags=--cuda 0 --dataset grss_dfc_2018 --skip-data-postprocessing --model-id 3d-densenet-modified --center-pixel --split-mode fixed --optimizer nadam %training_params% %hs_flags%

@REM :: Run Experiments
@REM set /A end=%max_num_channels%
@REM for /l %%x in (1, 1, %end%) do (
@REM     if %%x LSS %max_num_channels% (
@REM         set experiment=%test_name%%title_params%_%%x_bands
@REM         python main.py --experiment-name !experiment! --experiment-number %%x --output-path %working_dir% --save-experiment-path "%save_experiment_path%!experiment!.json" --band-reduction-method issc --n-components %%x %flags% 
@REM     ) else ( if %%x EQU %max_num_channels% (
@REM         set experiment=%test_name%%title_params%_all_bands
@REM         python main.py --experiment-name !experiment! --experiment-number %%x --output-path %working_dir% --save-experiment-path "%save_experiment_path%!experiment!.json" --skip-band-selection %flags% 
@REM     ))
    
    
@REM     if %%x LEQ %max_num_channels% (
@REM         move %working_dir%*.csv %working_dir%results\
@REM         move %working_dir%*.hdf5 %working_dir%checkpoints\
@REM         move %working_dir%*.png %working_dir%images\
@REM         move %working_dir%*.txt %working_dir%training_summaries\
@REM     )
    
@REM )
