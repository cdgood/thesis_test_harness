@echo off
@break off
@title Run hyperspectral ISSC band selection tests with data augmentation
@color 0a
@cls

setlocal EnableDelayedExpansion

if not exist "training_results\" (
  mkdir "training_results\"
  if "!errorlevel!" == "0" (
    echo "'training_results\' created successfully"
  ) else (
    echo "Error while creating 'training_results\'..."
  )
) else (
  echo "'training_results\' already exists!"
)

if not exist "training_results\test_hs_issc_augmented\" (
  mkdir "training_results\test_hs_issc_augmented\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_hs_issc_augmented\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_hs_issc_augmented\'..."
  )
) else (
  echo "'training_results\test_hs_issc_augmented\' already exists!"
)

if not exist "training_results\test_hs_issc_augmented\experiments\" (
  mkdir "training_results\test_hs_issc_augmented\experiments\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_hs_issc_augmented\experiments\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_hs_issc_augmented\experiments\'..."
  )
) else (
  echo "'training_results\test_hs_issc_augmented\experiments\' already exists!"
)

if not exist "training_results\test_hs_issc_augmented\images\" (
  mkdir "training_results\test_hs_issc_augmented\images\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_hs_issc_augmented\images\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_hs_issc_augmented\images\'..."
  )
) else (
  echo "'training_results\test_hs_issc_augmented\images\' already exists!"
)

if not exist "training_results\test_hs_issc_augmented\training_summaries\" (
  mkdir "training_results\test_hs_issc_augmented\training_summaries\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_hs_issc_augmented\training_summaries\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_hs_issc_augmented\training_summaries\'..."
  )
) else (
  echo "'training_results\test_hs_issc_augmented\training_summaries\' already exists!"
)

if not exist "training_results\test_hs_issc_augmented\checkpoints\" (
  mkdir "training_results\test_hs_issc_augmented\checkpoints\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_hs_issc_augmented\checkpoints\' created successfully"
  ) else (
    echo "Error while creating 'checkpoints\test_hs_issc_augmented\checkpoints\'..."
  )
) else (
  echo "'training_results\test_hs_issc_augmented\checkpoints\' already exists!"
)

if not exist "training_results\test_hs_issc_augmented\results\" (
  mkdir "training_results\test_hs_issc_augmented\results\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_hs_issc_augmented\results\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_hs_issc_augmented\results\'..."
  )
) else (
  echo "'training_results\test_hs_issc_augmented\results\' already exists!"
)

if not exist "training_results\test_hs_issc_augmented\results\individual\" (
  mkdir "training_results\test_hs_issc_augmented\results\individual\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_hs_issc_augmented\results\individual\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_hs_issc_augmented\results\individual\'..."
  )
) else (
  echo "'training_results\test_hs_issc_augmented\results\individual\' already exists!"
)

set save_experiment_path=.\training_results\test_hs_issc_augmented\experiments\
set params=--cuda 0 --output-path .\training_results\test_hs_issc_augmented --dataset grss_dfc_2018 --skip-data-postprocessing --model-id 3d-densenet-modified --random-seed 13 --epochs 100 --batch-size 16 --patch-size 13 --center-pixel --split-mode fixed --model-save-period 20 --optimizer nadam --lr 0.00005 --flip-augmentation --radiation-augmentation --mixture-augmentation --use-hs-data --hs-resampling average

for /l %%x in (1, 1, 47) do (
    python main.py --experiment-name "test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts_%%x_bands" --experiment-number %%x --save-experiment-path "%save_experiment_path%test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts_%%x_bands.json" --band-reduction-method issc --n-components %%x %params% 
    move training_results\test_hs_issc_augmented\*.csv training_results\test_hs_issc_augmented\results\
    move training_results\test_hs_issc_augmented\*.hdf5 training_results\test_hs_issc_augmented\checkpoints\
    move training_results\test_hs_issc_augmented\*.png training_results\test_hs_issc_augmented\images\
    move training_results\test_hs_issc_augmented\*.txt training_results\test_hs_issc_augmented\training_summaries\
)

python main.py --experiment-name test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts_all_bands --experiment-number 48 --save-experiment-path "%save_experiment_path%test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts_all_bands.json" --skip-band-selection %params% 
move training_results\test_hs_issc_augmented\*.csv training_results\test_hs_issc_augmented\results\
move training_results\test_hs_issc_augmented\*.hdf5 training_results\test_hs_issc_augmented\checkpoints\
move training_results\test_hs_issc_augmented\*.png training_results\test_hs_issc_augmented\images\
move training_results\test_hs_issc_augmented\*.txt training_results\test_hs_issc_augmented\training_summaries\

cd training_results\test_hs_issc_augmented\results\
copy /b *__class_results.csv ..\test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv
move *__class_results.csv individual\

copy /b *__selected_band_results.csv ..\test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv
move *__selected_band_results.csv individual\

copy /b *_results.csv ..\test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv
move *_results.csv individual\

cd ..

sort /c /unique "test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv" /o "results\test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv"
sort /c /unique "test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv" /o "results\test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv"
sort /c /unique "test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv" /o "results\test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv"

del "test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv"
del "test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv"
del "test_hs_issc_augmented_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv"

cd ..\..

echo "test_hs_issc_augmented.bat script completed!"