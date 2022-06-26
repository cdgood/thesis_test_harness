@echo off
@break off
@title Run ISSC band selection tests on all modalities
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

if not exist "training_results\test_all_modality_issc\" (
  mkdir "training_results\test_all_modality_issc\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_all_modality_issc\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_all_modality_issc\'..."
  )
) else (
  echo "'training_results\test_all_modality_issc\' already exists!"
)

if not exist "training_results\test_all_modality_issc\experiments\" (
  mkdir "training_results\test_all_modality_issc\experiments\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_all_modality_issc\experiments\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_all_modality_issc\experiments\'..."
  )
) else (
  echo "'training_results\test_all_modality_issc\experiments\' already exists!"
)

if not exist "training_results\test_all_modality_issc\images\" (
  mkdir "training_results\test_all_modality_issc\images\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_all_modality_issc\images\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_all_modality_issc\images\'..."
  )
) else (
  echo "'training_results\test_all_modality_issc\images\' already exists!"
)

if not exist "training_results\test_all_modality_issc\training_summaries\" (
  mkdir "training_results\test_all_modality_issc\training_summaries\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_all_modality_issc\training_summaries\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_all_modality_issc\training_summaries\'..."
  )
) else (
  echo "'training_results\test_all_modality_issc\training_summaries\' already exists!"
)

if not exist "training_results\test_all_modality_issc\checkpoints\" (
  mkdir "training_results\test_all_modality_issc\checkpoints\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_all_modality_issc\checkpoints\' created successfully"
  ) else (
    echo "Error while creating 'checkpoints\test_all_modality_issc\checkpoints\'..."
  )
) else (
  echo "'training_results\test_all_modality_issc\checkpoints\' already exists!"
)

if not exist "training_results\test_all_modality_issc\results\" (
  mkdir "training_results\test_all_modality_issc\results\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_all_modality_issc\results\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_all_modality_issc\results\'..."
  )
) else (
  echo "'training_results\test_all_modality_issc\results\' already exists!"
)

if not exist "training_results\test_all_modality_issc\results\individual\" (
  mkdir "training_results\test_all_modality_issc\results\individual\"
  if "!errorlevel!" == "0" (
    echo "'training_results\test_all_modality_issc\results\individual\' created successfully"
  ) else (
    echo "Error while creating 'training_results\test_all_modality_issc\results\individual\'..."
  )
) else (
  echo "'training_results\test_all_modality_issc\results\individual\' already exists!"
)

set save_experiment_path=.\training_results\test_all_modality_issc\experiments\
set params=--cuda 0 --output-path .\training_results\test_all_modality_issc --dataset grss_dfc_2018 --skip-data-postprocessing --model-id 3d-densenet-modified --random-seed 13 --epochs 100 --batch-size 16 --patch-size 13 --center-pixel --split-mode fixed --model-save-period 20 --optimizer nadam --lr 0.00005 --use-all-data --hs-resampling average --vhr-resampling cubic_spline

for /l %%x in (1, 1, 54) do (
    python main.py --experiment-name "test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts_%%x_bands" --experiment-number %%x --save-experiment-path "%save_experiment_path%test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts_%%x_bands.json" --band-reduction-method issc --n-components %%x %params% 
    move training_results\test_all_modality_issc\*.csv training_results\test_all_modality_issc\results\
    move training_results\test_all_modality_issc\*.hdf5 training_results\test_all_modality_issc\checkpoints\
    move training_results\test_all_modality_issc\*.png training_results\test_all_modality_issc\images\
    move training_results\test_all_modality_issc\*.txt training_results\test_all_modality_issc\training_summaries\
)

python main.py --experiment-name test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts_all_bands --experiment-number 48 --save-experiment-path "%save_experiment_path%test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts_all_bands.json" --skip-band-selection %params% 
move training_results\test_all_modality_issc\*.csv training_results\test_all_modality_issc\results\
move training_results\test_all_modality_issc\*.hdf5 training_results\test_all_modality_issc\checkpoints\
move training_results\test_all_modality_issc\*.png training_results\test_all_modality_issc\images\
move training_results\test_all_modality_issc\*.txt training_results\test_all_modality_issc\training_summaries\

cd training_results\test_all_modality_issc\results\
copy /b *__class_results.csv ..\test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv
move *__class_results.csv individual\

copy /b *__selected_band_results.csv ..\test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv
move *__selected_band_results.csv tindividual\

copy /b *_results.csv ..\test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv
move *_results.csv individual\

cd ..

sort /c /unique "test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv" /o "results\test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv"
sort /c /unique "test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv" /o "results\test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv"
sort /c /unique "test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv" /o "results\test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv"

del "test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__class_results.csv"
del "test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__selected_band_results.csv"
del "test_all_modality_issc_lr5e-5_e100_b16_p13_rs13_no-ts__results.csv"

cd ..\..

echo "test_all_modality_issc.bat script completed!"