#!/bin/bash

# Get arguments
starting_dir="./"
start_arg=0
end_arg=0
epoch_arg=0
batch_arg=0
patch_arg=0
lr_arg=0
random_seed_arg=0
bs_method_arg=""
use_augmentation=0
while getopts d:s:f:e:b:p:l:r:m:a flag
do
    case "${flag}" in
        d) starting_dir=${OPTARG};;
        s) start_arg=${OPTARG};;
        f) end_arg=${OPTARG};;
        e) epoch_arg=${OPTARG};;
        b) batch_arg=${OPTARG};;
        p) patch_arg=${OPTARG};;
        l) lr_arg=${OPTARG};;
        r) random_seed_arg=${OPTARG};;
        m) bs_method_arg=${OPTARG};;
        a) use_augmentation=1;;
    esac
done

# Determine band selection method
band_selection_method="pca"
if [ "$bs_method_arg" == "" ]; then
    band_selection_method=$bs_method_arg  
fi

# Create directories if they do not exist
if [ use_augmentation -eq 0]; then
    test_name="test_hs_${band_selection_method}"
else
    test_name="test_hs_${band_selection_method}_augmented"
fi
tr_dir="${starting_dir}/training_results/"
working_dir="$tr_dir$test_name/"
dataset_dir="${starting_dir}datasets/grss_dfc_2018/"

dirs=("$tr_dir")
dirs+=("$working_dir")
dirs+=("${working_dir}experiments/")
dirs+=("${working_dir}images/")
dirs+=("${working_dir}training_summaries/")
dirs+=("${working_dir}checkpoints/")
dirs+=("${working_dir}results/")
dirs+=("${working_dir}results/individual/")

num_dirs=${#dirs[@]}

for dir in "${dirs[@]}"
do
    mkdir -p "${dir}"
done


# Set parameter values
max_num_channels=55

random_seed=13
if [ $random_seed_arg -gt 0 ]; then
    random_seed=$random_seed_arg  
fi

epochs=100
if [ $epoch_arg -gt 0 ]; then
    epochs=$epoch_arg  
fi

batch_size=16
if [ $batch_arg -gt 0 ]; then
    batch_size=$batch_arg  
fi

patch_size=13
if [ $patch_arg -gt 0 ]; then
    patch_size=$patch_arg  
fi

lr=0.00005
if [ $lr_arg -ne 0 ]; then
    lr=$lr_arg  
fi

save_period=$epochs
title_params="_lr5e-5_${epochs}_b${batch_size}_p${patch_size}_rs${random_seed}_no-ts"
training_params="--random-seed ${random_seed} --epochs ${epochs} --batch-size ${batch_size} --patch-size ${patch_size} --lr ${lr} --model-save-period ${save_period}"
checkpoint_params="--random-seed ${random_seed} --batch-size ${batch_size} --patch-size ${patch_size} --lr ${lr}"

# Set flag variables
save_experiment_path="${working_dir}experiments/"
hs_flags="--use-hs-data --hs-resampling average"
lidar_ms_flags="--use-lidar-ms-data"
lidar_ndsm_flags="--use-lidar-ndsm-data"
vhr_rgb_flags="--use-vhr-data --vhr-resampling cubic_spline"
augmentation_flags="--flip-augmentation --radiation-augmentation --mixture-augmentation"
if [ use_augmentation -eq 0]; then
    flags="--cuda 0 --dataset grss_dfc_2018 --path-to-dataset ${dataset_dir} --skip-data-postprocessing --model-id 3d-densenet --center-pixel --split-mode fixed --optimizer nadam  ${training_params} ${hs_flags} ${lidar_ms_flags} ${lidar_ndsm_flags} ${vhr_rgb_flags}"
else
    flags="--cuda 0 --dataset grss_dfc_2018 --path-to-dataset ${dataset_dir} --skip-data-postprocessing --model-id 3d-densenet --center-pixel --split-mode fixed --optimizer nadam ${augmentation_flags} ${training_params} ${hs_flags} ${lidar_ms_flags} ${lidar_ndsm_flags} ${vhr_rgb_flags}"
fi

# Run experiments
start=1
if [ $start_arg -gt 0 ]; then
    start=$start_arg  
fi

end=$max_num_channels + 1
if [ $end_arg -gt 0 ]; then
    end=$end_arg
fi

for (( i=$start ; i<$end ; i++ ))
do
    if [ $i -lt $max_num_channels ]; then
        experiment="${test_name}${title_params}_${i}_bands"
        python main.py --experiment-name $experiment --experiment-number $i --output-path $working_dir --save-experiment-path "${save_experiment_path}${experiment}.json" --band-reduction-method $band_selection_method --n-components $i $flags
    elif [ $i -eq $max_num_channels ]; then
        experiment="${test_name}${title_params}_all_bands"
        python main.py --experiment-name $experiment --experiment-number $max_num_channels --output-path $working_dir --save-experiment-path "${save_experiment_path}${experiment}.json" --skip-band-selection $flags
    else
        break
    fi

    mv "${working_dir}*.csv" "${working_dir}results/"
    mv "${working_dir}*.hdf5" "${working_dir}checkpoints/"
    mv "${working_dir}*.png" "${working_dir}images/"
    mv "${working_dir}*.txt" "${working_dir}training_summaries/"
done


# Move and combine results files
cat "${working_dir}results/*__class_results.csv" >> "${working_dir}results/${test_name}${title_params}__class_results_tmp.csv"
mv "${working_dir}results/*__class_results.csv" "${working_dir}results/individual/"
sort "${working_dir}results/${test_name}${title_params}__class_results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}${title_params}__class_results.csv"
rm "${working_dir}results/${test_name}${title_params}__class_results_tmp.csv"

cat "${working_dir}results/*__selected_band_results.csv" >> "${working_dir}results/${test_name}${title_params}__selected_band_results_tmp.csv"
mv "${working_dir}results/*__selected_band_results.csv" "${working_dir}results/individual/"
sort "${working_dir}results/${test_name}${title_params}__selected_band_results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}${title_params}__selected_band_results.csv"
rm "${working_dir}results/${test_name}${title_params}__selected_band_results_tmp.csv"

cat "${working_dir}results/*_results.csv" >> "${working_dir}results/${test_name}${title_params}__results_tmp.csv"
mv "${working_dir}results/*_results.csv" "${working_dir}results/individual/"
sort "${working_dir}results/${test_name}${title_params}__results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}${title_params}__results.csv"
rm "${working_dir}results/${test_name}${title_params}__results_tmp.csv"

echo "test_fusion_issc_augmented.sh script completed!"