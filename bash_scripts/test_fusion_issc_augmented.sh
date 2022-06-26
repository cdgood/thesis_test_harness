#!/bin/bash

# Get arguments
starting_dir="./"
start_arg=0
end_arg=0
while getopts d::s::e:: flag
do
    case "${flag}" in
        d) starting_dir=${OPTARG};;
        s) start_arg=${OPTARG};;
        e) end_arg=${OPTARG};;
    esac
done

# Create directories if they do not exist
test_name=test_fusion_issc_augmented
dataset_dir="${starting_dir}datasets/grss_dfc_2018/"
tr_dir="${starting_dir}training_results/"
working_dir="$tr_dir$test_name/"

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
epochs=100
batch_size=16
patch_size=13
lr=0.00005
save_period=$epochs
title_params="_lr5e-5_e${epochs}_b${batch_size}_p${patch_size}_rs${random_seed}_no-ts"
training_params="--random-seed ${random_seed} --epochs ${epochs} --batch-size ${batch_size} --patch-size ${patch_size} --lr ${lr} --model-save-period ${save_period}"
checkpoint_params="--random-seed ${random_seed} --batch-size ${batch_size} --patch-size ${patch_size} --lr ${lr}"

# Set flag variables
save_experiment_path="${working_dir}experiments/"
hs_flags="--use-hs-data --hs-resampling average"
lidar_ms_flags="--use-lidar-ms-data"
lidar_ndsm_flags="--use-lidar-ndsm-data"
vhr_rgb_flags="--use-vhr-data --vhr-resampling cubic_spline"
flags="--cuda 0 --dataset grss_dfc_2018 --path-to-dataset ${dataset_dir} --skip-data-postprocessing --model-id 3d-densenet-fusion --center-pixel --split-mode fixed --optimizer nadam --flip-augmentation --radiation-augmentation --mixture-augmentation ${training_params} ${hs_flags} ${lidar_ms_flags} ${lidar_ndsm_flags} ${vhr_rgb_flags}"

# Run experiments
start=1
if [ $start_arg -gt 0 ]; then
    start=$start_arg  
fi

end=$(($max_num_channels + 1))
if [ $end_arg -gt 0 ]; then
    end=$end_arg
fi

for (( i=$start ; i<$end ; i++ ))
do
    if [ $i -lt $max_num_channels ]; then
        experiment="${test_name}${title_params}_${i}_bands"
        python main.py --experiment-name $experiment --experiment-number $i --output-path $working_dir --save-experiment-path "${save_experiment_path}${experiment}.json" --band-reduction-method issc --select-only-hs-bands --n-components $i $flags
    elif [ $i -eq $max_num_channels ]; then
        experiment="${test_name}${title_params}_all_bands"
        python main.py --experiment-name $experiment --experiment-number $max_num_channels --output-path $working_dir --save-experiment-path "${save_experiment_path}${experiment}.json" --skip-band-selection $flags
    else
        break
    fi

    mv "${working_dir}"*.csv "${working_dir}results/"
    mv "${working_dir}"*.hdf5 "${working_dir}checkpoints/"
    mv "${working_dir}"*.png "${working_dir}images/"
    mv "${working_dir}"*.txt "${working_dir}training_summaries/"
done


# Move and combine results files
cat "${working_dir}results"/*__class_results.csv >> "${working_dir}results/${test_name}${title_params}__class_results_tmp.csv"
mv "${working_dir}results"/*__class_results.csv "${working_dir}results/individual/"
sort "${working_dir}results/${test_name}${title_params}__class_results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}${title_params}__class_results.csv"
rm "${working_dir}results/${test_name}${title_params}__class_results_tmp.csv"

cat "${working_dir}results"/*__selected_band_results.csv >> "${working_dir}results/${test_name}${title_params}__selected_band_results_tmp.csv"
mv "${working_dir}results"/*__selected_band_results.csv "${working_dir}results/individual/"
sort "${working_dir}results/${test_name}${title_params}__selected_band_results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}${title_params}__selected_band_results.csv"
rm "${working_dir}results/${test_name}${title_params}__selected_band_results_tmp.csv"

cat "${working_dir}results"/*_results.csv >> "${working_dir}results/${test_name}${title_params}__results_tmp.csv"
mv "${working_dir}results"/*_results.csv "${working_dir}results/individual/"
sort "${working_dir}results/${test_name}${title_params}__results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}${title_params}__results.csv"
rm "${working_dir}results/${test_name}${title_params}__results_tmp.csv"

echo "test_fusion_issc_augmented.sh script completed!"