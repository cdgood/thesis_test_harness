#!/bin/bash

# Get arguments
starting_dir="./"
start_arg=0
end_arg=0
begin_run_arg=0
bs_method_arg=""
while getopts d::s::f::b::m:: flag
do
    case "${flag}" in
        d) starting_dir=${OPTARG};;
        s) start_arg=${OPTARG};;
        f) end_arg=${OPTARG};;
        b) begin_run_arg=${OPTARG};;
        m) bs_method_arg=${OPTARG};;
    esac
done

# Determine band selection method
band_selection_method="pca"
if [ "$bs_method_arg" == "" ]; then
    band_selection_method=$bs_method_arg  
fi

# Create directories if they do not exist
test_name="test_hs_${band_selection_method}"
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
epochs=100
batch_size=16
patch_size=13
lr=0.00005
save_period=$epochs
runs=1
if [ $runs_arg -gt 0 ]; then
    runs=$runs_arg  
fi
# title_params="_lr5e-5_${epochs}_b${batch_size}_p${patch_size}_rs${random_seed}_no-ts"
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
    flags="--cuda 0 --dataset grss_dfc_2018 --path-to-dataset ${dataset_dir} --skip-data-postprocessing --model-id 3d-densenet --center-pixel --split-mode fixed --optimizer nadam  ${training_params} ${hs_flags}"
else
    flags="--cuda 0 --dataset grss_dfc_2018 --path-to-dataset ${dataset_dir} --skip-data-postprocessing --model-id 3d-densenet --center-pixel --split-mode fixed --optimizer nadam ${augmentation_flags} ${training_params} ${hs_flags}"
fi

# Run experiments
start=1
if [ $start_arg -gt 0 ]; then
    start=$start_arg  
fi

end=$num_experiments
if [ $end_arg -gt 0 ]; then
    end=$end_arg
fi

begin_run=0
if [ $begin_run_arg -gt 0 ]; then
    begin_run=$begin_run_arg
fi

echo "start: ${start}"
echo "end:   ${end}"

for (( i=$start ; i<$end ; i++ ))
do
    

    mv "${working_dir}*.csv" "${working_dir}results/"
    mv "${working_dir}*.hdf5" "${working_dir}checkpoints/"
    mv "${working_dir}*.png" "${working_dir}images/"
    mv "${working_dir}*.txt" "${working_dir}training_summaries/"

    if [ $i -gt 0 ]; then
        for (( j=1 ; j<=$runs ; j++ ))
        do
            if [[ $i -ne $start || $j -ge $begin_run ]]; then

                experiment_num=$(( ( $i - 1 ) * $runs + $j ))

                if [ $i -lt $max_num_channels ]; then
                    experiment="${test_name}_${i}_bands_run$j"
                    python main.py --experiment-name $experiment --experiment-number $experiment_num --output-path $working_dir --save-experiment-path "${save_experiment_path}${experiment}.json" --band-reduction-method $band_selection_method --n-components $i $flags
                elif [ $i -eq $max_num_channels ]; then
                    experiment="${test_name}_all_bands_run$j"
                    python main.py --experiment-name $experiment --experiment-number $max_num_channels --output-path $working_dir --save-experiment-path "${save_experiment_path}${experiment}.json" --skip-band-selection $flags
                else
                    break
                fi

                mv "${working_dir}"*.csv "${working_dir}results/"
                mv "${working_dir}"*.hdf5 "${working_dir}checkpoints/"
                mv "${working_dir}"*.png "${working_dir}images/"
                mv "${working_dir}"*.txt "${working_dir}training_summaries/"
            fi
        done
    fi
done


# Move and combine results files
cat "${working_dir}results"/*__class_results.csv >> "${working_dir}results/${test_name}__class_results_tmp.csv"
mv "${working_dir}results"/*__class_results.csv "${working_dir}results/individual/"

cat "${working_dir}results"/*__selected_band_results.csv >> "${working_dir}results/${test_name}__selected_band_results_tmp.csv"
mv "${working_dir}results"/*__selected_band_results.csv "${working_dir}results/individual/"

cat "${working_dir}results"/*_results.csv >> "${working_dir}results/${test_name}__results_tmp.csv"
mv "${working_dir}results"/*_results.csv "${working_dir}results/individual/"

echo "Creating final results files..."

sort "${working_dir}results/${test_name}__class_results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}__class_results.csv"
rm "${working_dir}results/${test_name}__class_results_tmp.csv"

sort "${working_dir}results/${test_name}__selected_band_results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}__selected_band_results.csv"
rm "${working_dir}results/${test_name}__selected_band_results_tmp.csv"

sort "${working_dir}results/${test_name}__results_tmp.csv" | uniq -u >> "${working_dir}results/${test_name}__results.csv"
rm "${working_dir}results/${test_name}__results_tmp.csv"

echo "test_fusion_issc_augmented.sh script completed!"