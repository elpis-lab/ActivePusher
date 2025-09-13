#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

object_names=(
    cracker_box_flipped
    mustard_bottle_flipped
    banana
    mug
    real_cracker_box_flipped
    real_mustard_bottle_flipped
)
model_classes=(
    mlp
    residual
)
n_experiments=5
n_query=100
batch_size=10
devices=(
    cuda:0
    # cuda:3
)

# Repeat experiment
device_idx=0
for object_name in ${object_names[@]}; do
    for model_class in ${model_classes[@]}; do
        device=${devices[$device_idx]}

        echo "Training $object_name with $model_class on $device"

        python "$script_dir/train_active.py" $object_name $model_class $n_experiments $n_query $batch_size $device &
        wait
        echo
    done
done

wait
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All training jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
