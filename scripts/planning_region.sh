#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

# List of object name
objs=(
    mustard_bottle_flipped
    real_mustard_bottle_flipped
)
model_learning_types=(
    mlp_random
    residual_bait
)
n_datas=(
    10
    20
    30
    40
    50
    60
    70
    80
    90
    100
)
active_sampling=(
    0
    1
)
n_reps=5
predef_controls=1

for obj in "${objs[@]}"; do
    for ml_type in "${model_learning_types[@]}"; do
        model_type=$(echo "$ml_type" | cut -d'_' -f1)
        learning_type=$(echo "$ml_type" | cut -d'_' -f2)
        # Active sampling or not
        for sampling in "${active_sampling[@]}"; do

            for n_data in "${n_datas[@]}"; do
                # Run plnning.py in the background
                echo "=== Planning $obj $n_reps times with $model_type $learning_type $n_data $sampling $predef_controls ==="
                python "$script_dir/planning_region.py" "$obj" "$model_type" "$learning_type" "$n_data" $sampling $predef_controls $n_reps &
                echo
            done
            wait
        done
    done
done

wait
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All planning jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
