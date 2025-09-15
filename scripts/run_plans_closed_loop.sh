#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

# List of object name
objs=(
    mustard_bottle_flipped
    cracker_box_flipped
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

for obj in "${objs[@]}"; do
    for ml_type in "${model_learning_types[@]}"; do
        model_type=$(echo "$ml_type" | cut -d'_' -f1)
        learning_type=$(echo "$ml_type" | cut -d'_' -f2)
        for sampling in "${active_sampling[@]}"; do

            for n_data in "${n_datas[@]}"; do
                # Run plnning.py in the background
                echo "=== Planning (Closed Loop) $obj with $model_type $learning_type $sampling $n_data ==="
                python "$script_dir/run_plans_closed_loop.py" "$obj" "$model_type" "$learning_type" "$n_data" $sampling &
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
