#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

# List of object names
# cracker_box_flipped
# master_chef_can_flipped
# mustard_bottle_flipped
# banana
# letter_t
objs=(
    # trash_truck
    # school_bus
    mug
)

for obj in "${objs[@]}"; do
    echo "=== Processing $obj ==="

    # Start collect_data.py in the background
    python "$script_dir/collect_data.py" "$obj" &
    wait

    echo "=== Done with $obj ==="
    echo
done

wait
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All collect data jobs completed in %02dh %02dm %02ds ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
