#!/bin/bash
set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Array of all algorithm scripts in the order you want to run them
algorithms=(
    # "run_gae.sh"
    # "run_eob.sh"
    "run_optimal_token_baseline"
    # "run_opo.sh"
    "run_reinforce_plus_plus.sh"
    "run_reinforce_plus_plus_baseline.sh"
    "run_remax.sh"
    "run_rloo.sh"
    "run_grpo_passk.sh"
    "run_gpg.sh"
    "run_rloo_vectorized.sh"
    "run_grpo_vectorized.sh"
)

# Log file to track progress
LOG_FILE="$SCRIPT_DIR/all_algorithms_$(date +%Y%m%d_%H%M%S).log"

echo "Starting sequential algorithm runs at $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# Run each algorithm sequentially
for algo in "${algorithms[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "Starting $algo at $(date)" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"

    # Run the algorithm script
    bash "$SCRIPT_DIR/$algo" 2>&1 | tee -a "$LOG_FILE"

    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ $algo completed successfully at $(date)" | tee -a "$LOG_FILE"
    else
        echo "✗ $algo failed at $(date)" | tee -a "$LOG_FILE"
        # You can choose to continue or stop on failure
        # To stop on failure, uncomment the next line:
        # exit 1
    fi

    echo "=============================================" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "All algorithms completed at $(date)" | tee -a "$LOG_FILE"
