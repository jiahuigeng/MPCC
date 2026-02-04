#!/bin/bash

# Create results directory
mkdir -p results

# Define tasks and difficulties
TASKS=("Calendar Planning" "Meeting Planning" "Flight Planning")
DIFFICULTIES=("easy" "medium" "hard")

# Map task names to file prefixes
declare -A FILE_PREFIXES
FILE_PREFIXES["Calendar Planning"]="calendar_plan"
FILE_PREFIXES["Meeting Planning"]="meeting_plan"
FILE_PREFIXES["Flight Planning"]="flight_plan"

for task in "${TASKS[@]}"; do
    prefix=${FILE_PREFIXES[$task]}
    # Clean task name for output filename (replace space with underscore, lowercase)
    task_slug=$(echo "$task" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    
    for diff in "${DIFFICULTIES[@]}"; do
        parquet_file="MPCC_HF/${task}/${prefix}_${diff}.parquet"
        # We pass a csv path to --save, the script handles it
        output_file="results/${task_slug}_${diff}_baseline.csv"
        
        echo "==================================================="
        echo "Running evaluation for: $task ($diff)"
        echo "Input: $parquet_file"
        echo "Output: $output_file"
        echo "==================================================="
        
        if [ -f "$parquet_file" ]; then
            python eval_mpcc_all.py \
                --parquet "$parquet_file" \
                --limit 100 \
                --save "$output_file" \
                --model openai
        else
            echo "Error: File not found: $parquet_file"
        fi
        echo ""
    done
done
