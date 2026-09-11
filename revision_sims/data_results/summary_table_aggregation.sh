#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:-$SCRIPT_DIR/results_mc}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/summary_tables_$(date +%Y%m%d_%H%M%S)}"
DATE_STRING="${3:?Usage: $0 [results_dir] [output_dir] date_string}"

if [[ ! -d "$RESULTS_DIR" ]]; then
	printf 'Results directory does not exist: %s\n' "$RESULTS_DIR" >&2
	exit 1
fi

mkdir -p "$OUTPUT_DIR"

copied=0
while IFS= read -r -d '' summary_file; do
	run_dir="$(basename -- "$(dirname -- "$summary_file")")"
	parameter_dir="$(basename -- "$(dirname -- "$(dirname -- "$summary_file")")")"

	if [[ "$run_dir" != "$DATE_STRING"* ]]; then
		continue
	fi

	if [[ "$parameter_dir" != ls_*_lt_* ]]; then
		printf 'Skipping unexpected directory: %s\n' "$summary_file" >&2
		continue
	fi

	output_name="summary_${parameter_dir}_${run_dir}.txt"
	cp -- "$summary_file" "$OUTPUT_DIR/$output_name"
	copied=$((copied + 1))
done < <(find "$RESULTS_DIR" -type f \( -name 'summary.txt' -o -name 'summary_table.txt' \) -print0)

printf 'Copied %d summary file(s) to %s\n' "$copied" "$OUTPUT_DIR"
